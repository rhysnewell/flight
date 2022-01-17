#!/usr/bin/env python
###############################################################################
# binning.py - A binning algorithm spinning off of the methodology of
#              Lorikeet
###############################################################################
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the GNU General Public License as published by        #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                #
# GNU General Public License for more details.                                #
#                                                                             #
# You should have received a copy of the GNU General Public License           #
# along with this program. If not, see <http://www.gnu.org/licenses/>.        #
#                                                                             #
###############################################################################

__author__ = "Rhys Newell"
__copyright__ = "Copyright 2020"
__credits__ = ["Rhys Newell"]
__license__ = "GPL3"
__maintainer__ = "Rhys Newell"
__email__ = "rhys.newell near hdr.qut.edu.au"
__status__ = "Development"

import logging
###############################################################################
# System imports
import threadpoolctl
import warnings
import matplotlib

# Function imports
import numpy as np
import seaborn as sns
from numba import njit, set_num_threads

# self imports
import flight.utils as utils
from flight.rosella.validating import Validator
import flight.distance as distance
import faulthandler

faulthandler.enable()

# Set plotting style
sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
matplotlib.use('pdf')

# Debug
debug = {
    1: logging.CRITICAL,
    2: logging.ERROR,
    3: logging.WARNING,
    4: logging.INFO,
    5: logging.DEBUG
}

###############################################################################
############################### - Exceptions - ################################


class BadTreeFileException(Exception):
    pass

###############################################################################                                                                                                                      [44/1010]
################################ - Functions - ################################

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

###############################################################################
################################ - Classes - ##################################


class Rosella(Validator):
    """
    Highest level child class for the Rosella binning pipeline. Enacts calls to lower level
    classes to generate bin set.
    """
    def quick_filter(
            self,
            plots,
            n=0,
            n_max=1,
            x_min=20,
            x_max=20,
            y_min=20,
            y_max=20
        ):
        while n <= n_max:
            plots, n = self.validate_bins(plots, n,
                                          x_min, x_max,
                                          y_min, y_max,
                                          quick_filter=True, big_only=True)
            self.sort_bins()
            n += 1

    def slow_refine(
            self,
            plots,
            n=0,
            n_max=100,
            x_min=20,
            x_max=20,
            y_min=20,
            y_max=20,
            large_bins_only=False,
    ):
        # Each cluster is checked for internal metrics. If the metrics look bad then
        # Recluster -> re-embed -> recluster. If any of the new clusters look better
        # Than the previous clusters then take them instead (Within reason).
        # Kind of time consuming, could potentially be sped up with multiprocessing
        # but thread control might get a bit heckers.
        while n <= n_max:
            self.overclustered = False  # large clusters
            plots, n = self.validate_bins(plots, n,
                                          x_min, x_max,
                                          y_min, y_max,
                                          reembed=True,
                                          large_bins_only=large_bins_only)
            self.sort_bins()
            n += 1
            if not self.overclustered:
                break  # no more clusters have broken

    def big_contig_filter(
            self,
            plots,
            n=0,
            n_max=5,
            x_min=20,
            x_max=20,
            y_min=20,
            y_max=20
    ):
        while n <= n_max:
            plots, n = self.validate_bins(plots, n, x_min, x_max, y_min, y_max,
                                          big_only=True)
            self.sort_bins()
            # plots, n = self.validate_bins(plots, n,
            #                               x_min, x_max,
            #                               y_min, y_max,
            #                               reembed=True)
            # self.sort_bins()

            n += 1

    def perform_limited_binning(self, unbinned):
        labels, _, _, _ = self.ensemble_cluster_multiple_embeddings(
            [self.precomputed_reducer_low.embedding_[unbinned],
             # self.precomputed_reducer_mid.embedding_,
             self.precomputed_reducer_high.embedding_[unbinned]],
            top_n=3,
            metric="euclidean",
            cluster_selection_methods=["eom"],
            solver="hbgf",
            embeddings_for_precomputed=[self.precomputed_reducer_low.embedding_[unbinned],
                                        # self.precomputed_reducer_mid.embedding_,
                                        self.precomputed_reducer_high.embedding_[unbinned]]
        )

        return labels[0]

    def perform_embedding(self, set_embedding=False, retry=0, retry_threshold=1):
        """
        set_embedding - Whether to overwrite the current self embeddings value
        retry - The current retry count
        retry_threshold - The amount of times we are willing to alter the seed and try again
        """
        if retry <= retry_threshold:
            try:
                # condensed ranked distance matrix for contigs
                stat = self.get_ranks()
                # generate umap embeddings
                logging.info("Fitting precomputed matrix using UMAP...")
                embeddings = self.fit_transform_precomputed(stat, set_embedding)
                # ensemble clustering against each umap embedding
                logging.info("Clustering UMAP embedding...")
                labels, validities, n_bins, unbinned = self.ensemble_cluster_multiple_embeddings(
                    embeddings,
                    top_n=3,
                    metric="euclidean",
                    cluster_selection_methods="eom",
                    solver="hbgf",
                    embeddings_for_precomputed=embeddings,
                    threads=self.threads,
                    use_multiple_processes=False
                )

                return labels[0]
            except IndexError:
                self.random_seed = np.random.randint(0, 1000000)
                self.perform_embedding(set_embedding, retry + 1, retry_threshold)
        else:
            raise IndexError

    def perform_binning(self, args):
        plots = []
        set_num_threads(int(self.threads))
        with threadpoolctl.threadpool_limits(limits=int(args.threads), user_api='blas'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if self.tnfs.values.shape[0] > int(args.n_neighbors):
                    # First pass quick TNF filter to speed up next steps and remove large contigs that
                    # are too distant from other contigs. These contigs tend to break UMAP results
                    logging.info("Filtering disconnected contigs...")
                    self.filter()
                    self.fit_disconnect()

                    # 1. First pass of embeddings + clustering
                    self.kmer_signature = self.tnfs[~self.disconnected].iloc[:, 2:].values
                    self.coverage_profile = self.large_contigs[~self.disconnected].iloc[:, 3:].values
                    self.labels = self.perform_embedding(set_embedding=True)

                    ## Plot limits
                    x_min = min(self.embeddings[:, 0]) - 10
                    x_max = max(self.embeddings[:, 0]) + 10
                    y_min = min(self.embeddings[:, 1]) - 10
                    y_max = max(self.embeddings[:, 1]) + 10

                    if plots is not None:
                        plots.append(
                            utils.plot_for_offset(self.embeddings, self.labels, x_min, x_max, y_min,
                                                  y_max, 0))
                    self.bin_contigs(args.assembly, int(args.min_bin_size))

                    self.findem = [
                        # 'contig_29111_pilon', 'contig_5229_pilon', 'contig_7458_pilon', # Ega
                        'contig_1570_pilon', 'scaffold_1358_pilon', 'contig_104_pilon', # Ret
                        # 'contig_3_pilon'
                        'contig_17512_pilon' # AalE
                    ]
                    self.plot(
                        self.findem
                    )

                    # self.embeddings = self.embeddings2
                    # self.plot(
                    #     self.findem,
                    #     suffix="_second"
                    # )
                    #
                    # self.embeddings = self.embeddings3
                    # self.plot(
                    #     self.findem,
                    #     suffix="_third"
                    # )

                    logging.info("Second embedding.")
                    self.sort_bins()
                    # 2. Recover the unbinned tids, and then perform the same procedure on them
                    #    The idea here is to pick up any obvious clusters that were missed. We reembed
                    #    again to try and make the relationships more obvious than the original embedding.
                    self.embed_unbinned("unbinned_1")
                    logging.info("Refining bins...")
                    # self.quick_filter(plots, 0, 1, x_min, x_max, y_min, y_max)
                    self.slow_refine(plots, 0, 5, x_min, x_max, y_min, y_max)
                    self.big_contig_filter(plots, 0, 3, x_min, x_max, y_min, y_max)
                    # self.quick_filter(plots, 0, 1, x_min, x_max, y_min, y_max)

                    logging.info("Third embedding.")
                    # 3. Recover the unbinned tids, and then perform the same procedure on them
                    #    The idea here is to pick up any obvious clusters that were missed. We reembed
                    #    again to try and make the relationships more obvious than the original embedding.
                    self.embed_unbinned("unbinned_2")

                    self.slow_refine(plots, 0, 1, x_min, x_max, y_min, y_max)
                    self.big_contig_filter(plots, 0, 3, x_min, x_max, y_min, y_max)
                    # self.quick_filter(plots, 0, 1, x_min, x_max, y_min, y_max)
                    self.get_labels_from_bins()
                    self.plot(
                        self.findem,
                        suffix="final"
                    )
                    self.bin_filtered(int(args.min_bin_size), keep_unbinned=False, unbinned_only=False)
                else:
                    self.rescue_contigs(int(args.min_bin_size))
            logging.info(f"Writing bins... {len(self.bins.keys())}")
            self.write_bins(int(args.min_bin_size))
            # try:
            #     imageio.mimsave(self.path + '/UMAP_projections.gif', plots, fps=1)
            # except RuntimeError:  # no plotting has occurred due to no embedding
            #     pass

    def do_nothing(self):
        pass

    def embed_unbinned(self, suffix="unbinned"):
        try:
            self.get_labels_from_bins()
            all_embedded = self.embeddings
            all_labels = self.labels
            unbinned = self.labels == -1
            unbinned += self.large_contigs[~self.disconnected]['tid'].isin(self.unbinned_tids).values
            max_bin_key = max(self.labels)
            # reset kmer sigs
            self.kmer_signature = self.tnfs[~self.disconnected][unbinned].iloc[:, 2:].values
            self.coverage_profile = self.large_contigs[~self.disconnected][unbinned].iloc[:, 3:].values

            unbinned_labels = self.perform_embedding(set_embedding=True)

            self.labels = unbinned_labels
            self.plot(
                [],
                plot_bin_ids=True,
                suffix=suffix
            )
            unbinned_labels[unbinned_labels != -1] += max_bin_key
            self.labels = all_labels
            self.labels[unbinned] = unbinned_labels
            self.embeddings = all_embedded
        except (ValueError, IndexError):
            # IndexError from call to perform_embedding when very few contigs left
            # not enough unbinned contigs
            self.get_labels_from_bins()

    def get_ranks(self):
        logging.info("Getting distance info...")
        contig_lengths = self.tnfs[~self.disconnected]["contigLen"].values
        de = distance.ProfileDistanceEngine()
        stat = de.makeRankStat(
            self.coverage_profile,
            self.kmer_signature,
            contig_lengths,
            silent=False,
            fun=lambda a: a / max(a),
            use_multiple_processes=False
        )

        return stat

def do_nothing():
    pass
