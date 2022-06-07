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


###############################################################################
# System imports
import threadpoolctl
import matplotlib
import warnings
import logging
import glob
import sys
import os

# Function imports
from numba import njit, set_num_threads
from Bio import SeqIO
import seaborn as sns
import numpy as np

# self imports
from flight.rosella.validating import Validator
import flight.distance as distance
import flight.utils as utils
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
            min_completeness=50.0,
            max_contamination=20.0,
            min_bin_size_for_averages=1e6,
            contaminated_only=False,
            refining_mode=False,
            debug=False,
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
                                          large_bins_only=large_bins_only,
                                          min_completeness=min_completeness,
                                          max_contamination=max_contamination,
                                          min_bin_size=min_bin_size_for_averages,
                                          contaminated_only=contaminated_only,
                                          refining_mode=refining_mode,
                                          debug=debug
                                          )
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
                                          big_only=True, min_bin_size=self.min_bin_size)
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


    def perform_embedding(self, tids, switches=None, set_embedding=False, retry=0, retry_threshold=1):
        """
        set_embedding - Whether to overwrite the current self embeddings value
        retry - The current retry count
        retry_threshold - The amount of times we are willing to alter the seed and try again
        """
        if retry <= retry_threshold:
            try:
                # condensed ranked distance matrix for contigs
                # stat = self.get_ranks()
                # generate umap embeddings
                logging.info("Fitting precomputed matrix using UMAP...")
                # embeddings = self.fit_transform_precomputed(stat, set_embedding)
                if len(tids) <= self.n_neighbors:
                    return np.array([-1 for _ in tids])
                embeddings = self.fit_transform(tids, switches=switches, set_embedding=set_embedding)
                # ensemble clustering against each umap embedding
                logging.info("Clustering UMAP embedding...")
                labels, validities, n_bins, unbinned = self.ensemble_cluster_multiple_embeddings(
                    [embeddings],
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
        plots = None
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
                    logging.info(f"Filtered {self.disconnected.sum()} contigs...")
                    # 1. First pass of embeddings + clustering
                    self.kmer_signature = self.tnfs[~self.disconnected].iloc[:, 2:].values
                    self.coverage_profile = self.large_contigs[~self.disconnected].iloc[:, 3:].values
                    if self.n_samples <= 1:
                        switches = [0, 1, None]
                    else:
                        switches = [0, None, 2]
                    self.labels = self.perform_embedding(self.large_contigs[~self.disconnected]['tid'].values, switches=[0, None, 2], set_embedding=True)

                    ## Plot limits
                    x_min = min(self.embeddings[:, 0]) - 10
                    x_max = max(self.embeddings[:, 0]) + 10
                    y_min = min(self.embeddings[:, 1]) - 10
                    y_max = max(self.embeddings[:, 1]) + 10

                    if plots is not None:
                        plots.append(
                            utils.plot_for_offset(self.embeddings, self.labels, x_min, x_max, y_min,
                                                  y_max, 0))
                    self.bin_contigs()

                    self.findem = [
                        'RL|S1|C13963', 'RL|S1|C11210', 'RL|S1|C12411', 'RL|S1|C13372', 'RL|S1|C14115', 'RL|S1|C16600', 'RL|S1|C17450',
                        'contig_810_pilon', 'scaffold_1358_pilon', # Ret
                        # 'contig_3_pilon'
                        'contig_17512_pilon' # AalE
                    ]
                    self.plot(
                        None,
                        self.findem
                    )

                    self.embed_unbinned(self.findem, "unbinned_1", switches)
                    logging.info("Second embedding.")
                    self.sort_bins()
                    # 2. Recover the unbinned tids, and then perform the same procedure on them
                    #    The idea here is to pick up any obvious clusters that were missed. We reembed
                    #    again to try and make the relationships more obvious than the original embedding.
                    # self.dissolve_bins(5e5)
                    # self.embed_unbinned("unbinned_1")
                    # logging.info("Refining bins...")
                    # self.quick_filter(plots, 0, 1, x_min, x_max, y_min, y_max)
                    self.slow_refine(plots, 0, 5, x_min, x_max, y_min, y_max)
                    self.big_contig_filter(plots, 0, 3, x_min, x_max, y_min, y_max)
                    # self.quick_filter(plots, 0, 1, x_min, x_max, y_min, y_max)

                    logging.info("Third embedding.")
                    # 3. Recover the unbinned tids, and then perform the same procedure on them
                    #    The idea here is to pick up any obvious clusters that were missed. We reembed
                    #    again to try and make the relationships more obvious than the original embedding.
                    # self.dissolve_bins(5e5)
                    # self.embed_unbinned("unbinned_2")
                    # self.slow_refine(plots, 0, 2, x_min, x_max, y_min, y_max)
                    self.dissolve_bins(1e6)
                    self.embed_unbinned(self.findem, "unbinned_2", switches)
                    self.embed_unbinned(self.findem, "unbinned_3", switches)
                    # self.slow_refine(plots, 0, 0, x_min, x_max, y_min, y_max)
                    # self.big_contig_filter(plots, 0, 2, x_min, x_max, y_min, y_max)
                    # self.dissolve_bins(1e6)
                    # self.embed_unbinned("unbinned_4")
                    # self.quick_filter(plots, 0, 1, x_min, x_max, y_min, y_max)
                    self.get_labels_from_bins()
                    self.plot(
                        None,
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


    def embed_unbinned(self, findem = None, suffix="unbinned", switches=None):
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

            try:
                if len(unbinned) > self.n_neighbors:
                    unbinned_labels = self.perform_embedding(self.large_contigs[~self.disconnected][unbinned]['tid'].values, switches=switches, set_embedding=True)
                else:
                    unbinned_labels = np.array([-1 for _ in range(len(unbinned))])
            except (IndexError, TypeError):
                unbinned_labels = np.array([-1 for _ in range(len(unbinned))])

            self.labels = unbinned_labels
            self.plot(
                unbinned,
                findem,
                plot_bin_ids=True,
                suffix=suffix
            )
            unbinned_labels[unbinned_labels != -1] += max_bin_key

            self.labels = all_labels
            self.labels[unbinned] = unbinned_labels

            self.unbinned_tids = []

            for (idx, label) in enumerate(unbinned_labels):
                if label != -1:
                    try:
                        self.bins[label.item() + 1].append(
                            self.assembly[self.large_contigs[~self.disconnected][unbinned].iloc[
                                idx, 0]])  # inputs values as tid
                    except KeyError:
                        # self.bin_validity[label.item() + 1] = self.validity_indices[label]
                        self.bins[label.item() + 1] = [
                            self.assembly[
                                self.large_contigs[
                                    ~self.disconnected
                                ][unbinned].iloc[idx, 0]]]
                else:
                    self.unbinned_tids.append(self.assembly[self.large_contigs[~self.disconnected][
                        unbinned].iloc[idx, 0]])

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


    def perform_refining(self, args):
        import pandas as pd

        plots = None
        set_num_threads(int(self.threads))
        with threadpoolctl.threadpool_limits(limits=int(args.threads), user_api='blas'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                logging.info("Reading in bin statistics...")
                # get checkm results if present
                if args.checkm_file is not None: self.checkm_file = pd.read_csv(args.checkm_file, sep='\t', comment="[")

                input_bin_stats, bin_dict = retrieve_stats(
                    self.large_contigs,
                    args.bin_paths,
                    args.bin_folder,
                    args.bin_extension,
                    self.checkm_file
                )
                self.disconnected = np.array([False for _ in range(self.large_contigs.shape[0])])
                self.disconnected_intersected = np.array([False for _ in range(self.large_contigs.shape[0])])
                self.embeddings = np.random.rand(self.large_contigs.shape[0], 2)
                self.bins = bin_dict

                if input_bin_stats is not None: self.input_bin_stats = pd.DataFrame(data=input_bin_stats)

                logging.info("Refining bins...")
                x_min, x_max, y_min, y_max = 20, 20, 20, 20 # default plotting margins

                self.slow_refine(plots, 0, 10, x_min, x_max, y_min, y_max, False,
                                 float(args.min_completeness), float(args.max_contamination),
                                 0, args.contaminated_only, True, False)
                self.big_contig_filter(plots, 0, 3, x_min, x_max, y_min, y_max)
                self.bin_filtered(int(args.min_bin_size), keep_unbinned=False, unbinned_only=False)
                logging.info(f"Writing bins... {len(self.bins.keys())}")
                self.write_bins(int(args.min_bin_size))



def retrieve_stats(coverage_file, bin_paths=None, bin_folder=None, bin_extension="fna", checkm_file=None):
    """
    Retrieves information about all bins within `bin_folder` given the file extension `bin_extension`

    if checkm file is not None
    returns a dict containing the bin_id, number of contigs, size in base pairs, completeness, and contamination

    Always returns dictionary with bin id and tids in that bin
    """
    if checkm_file is not None:
        output_dict = {"bin_id": [], "bin_index": [], "n_contigs": [], "size": [], "completeness": [], "contamination": []}
    else:
        output_dict = None


    bin_index_dict = {}

    if bin_folder is not None:
        bin_paths = glob.glob(f"{bin_folder}/*.{bin_extension}")
    elif bin_paths is None:
        sys.exit(f"No bin folder or bin paths supplied. Please supply one or the other and try again...")

    for bin_index, fasta_path in enumerate(bin_paths):
        bin_index += 1
        bin_id = fasta_path.split("/")[-1]
        bin_id = os.path.splitext(bin_id)[0]

        contig_ids = []
        for sequence in SeqIO.parse(open(fasta_path), "fasta"):
            contig_ids.append(sequence.id)
        contigs_in_bin = coverage_file[coverage_file["contigName"].isin(contig_ids)]
        bin_index_dict[bin_index] = contigs_in_bin["tid"].values.tolist()

        if checkm_file is not None:
            output_dict["bin_id"].append(bin_id)
            output_dict["bin_index"].append(bin_index)
            output_dict["n_contigs"].append(len(contig_ids))
            output_dict["size"].append(
                contigs_in_bin['contigLen'].sum())

            try:
                # checkm1 uses Bin Id
                checkm_stats = checkm_file[checkm_file["Bin Id"] == bin_id]
            except KeyError:
                # checkm2 uses Name
                checkm_stats = checkm_file[checkm_file["Name"] == bin_id]

            output_dict["completeness"].append(checkm_stats["Completeness"].values[0])
            output_dict["contamination"].append(checkm_stats["Contamination"].values[0])

    return (output_dict, bin_index_dict)

def do_nothing():
    pass
