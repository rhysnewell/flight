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
import imageio
from numba import njit

# self imports
import flight.utils as utils
from flight.rosella.validating import Validator


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
                                          quick_filter=True)
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
            y_max=20
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
                                          reembed=True)
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
            plots, n = self.validate_bins(plots, n,
                                          x_min, x_max,
                                          y_min, y_max,
                                          reembed=True)
            self.sort_bins()

            n += 1

    def force_splitting(
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
            self.overclustered = False  # large clusters
            # Clean up leftover stuff
            self.reembed(self.unbinned_tids,
                         max(self.bins.keys()), plots,
                         x_min, x_max, y_min, y_max, n, delete_unbinned=True,
                         skip_clustering=True, reembed=True, force=True,
                         update_embeddings=False)

            self.sort_bins()
            plots, n = self.validate_bins(plots, n,
                                          x_min, x_max,
                                          y_min, y_max,
                                          reembed=True)
            self.sort_bins()
            plots, n = self.validate_bins(plots, n,
                                          x_min, x_max,
                                          y_min, y_max,
                                          force=True)
            self.sort_bins()
            n += 1
            if not self.overclustered:
                break  # no more clusters have broken

    def assign_unbinned(self, n=0, n_max=5, min_bin_size=5e5, debug=False):
        while n <= n_max:
            # Clean up leftover stuff
            if self.check_bad_bins_and_unbinned(min_bin_size=min_bin_size, debug=debug):
                self.sort_bins()
                n += 1
            else:
                self.sort_bins()
                break  # no more clusters have broken

    def perform_binning(self, args):
        self.update_umap_params(self.large_contigs.shape[0])
        plots = []
        with threadpoolctl.threadpool_limits(limits=int(args.threads), user_api='blas'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if self.tnfs.values.shape[0] > int(args.n_neighbors):
                    # First pass quick TNF filter to speed up next steps and remove large contigs that
                    # are too distant from other contigs. These contigs tend to break UMAP results
                    self.filter()
                    if self.tnfs[~self.disconnected].values.shape[0] > int(args.n_neighbors) * 5:
                        # Second pass intersection filtering
                        self.update_parameters()
                        self.fit_disconnect()
                        if self.tnfs[~self.disconnected][~self.disconnected_intersected].values.shape[
                            0] > int(args.n_neighbors) * 2:
                            # Final fully filtered embedding to cluster on
                            self.update_umap_params(self.large_contigs[~self.disconnected][
                                                             ~self.disconnected_intersected].shape[0])
                            self.fit_transform(
                                self.large_contigs[~self.disconnected][~self.disconnected_intersected][
                                    'tid'],
                                int(args.n_neighbors))
                            self.embeddings = self.intersection_mapper.embedding_

                            logging.info("HDBSCAN - Performing initial clustering.")
                            self.labels = self.iterative_clustering(self.embeddings,
                                                                              prediction_data=True,
                                                                              allow_single_cluster=False,
                                                                              double=False)

                            ## Plot limits
                            x_min = min(self.embeddings[:, 0]) - 10
                            x_max = max(self.embeddings[:, 0]) + 10
                            y_min = min(self.embeddings[:, 1]) - 10
                            y_max = max(self.embeddings[:, 1]) + 10

                            plots.append(
                                utils.plot_for_offset(self.embeddings, self.labels, x_min, x_max, y_min,
                                                      y_max,
                                                      0))
                            self.bin_contigs(args.assembly, int(args.min_bin_size))

                            self.plot(
                                ['contig_1357_pilon', 'contig_1570_pilon', 'contig_810_pilon', 'scaffold_1358_pilon'])

                            logging.info("Reclustering individual bins.")

                            self.sort_bins()

                            # Clean up leftover stuff
                            self.reembed(self.unbinned_tids,
                                              max(self.bins.keys()), plots,
                                              x_min, x_max, y_min, y_max, 0, delete_unbinned=True,
                                              skip_clustering=True, reembed=True, force=True,
                                              update_embeddings=False)

                            self.sort_bins()
                            self.quick_filter(plots, 0, 1, x_min, x_max, y_min, y_max)
                            self.slow_refine(plots, 0, 100, x_min, x_max, y_min, y_max)
                            self.big_contig_filter(plots, 0, 5, x_min, x_max, y_min, y_max)
                            self.force_splitting(plots, 0, 5, x_min, x_max, y_min, y_max)

                            # self.get_labels_from_bins()
                            # self.combine_bins(threshold=0.001)
                            # self.sort_bins()


                            self.bin_filtered(int(args.min_bin_size), keep_unbinned=False, unbinned_only=False)
                            # self.assign_unbinned(0, 0, 2e5)

                        else:
                            self.rescue_contigs(int(args.min_bin_size))
                    else:
                        self.rescue_contigs(int(args.min_bin_size))
                else:
                    self.rescue_contigs(int(args.min_bin_size))
            logging.debug("Writing bins...", len(self.bins.keys()))
            self.write_bins(int(args.min_bin_size))
            try:
                imageio.mimsave(self.path + '/UMAP_projections.gif', plots, fps=1)
            except RuntimeError:  # no plotting has occurred due to no embedding
                pass