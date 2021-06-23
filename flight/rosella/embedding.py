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
import logging

# Function imports
import numpy as np
from numba import njit
import pandas as pd
import seaborn as sns
import matplotlib
import umap

# self imports
import flight.metrics as metrics
from flight.rosella.binning import Binner

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

class Embedder(Binner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_umap_params(self, nrows):
        if nrows <= 20000:  # high gear
            # Small datasets can have larger n_neighbors without being prohibitively slow
            if nrows <= 1000:  # wheels fell off
                self.tnf_reducer.n_neighbors = nrows // 10
                self.euc_reducer.n_neighbors = nrows // 10
                self.depth_reducer.n_neighbors = nrows // 10
            else:
                self.tnf_reducer.n_neighbors = 100
                self.euc_reducer.n_neighbors = 100
                self.depth_reducer.n_neighbors = 100
            # self.tnf_reducer.n_epochs = 500
            # self.depth_reducer.n_epochs = 500
        elif nrows <= 100000:  # mid gear
            # Things start to get too slow around here, so scale back params
            self.tnf_reducer.n_neighbors = 100
            self.euc_reducer.n_neighbors = 100
            # self.tnf_reducer.n_epochs = 500
            self.depth_reducer.n_neighbors = 100
            # self.depth_reducer.n_epochs = 500
        else:  # low gear
            # This is the super slow zone, but don't want to dip values below this
            # Hopefully pick out easy bins, then scale data down with each iterations
            # Allowing the params to bump up into other gears
            self.tnf_reducer.n_neighbors = 100
            self.euc_reducer.n_neighbors = 100
            # self.tnf_reducer.n_epochs = 500
            self.depth_reducer.n_neighbors = 100
            # self.depth_reducer.n_epochs = 500


    def filter(self):
        """
        Performs quick UMAP embeddings with stringent disconnection distances
        """
        try:

            # We check the euclidean distances of large contigs as well.
            disconnections_very_big = self._check_contigs(self.large_contigs[self.large_contigs['contigLen'] >= 2.5e6]['tid'], 0.05, 2)
            disconnections_big = self._check_contigs(self.large_contigs[(self.large_contigs['contigLen'] >= 1.5e6)
                                                                        & (self.large_contigs['contigLen'] < 2.5e6)]['tid'], 0.1, 2)
            disconnections_small = self._check_contigs(self.large_contigs[(self.large_contigs['contigLen'] > 2.5e3)
                                                                        & (self.large_contigs['contigLen'] <= 1e4)]['tid'], 0.1, 2)
            disconnections_very_small = self._check_contigs(self.large_contigs[self.large_contigs['contigLen'] <= 2.5e3]['tid'], 0.05, 2)

            initial_disconnections = disconnections_very_big + disconnections_big + \
                             disconnections_small + disconnections_very_small
            logging.info("Running UMAP Filter - %s" % self.rho_reducer)
            self.rho_reducer.n_neighbors = 5
            self.rho_reducer.disconnection_distance = 0.15

            contigs, log_lengths, tnfs = self.extract_contigs(
                self.large_contigs[~initial_disconnections]['tid']
            )

            filterer_rho = self.rho_reducer.fit(
                np.concatenate((log_lengths.values[:, None],
                                tnfs.iloc[:, 2:]), axis=1))

            disconnected_umap = umap.utils.disconnected_vertices(filterer_rho)

            disconnected = initial_disconnections + np.array(self.large_contigs['tid'].isin(
                self.large_contigs[~initial_disconnections][disconnected_umap]['tid']
            ))

        except ValueError: # Everything was disconnected
            disconnected = np.array([True for i in range(self.large_contigs.values.shape[0])])

        self.disconnected = disconnected



    def _check_contigs(self, tids, rho_threshold=0.05, euc_threshold=2):
        logging.info("Checking TNF connections...")
        disconnected_tids = []
        if self.n_samples > 0:
            n_samples = self.n_samples
            sample_distances = self.short_sample_distance
        else:
            n_samples = self.long_samples
            sample_distances = self.long_sample_distance

        for tid in tids:

            current_contigs, current_log_lengths, current_tnfs = \
                self.extract_contigs([tid])
            other_contigs, other_log_lengths, other_tnfs = \
                self.extract_contigs(self.large_contigs[self.large_contigs['tid'] != tid]['tid'])

            current = np.concatenate((current_contigs.iloc[:, 3:].values,
                                      current_log_lengths.values[:, None],
                                      current_tnfs.iloc[:, 2:].values), axis=1)

            others = np.concatenate((other_contigs.iloc[:, 3:].values,
                                    other_log_lengths.values[:, None],
                                    other_tnfs.iloc[:, 2:].values), axis=1)


            rho_connected, euc_connected = metrics.check_connections(
                current, others, n_samples, rho_threshold=rho_threshold, euc_threshold=euc_threshold
            )

            if not rho_connected or not euc_connected:
                # print(tid, self.large_contigs[self.large_contigs['tid'] == tid])
                # idx = self.large_contigs.index[self.large_contigs['tid'] == tid].tolist()
                # print(idx)
                # disconnections[idx[0]] = True
                disconnected_tids.append(tid)

        disconnections = np.array(self.large_contigs['tid'].isin(disconnected_tids))

        return disconnections

    def fit_disconnect(self):
        """
        Filter contigs based on ADP connections
        """
        ## Calculate the UMAP embeddings

        self.depths = np.nan_to_num(
            np.concatenate((self.large_contigs[~self.disconnected].iloc[:, 3:].drop(['tid'], axis=1),
                            self.log_lengths[~self.disconnected].values[:, None],
                            self.tnfs[~self.disconnected].iloc[:, 2:]), axis=1))

        # Get all disconnected points, i.e. contigs that were disconnected in ANY mapping
        # logging.info("Running UMAP Filter - %s" % self.depth_reducer)
        depth_mapping = self.md_reducer.fit(self.depths)

        logging.info("Finding disconnections...")
        disconnected_intersected = umap.utils.disconnected_vertices(depth_mapping)  # + \
        # umap.utils.disconnected_vertices(tnf_mapping)

        # Only disconnect big contigs
        for i, dis in enumerate(disconnected_intersected):
            if dis:
                contig = self.large_contigs[~self.disconnected].iloc[i, :]
                if contig['contigLen'] < self.min_bin_size:
                    disconnected_intersected[i] = False

        logging.info("Found %d disconnected points. %d TNF disconnected and %d ADP disconnected..." %
                     (sum(self.disconnected) + sum(disconnected_intersected), sum(self.disconnected),
                      sum(disconnected_intersected)))
        # logging.debug out all disconnected contigs
        pd.concat([self.large_contigs[self.disconnected],
                   self.large_contigs[~self.disconnected][disconnected_intersected]]) \
            .to_csv(self.path + "/disconnected_contigs.tsv", sep="\t", header=True)

        self.disconnected_intersected = disconnected_intersected

        # if np.median(self.large_contigs[~self.disconnected][~self.disconnected_intersected]['contigLen']) < 25000:
        #     # Lower median can use euclidean UMAP
        #     self.use_euclidean = True
        # else:
        #     # Distribution of contigs tends to be larger, so euclidean distance breaks down
        #     self.use_euclidean = False


    def fit_transform(self, tids, n_neighbors):
        """
        Main function for performing UMAP embeddings and intersections
        """
        # update parameters to artificially high values to avoid disconnected vertices in the final manifold
        self.depth_reducer.n_neighbors = min(n_neighbors, len(tids) - 1)
        self.depth_reducer.disconnection_distance = 0.99
        self.tnf_reducer.n_neighbors = min(n_neighbors, len(tids) - 1)
        self.tnf_reducer.disconnection_distance = 1
        self.euc_reducer.n_neighbors = min(n_neighbors, len(tids) - 1)
        self.euc_reducer.disconnection_distance = 100

        # self.update_umap_params(self.large_contigs[~self.disconnected][~self.disconnected_intersected].shape[0])
        contigs, log_lengths, tnfs = self.extract_contigs(tids)

        logging.debug("Running UMAP - %s" % self.tnf_reducer)
        tnf_mapping = self.tnf_reducer.fit(
            np.concatenate(
                (log_lengths.values[:, None],
                 tnfs.iloc[:, 2:]),
                axis=1))


        logging.debug("Running UMAP - %s" % self.depth_reducer)
        depth_mapping = self.depth_reducer.fit(np.concatenate(
            (contigs.iloc[:, 3:], log_lengths.values[:, None], tnfs.iloc[:, 2:]), axis=1))

        if self.use_euclidean:
            logging.debug("Running UMAP - %s" % self.euc_reducer)
            euc_mapping = self.euc_reducer.fit(
                np.concatenate(
                    (log_lengths.values[:, None],
                     tnfs.iloc[:, 2:]),
                    axis=1))
            ## Intersect all of the embeddings
            intersection_mapper = depth_mapping * euc_mapping * tnf_mapping
        else:
            intersection_mapper = depth_mapping * tnf_mapping

        self.intersection_mapper = intersection_mapper


    def rho_md_transform(self, tids, n_neighbors):
        self.depth_reducer.n_neighbors = min(n_neighbors, len(tids) - 1)
        self.depth_reducer.disconnection_distance = 0.99
        self.tnf_reducer.n_neighbors = min(n_neighbors, len(tids) - 1)
        self.tnf_reducer.disconnection_distance = 1

        # self.update_umap_params(self.large_contigs[~self.disconnected][~self.disconnected_intersected].shape[0])
        contigs, log_lengths, tnfs = self.extract_contigs(tids)

        logging.debug("Running UMAP - %s" % self.tnf_reducer)
        tnf_mapping = self.tnf_reducer.fit(
            np.concatenate(
                (log_lengths.values[:, None],
                 tnfs.iloc[:, 2:]),
                axis=1))

        logging.debug("Running UMAP - %s" % self.depth_reducer)
        depth_mapping = self.depth_reducer.fit(np.concatenate(
            (contigs.iloc[:, 3:], log_lengths.values[:, None], tnfs.iloc[:, 2:]), axis=1))


        intersection_mapper = depth_mapping * tnf_mapping

        self.intersection_mapper = intersection_mapper

    def euc_md_transform(self, tids, n_neighbors):
        """
        Main function for performing UMAP embeddings and intersections
        """
        # update parameters to artificially high values to avoid disconnected vertices in the final manifold
        self.depth_reducer.n_neighbors = min(n_neighbors, len(tids) - 1)
        self.depth_reducer.disconnection_distance = 0.99
        self.euc_reducer.n_neighbors = min(n_neighbors, len(tids) - 1)
        self.euc_reducer.disconnection_distance = 100

        # self.update_umap_params(self.large_contigs[~self.disconnected][~self.disconnected_intersected].shape[0])
        contigs, log_lengths, tnfs = self.extract_contigs(tids)

        logging.debug("Running UMAP - %s" % self.depth_reducer)
        depth_mapping = self.depth_reducer.fit(np.concatenate(
            (contigs.iloc[:, 3:], log_lengths.values[:, None], tnfs.iloc[:, 2:]), axis=1))

        logging.debug("Running UMAP - %s" % self.euc_reducer)
        euc_mapping = self.euc_reducer.fit(
            np.concatenate(
                (log_lengths.values[:, None],
                 tnfs.iloc[:, 2:]),
                axis=1))
        ## Intersect all of the embeddings
        intersection_mapper = depth_mapping * euc_mapping

        self.intersection_mapper = intersection_mapper


    def multi_transform(self, tids, n_neighbors, switch=0):
        """
        Main function for performing UMAP embeddings and intersections
        """
        # update parameters to artificially high values to avoid disconnected vertices in the final manifold
        self.depth_reducer.n_neighbors = min(n_neighbors, len(tids) - 1)
        self.depth_reducer.disconnection_distance = 0.99
        self.tnf_reducer.n_neighbors = min(n_neighbors, len(tids) - 1)
        self.tnf_reducer.disconnection_distance = 1
        self.euc_reducer.n_neighbors = min(n_neighbors, len(tids) - 1)
        self.euc_reducer.disconnection_distance = 100

        # self.update_umap_params(self.large_contigs[~self.disconnected][~self.disconnected_intersected].shape[0])
        contigs, log_lengths, tnfs = self.extract_contigs(tids)

        logging.debug("Running UMAP - %s" % self.depth_reducer)
        self.depth_mapping = self.depth_reducer.fit(np.concatenate(
            (contigs.iloc[:, 3:], log_lengths.values[:, None], tnfs.iloc[:, 2:]), axis=1))

        logging.debug("Running UMAP - %s" % self.tnf_reducer)
        self.tnf_mapping = self.tnf_reducer.fit(
            np.concatenate(
                (log_lengths.values[:, None],
                 tnfs.iloc[:, 2:]),
                    axis=1))
        if (switch == 0 and self.use_euclidean) or switch == 2:
            logging.debug("Running UMAP - %s" % self.euc_reducer)
            self.euc_mapping = self.euc_reducer.fit(
                np.concatenate(
                    (log_lengths.values[:, None],
                     tnfs.iloc[:, 2:]),
                    axis=1))


    def switch_intersector(self, switch=0):
        if switch == 0 and self.use_euclidean:
            # All
            self.intersection_mapper = self.depth_mapping * self.euc_mapping * self.tnf_mapping
        elif switch == 1 or (switch==0 and not self.use_euclidean):
            # Rho and MD
            self.intersection_mapper = self.depth_mapping * self.tnf_mapping
        else:
            self.intersection_mapper = self.depth_mapping * self.euc_mapping * self.tnf_mapping


