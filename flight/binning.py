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
import sys
import argparse
import logging
import warnings

# Function imports
import math
import numpy as np
from numba import njit, set_num_threads
import pandas as pd
import hdbscan
import seaborn as sns
import json
import matplotlib
import matplotlib.pyplot as plt
from Bio import SeqIO
import skbio.stats.composition
import umap
from itertools import product, combinations
import scipy.stats as sp_stats
from numpy import int64
import sklearn.metrics as sk_metrics

# self imports
import flight.metrics as metrics
import flight.utils as utils

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
################################ - Globals - ################################

# tnfs = {}

###############################################################################                                                                                                                      [44/1010]
################################ - Functions - ################################

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

###############################################################################
################################ - Classes - ##################################


class Binner():
    def __init__(
            self,
            count_path,
            long_count_path,
            kmer_frequencies,
            output_prefix,
            assembly,
            n_neighbors=100,
            min_dist=0.1,
            min_contig_size=2500,
            threads=8,
            a=1.58,
            b=0.4,
            min_bin_size=200000,
            initialization='spectral',
            random_seed=42069
    ):
        # config.THREADING_LAYER = 'tbb'
        # config.NUMBA_NUM_THREADS = threads
        self.min_contig_size = min_contig_size
        self.min_bin_size = min_bin_size
        self.threads = threads
        self.checked_bins = [] # Used in the pdist function
        self.survived = []
        # Open up assembly
        self.assembly = {} 
        self.assembly_names = {}
        if assembly is not None:
            for (tid, rec) in enumerate(SeqIO.parse(assembly, "fasta")):
                self.assembly[rec.id] = tid
                self.assembly_names[tid] = rec.id

        # initialize bin dictionary Label: Vec<Contig>
        self.bins = {}
        self.bin_validity = {}

        ## Set up clusterer and UMAP
        self.path = output_prefix

        ## These tables should have the same ordering as each other if they came from rosella.
        ## I.e. all the rows match the same contig
        if count_path is not None and long_count_path is not None:
            self.coverage_table = pd.read_csv(count_path, sep='\t')
            self.long_depths = pd.read_csv(long_count_path, sep='\t')
            self.coverage_table['coverageSum'] = (self.coverage_table.iloc[:, 3::2] > 0).any(axis=1)
            self.long_depths['coverageSum'] = (self.long_depths.iloc[:, 3::2] > 0).any(axis=1)

            self.large_contigs = self.coverage_table[(self.coverage_table["contigLen"] >= min_contig_size) 
                                 & ((self.coverage_table["coverageSum"])
                                 & (self.long_depths["coverageSum"]))]
            self.small_contigs = self.coverage_table[(self.coverage_table["contigLen"] < min_contig_size)
                                 | ((~self.coverage_table["coverageSum"])
                                 | (~self.long_depths["coverageSum"]))]
            self.long_depths = self.long_depths[self.long_depths['contigName'].isin(self.large_contigs['contigName'])]

            self.large_contigs = self.large_contigs.drop('coverageSum', axis=1)
            self.small_contigs = self.small_contigs.drop('coverageSum', axis=1)
            self.long_depths = self.long_depths.drop('coverageSum', axis=1)


            self.large_contigs = pd.concat([self.large_contigs, self.long_depths.iloc[:, 3:]], axis = 1)
            self.n_samples = len(self.large_contigs.columns[3::2])
            self.long_samples = 0
            self.short_sample_distance = utils.sample_distance(self.large_contigs)
            self.long_sample_distance = utils.sample_distance(self.long_depths)



        elif count_path is not None:
            self.coverage_table = pd.read_csv(count_path, sep='\t')
            self.coverage_table['coverageSum'] = (self.coverage_table.iloc[:, 3::2] > 0).any(axis=1)
            self.large_contigs = self.coverage_table[(self.coverage_table["contigLen"] >= min_contig_size)
                                 & (self.coverage_table["coverageSum"])]
            self.small_contigs = self.coverage_table[(self.coverage_table["contigLen"] < min_contig_size)
                                 | (~self.coverage_table["coverageSum"])]
                                 
            self.large_contigs = self.large_contigs.drop('coverageSum', axis=1)
            self.small_contigs = self.small_contigs.drop('coverageSum', axis=1)
            self.short_sample_distance = utils.sample_distance(self.large_contigs)

            self.n_samples = len(self.large_contigs.columns[3::2])
            self.long_samples = 0

        else:
            ## Treat long coverages as the default set
            self.coverage_table = pd.read_csv(long_count_path, sep='\t')
            self.coverage_table['coverageSum'] = (self.coverage_table.iloc[:, 3::2] > 0).any(axis=1)
            self.large_contigs = self.coverage_table[(self.coverage_table["contigLen"] >= min_contig_size)
                                 & (self.coverage_table["coverageSum"])]
            self.small_contigs = self.coverage_table[(self.coverage_table["contigLen"] < min_contig_size)
                                 | (~self.coverage_table["coverageSum"])]
            self.large_contigs = self.large_contigs.drop('coverageSum', axis=1)
            self.small_contigs = self.small_contigs.drop('coverageSum', axis=1)
            self.long_sample_distance = utils.sample_distance(self.large_contigs)
            self.long_samples = 0
            self.n_samples = len(self.large_contigs.columns[3::2])


        if assembly is None:
            for (tid, rec) in enumerate(self.coverage_table['contigName']):
                self.assembly[rec] = tid
                self.assembly_names[tid] = rec
        
        tids = []
        for name in self.large_contigs['contigName']:
            tids.append(self.assembly[name])
        self.large_contigs['tid'] = tids
        
        ## Handle TNFs
        self.tnfs = pd.read_csv(kmer_frequencies, sep='\t')
        self.tnfs = self.tnfs[self.tnfs['contigName'].isin(self.large_contigs['contigName'])]
        ## Divide by row sums to get frequencies
        self.tnfs.iloc[:, 2:] = skbio.stats.composition.clr(self.tnfs.iloc[:, 2:].astype(np.float64) + 1)
        ## Set custom log base change for lengths
        self.log_lengths = np.log(self.tnfs['contigLen']) / np.log(max(sp_stats.mstats.gmean(self.tnfs['contigLen']), 10000))
        
        ## Check the ordering of the contig names for sanity check
        if list(self.large_contigs['contigName']) != list(self.tnfs['contigName']):
            sys.exit("Contig ordering incorrect for kmer table or coverage table")

        if np.median(self.large_contigs['contigLen']) < 10000:
            # Lower mean can use euclidean UMAP
            self.use_euclidean = True
        else:
            # Distribution of contigs tends to be larger, so euclidean distance breaks down
            self.use_euclidean = False

        self.binning_method = 'eom'
        self.min_cluster_size = 2

        n_components = min(max(self.n_samples, self.long_samples, 2), 5)



        self.rho_reducer = umap.UMAP(
            metric=metrics.rho,
            n_neighbors=int(n_neighbors),
            n_components=2,
            min_dist=0,
            disconnection_distance=0.05,
            set_op_mix_ratio=1,
            a=a,
            b=b,
            init=initialization,
            random_state=random_seed,
        )

        self.tnf_reducer = umap.UMAP(
            metric=metrics.rho,
            n_neighbors=int(n_neighbors),
            n_components=2,
            min_dist=0,
            disconnection_distance=0.05,
            set_op_mix_ratio=1,
            a=a,
            b=b,
            init=initialization,
            random_state=random_seed
        )

        self.euc_reducer = umap.UMAP(
            metric=metrics.tnf_euclidean,
            n_neighbors=int(n_neighbors),
            n_components=2,
            min_dist=0,
            set_op_mix_ratio=1,
            a=a,
            b=b,
            init=initialization,
            random_state=random_seed
        )
        
        self.depth_reducer = umap.UMAP(
            metric=metrics.aggregate_tnf,
            metric_kwds={"n_samples": self.n_samples, "sample_distances": self.short_sample_distance},
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            set_op_mix_ratio=1,
            a=a,
            b=b,
            init=initialization,
            random_state=random_seed
        )

        self.md_reducer = umap.UMAP(
            metric=metrics.aggregate_md,
            metric_kwds={"n_samples": self.n_samples, "sample_distances": self.short_sample_distance},
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            set_op_mix_ratio=1,
            a=a,
            b=b,
            init=initialization,
            random_state=random_seed
        )

        self.n_neighbors = n_neighbors

        self.update_umap_params(self.large_contigs.shape[0])

    def update_umap_params(self, nrows):
        if nrows <= 20000: # high gear
            # Small datasets can have larger n_neighbors without being prohibitively slow
            if nrows <= 1000: # wheels fell off
                self.tnf_reducer.n_neighbors = nrows // 10
                self.euc_reducer.n_neighbors = nrows // 10
                self.depth_reducer.n_neighbors = nrows // 10
            else:
                self.tnf_reducer.n_neighbors = 100
                self.euc_reducer.n_neighbors = 100
                self.depth_reducer.n_neighbors = 100
            # self.tnf_reducer.n_epochs = 500
            # self.depth_reducer.n_epochs = 500
        elif nrows <= 100000: # mid gear
            # Things start to get too slow around here, so scale back params
            self.tnf_reducer.n_neighbors = 100
            self.euc_reducer.n_neighbors = 100
            # self.tnf_reducer.n_epochs = 500
            self.depth_reducer.n_neighbors = 100
            # self.depth_reducer.n_epochs = 500
        else: # low gear
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
            disconnections = self.check_contigs(self.large_contigs[self.large_contigs['contigLen'] > 1e6]['tid'])

            logging.info("Running UMAP Filter - %s" % self.rho_reducer)
            self.rho_reducer.n_neighbors = 5
            self.rho_reducer.disconnection_distance = 0.05

            filterer_rho = self.rho_reducer.fit(
                np.concatenate((self.log_lengths.values[:, None],
                                self.tnfs.iloc[:, 2:]), axis = 1))

            self.disconnected = umap.utils.disconnected_vertices(filterer_rho) + disconnections

        except ValueError: # Everything was disconnected
            self.disconnected = np.array([True for i in range(self.large_contigs.values.shape[0])])


    def check_contigs(self, tids, rho_threshold=0.05, euc_threshold=2):
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



    def n50(self):
        """
        Calculates N50 of contigs greater than the min contig size
        """
        lengths = np.sort(self.large_contigs['contigLen'])[::-1]

        idx, n50 = 0, 0
        contig_sum = lengths.sum() / 2
        for counter in range(1, len(lengths) + 1):
            if lengths[0:counter].sum() > contig_sum:
                idx = counter - 1
                n50 = lengths[counter - 1]
                break

        return idx, n50


    def update_parameters(self):
        """
        Mainly deprecated
        """
        # Generate CDF of the lognormal distribution using current contig size distirbution
        # We log normalize everything as well to log base 10000 using sick log rules
        # We also want to reparameterize after we filter contigs since the size distirbutions will change
        # Lognormal keeps everything above 0, 
        # Fun Fact: Lognorm is usually used for stock probability prediction since stock prices never fall below 0 and stonks only go up
        try:
            lognorm_cdf = sp_stats.lognorm.cdf(np.log(self.large_contigs[~self.disconnected][~self.disconnected_intersected]['contigLen']), 
                                        np.log(self.large_contigs[~self.disconnected][~self.disconnected_intersected]['contigLen']).std(), # S - shape parameter
                                        np.log(self.large_contigs[~self.disconnected][~self.disconnected_intersected]['contigLen']).mean(), # loc parameter
                                        1) # scale parameter
            self.update_umap_params(self.large_contigs[~self.disconnected][~self.disconnected_intersected].shape[0])

        except AttributeError:
            try:
                lognorm_cdf = sp_stats.lognorm.cdf(np.log(self.large_contigs[~self.disconnected]['contigLen']), 
                                                        np.log(self.large_contigs[~self.disconnected]['contigLen']).std(), # S - shape parameter
                                                        np.log(self.large_contigs[~self.disconnected]['contigLen']).mean(), # loc parameter
                                                        1) # scale parameter
                self.update_umap_params(self.large_contigs[~self.disconnected].shape[0])
            except AttributeError:
                lognorm_cdf = sp_stats.lognorm.cdf(np.log(self.large_contigs['contigLen']), 
                                                        np.log(self.large_contigs['contigLen']).std(), # S - shape parameter
                                                        np.log(self.large_contigs['contigLen']).mean(), # loc parameter
                                                        1) # scale parameter
                self.update_umap_params(self.large_contigs.shape[0])


        disconnection_stringent = max(lognorm_cdf.mean(), 0.05)
        self.depth_reducer.disconnection_distance = 0.25
        self.md_reducer.disconnection_distance = 0.15

        self.depth_reducer.n_neighbors = 5
        self.tnf_reducer.n_neighbors = 5
        self.md_reducer.n_neighbors = 5

        self.filter_value = disconnection_stringent
 
    def fit_disconnect(self):
        """
        Filter contigs based on ADP connections
        """
        ## Calculate the UMAP embeddings

        self.depths = np.nan_to_num(np.concatenate((self.large_contigs[~self.disconnected].iloc[:, 3:].drop(['tid'], axis=1),
                                                    self.log_lengths[~self.disconnected].values[:, None],
                                                    self.tnfs[~self.disconnected].iloc[:, 2:]), axis=1))
                                                    
        # Get all disconnected points, i.e. contigs that were disconnected in ANY mapping
        # logging.info("Running UMAP Filter - %s" % self.depth_reducer)
        depth_mapping = self.md_reducer.fit(self.depths)

        
        logging.info("Finding disconnections...")
        self.disconnected_intersected = umap.utils.disconnected_vertices(depth_mapping) #+ \
                                        # umap.utils.disconnected_vertices(tnf_mapping)


        # Only disconnect big contigs
        for i, dis in enumerate(self.disconnected_intersected):
            if dis:
                contig = self.large_contigs[~self.disconnected].iloc[i, :]
                if contig['contigLen'] < self.min_bin_size:
                    self.disconnected_intersected[i] = False
                    
        logging.info("Found %d disconnected points. %d TNF disconnected and %d ADP disconnected..." % 
                    (sum(self.disconnected) + sum(self.disconnected_intersected), sum(self.disconnected), sum(self.disconnected_intersected)))
        # logging.debug out all disconnected contigs
        pd.concat([self.large_contigs[self.disconnected],
                   self.large_contigs[~self.disconnected][self.disconnected_intersected]])\
            .to_csv(self.path + "/disconnected_contigs.tsv", sep="\t", header=True)

        return True

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
            self.intersection_mapper = depth_mapping * euc_mapping * tnf_mapping
        else:
            self.intersection_mapper = depth_mapping * tnf_mapping

        # self.embeddings = self.intersection_mapper.embedding_

    # def save_embedding(self):
    #     pd.DataFrame(self.embeddings).to_csv(self.path + '/embedding.tsv',
    #                                       sep="\t",
    #                                       header=False,
    #                                       index=False,)
    #
    # def read_embedding(self):
    #     self.embeddings = np.array(pd.read_csv(self.path))

    def sort_bins(self):
        """
        Helper functiont that sorts bin tids
        """
        bins = self.bins.keys()
        for bin in bins:
            tids = self.bins[bin]
            self.bins[bin] = list(np.sort(tids))

    def pairwise_distances(self, plots, n, x_min, x_max, y_min, y_max,
                           bin_unbinned=False, reembed=False,
                           size_only=False, big_only=False,
                           dissolve=False, debug=False):
        """
        Function for deciding whether a bin needs to be reembedded or split up
        Uses internal bin statistics, mainly mean ADP and Rho values
        """

        if self.n_samples > 0:
            n_samples = self.n_samples
            sample_distances = self.short_sample_distance
        else:
            n_samples = self.long_samples
            sample_distances = self.long_sample_distance

        bins_to_remove = []
        new_bins = {}
        new_bin_counter = 0
        logging.debug("Checking bin internal distances...")
        big_tids = []
        reembed_separately = [] # container for bin ids that look like chimeras
        force_new_clustering = []
        reembed_if_no_cluster = []
        problem_bin = None
        bins = self.bins.keys()
        for bin in bins:
            logging.debug("Beginning check on bin: ", bin)
            tids = self.bins[bin]
            if len(tids) == 1:
                continue
            elif bin == 0:
                continue


            # validity = self.bin_validity[bin]
            # if len(tids) != len(set(tids)):
            #     tids = list(set(tids))
            #     self.bins[bin] = tids
            if dissolve:

                contigs, log_lengths, tnfs = self.extract_contigs(tids)
                bin_size = contigs['contigLen'].sum()
                if bin_size < 1e6:
                    self.unbinned_tids = self.unbinned_tids + tids
                    bins_to_remove.append(bin)
                else:
                    try:
                        mean_md, \
                        mean_tnf, \
                        mean_euc, \
                        mean_agg, \
                        per_contig_avg = \
                            metrics.get_averages(np.concatenate((contigs.iloc[:, 3:].values,
                                                                 log_lengths.values[:, None],
                                                                 tnfs.iloc[:, 2:].values), axis=1),
                                                 n_samples,
                                                 sample_distances)

                        per_contig_avg = np.array(per_contig_avg)
                    except ZeroDivisionError:
                        # Only one contig left, break out
                        break

                    # IFF the bin is extra busted just obliterate it
                    if (mean_md >= 0.45 or mean_agg >= 0.45) and (mean_tnf >= 0.1 or mean_euc >= 3):
                        self.unbinned_tids = self.unbinned_tids + tids
                        bins_to_remove.append(bin)


            elif big_only:

                # filtering = True
                remove = False # Whether to completely remove original bin
                removed_single = [] # Single contig bin
                removed_together = [] # These contigs form their own bin
                contigs, log_lengths, tnfs = self.extract_contigs(tids)

                bin_size = contigs['contigLen'].sum()

                if bin_size >= 5e6:
                    # while filtering:

                    # Extract current contigs and get statistics
                    # contigs, log_lengths, tnfs = self.extract_contigs(tids)
                    try:
                        mean_md, \
                        mean_tnf, \
                        mean_euc, \
                        mean_agg, \
                        per_contig_avg = \
                            metrics.get_averages(np.concatenate((contigs.iloc[:, 3:].values,
                                                                 log_lengths.values[:, None],
                                                                 tnfs.iloc[:, 2:].values), axis=1),
                                                 n_samples,
                                                 sample_distances)

                        per_contig_avg = np.array(per_contig_avg)
                    except ZeroDivisionError:
                        # Only one contig left, break out
                        break

                    removed_inner = [] # inner container that is rewritten every iteration

                    if len(tids) == 2:
                        # Higher thresholds for fewer contigs
                        md_filt = 0.35
                        agg_filt = 0.4
                        euc_filt = 2
                        rho_filt = 0.05
                        # Two contigs by themselves that are relatively distant. Remove them separately
                        together = False
                    elif len(tids) <= 5:
                        # Higher thresholds for fewer contigs
                        md_filt = 0.35
                        agg_filt = 0.4
                        euc_filt = 2
                        rho_filt = 0.05
                        together = False
                    else:
                        # Lower thresholds for fewer contigs
                        md_filt = 0.35
                        agg_filt = 0.4
                        euc_filt = 2
                        rho_filt = 0.05
                        together = False



                    if mean_md >= 0.15 or mean_agg >= 0.3:
                        md_std = max(np.std(per_contig_avg[:, 0]), 0.15)
                        rho_std = max(np.std(per_contig_avg[:, 1]), 0.05)
                        euc_std = max(np.std(per_contig_avg[:, 2]), 0.5)
                        agg_std = max(np.std(per_contig_avg[:, 3]), 0.15)
                        for max_idx in range(per_contig_avg.shape[0]):
                            # max_idx = np.argmax(per_contig_avg[:, 3]) # Check mean_agg first
                            max_values = per_contig_avg[max_idx, :]
                            contig_length = contigs['contigLen'].iloc[max_idx]

                            if contig_length >= min(bin_size // 2, 3e6):
                                if (max_values[3] >= agg_filt or max_values[0] >= md_filt
                                    or max_values[3] >= (mean_agg + agg_std)
                                    or max_values[0] >= (mean_md + md_std)) and \
                                        ((max_values[1] >= rho_filt
                                          or max_values[1] >= (mean_tnf + rho_std))
                                         or (max_values[2] >= euc_filt
                                             or max_values[2] >= (mean_euc + euc_std))):
                                    if together:
                                        removed_inner.append(tids[max_idx])
                                        removed_together.append(tids[max_idx])
                                    else:
                                        removed_inner.append(tids[max_idx])
                                        removed_single.append(tids[max_idx])
                                elif (max_values[1] >= 0.2 or max_values[2] >= 4):
                                    if together:
                                        removed_inner.append(tids[max_idx])
                                        removed_together.append(tids[max_idx])
                                    else:
                                        removed_inner.append(tids[max_idx])
                                        removed_single.append(tids[max_idx])

                        if len(removed_inner) > 0:
                            [tids.remove(r) for r in removed_inner]


                    # logging.debug(filtering, len(removed_single), len(removed_together), bin)
                    if len(removed_single) > 0 or len(removed_together) > 0:
                        [big_tids.append(r) for r in removed_single]

                        new_bins[new_bin_counter] = []
                        [new_bins[new_bin_counter].append(r) for r in removed_together]
                        new_bin_counter += 1

                        current_contigs, current_lengths, current_tnfs = self.extract_contigs(tids)
                        if current_contigs['contigLen'].sum() <= self.min_bin_size:
                            [self.unbinned_tids.append(tid) for tid in tids]
                            remove = True

                        if bin in self.survived:
                            self.survived.remove(bin)

                        if len(tids) == 0 or remove:
                            bins_to_remove.append(bin)


            elif not size_only \
                and reembed \
                and bin != 0 \
                and len(tids) > 1:
                
                contigs, log_lengths, tnfs = self.extract_contigs(tids)
                bin_size = contigs['contigLen'].sum()

                if debug:
                    print(bin, bin_size)

                if bin not in self.survived:

                    try:
                        mean_md, \
                        mean_tnf, \
                        mean_euc, \
                        mean_agg, \
                        per_contig_avg = \
                            metrics.get_averages(np.concatenate((contigs.iloc[:, 3:].values,
                                                                    log_lengths.values[:, None],
                                                                    tnfs.iloc[:, 2:].values), axis=1),
                                                                    n_samples,
                                                                    sample_distances)
                    except ZeroDivisionError:
                        continue

                    removed = []

                    if debug:
                        print('before check for distant contigs: ', len(tids))
                        _, _, _, _ = self.compare_bins(bin)

                    if mean_md >= 0.15 or mean_agg >= 0.25:
                        # Simply remove
                        for (tid, avgs) in zip(tids, per_contig_avg):
                            if (avgs[0] >= 0.6 and
                                 (avgs[1] > 0.1 or avgs[2] >= 4.5)):
                                removed.append(tid)

                    remove = False
                    if len(removed) > 0 and len(removed) != len(tids):
                        new_bins[new_bin_counter] = []
                        [(tids.remove(r), new_bins[new_bin_counter].append(r)) for r in removed]
                        new_bin_counter += 1

                        current_contigs, current_lengths, current_tnfs = self.extract_contigs(tids)
                        if current_contigs['contigLen'].sum() <= self.min_bin_size:
                            [self.unbinned_tids.append(tid) for tid in tids]
                            remove = True

                        if len(tids) == 0 or remove:
                            bins_to_remove.append(bin)

                    if not remove:
                        f_level = 0.15
                        m_level = 0.15
                        shared_level = 0.1

                        if len(removed) >= 1:
                            # calc new bin size and stats
                            contigs, log_lengths, tnfs = self.extract_contigs(tids)
                            bin_size = contigs['contigLen'].sum()
                            try:
                                mean_md, \
                                mean_tnf, \
                                mean_euc, \
                                mean_agg, \
                                per_contig_avg = \
                                    metrics.get_averages(np.concatenate((contigs.iloc[:, 3:].values,
                                                                            log_lengths.values[:, None],
                                                                            tnfs.iloc[:, 2:].values), axis=1),
                                                                            n_samples,
                                                                            sample_distances)
                            except ZeroDivisionError:
                                continue

                            if debug:
                                print('contigs removed: ', len(tids))
                                _, _, _, _ = self.compare_bins(bin)

                        if ((mean_md >= m_level
                            or mean_agg >= f_level
                            or (mean_md >= shared_level and (mean_tnf >= shared_level or mean_euc >= 2))
                                    or ((mean_md >= 0.05 or mean_agg >= 0.15) and (mean_tnf >= 0.1 or mean_euc >= 2)))
                                and bin_size > 1e6) or bin_size >= 12e6:
                            logging.debug(bin, mean_md, mean_tnf, mean_agg, len(tids))
                            reembed_separately.append(bin)
                            if (((mean_md >= 0.35 or mean_agg >= 0.45) and (mean_tnf >= 0.1 or mean_euc >= 3)) \
                                    or bin_size >= 13e6):
                                if debug:
                                    logging.debug("Forcing bin %d" % bin)
                                    self.compare_bins(bin)
                                force_new_clustering.append(True) # send it to turbo hell
                                reembed_if_no_cluster.append(True)
                            elif bin_size > 1e6:
                                if debug:
                                    print("Reclustering bin %d" % bin)
                                force_new_clustering.append(False) # send it to regular hell
                                reembed_if_no_cluster.append(True)
                        else:
                            # reembed_separately.append(bin)
                            # force_new_clustering.append(False)  # send it to regular hell
                            # reembed_if_no_cluster.append(False) # take it easy, okay?
                            if debug:
                                print("bin survived %d" % bin)
                                self.compare_bins(bin)
                            self.survived.append(bin)
                else:
                    logging.debug(bin, self.survived)
                

                    
            elif size_only:
                logging.debug("Size only check when size only is ", size_only)
                contigs, log_lengths, tnfs = self.extract_contigs(tids)
                bin_size = contigs['contigLen'].sum()

                if bin_size >= 13e6 and bin!=0:
                    # larger than most bacterial genomes, way larger than archaeal
                    # Likely strains getting bunched together. But they won't disentangle, so just dismantle the bin
                    # rescuing any large contigs. Only way I can think  of atm to deal with this.
                    # Besides perhaps looking at variation level?? But this seems to be a problem with
                    # the assembly being TOO good.
                    if reembed:
                        reembed_separately.append(bin)
                        reembed_if_no_cluster.append(True)
                        force_new_clustering.append(True) # turbo hell
                elif bin_size >= 1e6 and bin!=0:
                    try:
                        mean_md, \
                        mean_tnf, \
                        mean_euc, \
                        mean_agg, \
                        per_contig_avg = \
                            metrics.get_averages(np.concatenate((contigs.iloc[:, 3:].values,
                                                                    log_lengths.values[:, None],
                                                                    tnfs.iloc[:, 2:].values), axis=1),
                                                                    n_samples,
                                                                    sample_distances)
                    except ZeroDivisionError:
                        continue

                    removed = []

                    if debug:
                        print('before check for distant contigs: ', len(tids))
                        _, _, _, _ = self.compare_bins(bin)

                    if mean_md >= 0.15 or mean_agg >= 0.25:
                        # Simply remove
                        for (tid, avgs) in zip(tids, per_contig_avg):
                            if (avgs[0] >= 0.6 and
                                    (avgs[1] > 0.1 or avgs[2] >= 4.5)):
                                removed.append(tid)

                    remove = False
                    if len(removed) > 0 and len(removed) != len(tids):
                        new_bins[new_bin_counter] = []
                        [(tids.remove(r), new_bins[new_bin_counter].append(r)) for r in removed]
                        new_bin_counter += 1

                        current_contigs, current_lengths, current_tnfs = self.extract_contigs(tids)
                        if current_contigs['contigLen'].sum() <= self.min_bin_size:
                            [self.unbinned_tids.append(tid) for tid in tids]
                            remove = True

                        if len(tids) == 0 or remove:
                            bins_to_remove.append(bin)

                    if not remove:
                        if len(removed) >= 1:
                            # calc new bin size and stats
                            contigs, log_lengths, tnfs = self.extract_contigs(tids)
                            bin_size = contigs['contigLen'].sum()
                            try:
                                mean_md, \
                                mean_tnf, \
                                mean_euc, \
                                mean_agg, \
                                per_contig_avg = \
                                    metrics.get_averages(np.concatenate((contigs.iloc[:, 3:].values,
                                                                            log_lengths.values[:, None],
                                                                            tnfs.iloc[:, 2:].values), axis=1),
                                                                            n_samples,
                                                                            sample_distances)
                            except ZeroDivisionError:
                                continue

                            if debug:
                                print('contigs removed: ', len(tids))
                                _, _, _, _ = self.compare_bins(bin)

                        if (mean_md >= 0.35 or mean_agg >= 0.45) and (mean_tnf >= 0.1 or mean_euc >= 3) \
                                and bin_size > 1e6:
                            if debug:
                                print("In final bit. ", bin)
                                self.compare_bins(bins)
                            reembed_separately.append(bin)
                            reembed_if_no_cluster.append(True)
                            force_new_clustering.append(True)  # send it to turbo hell
                        else:
                            # self.survived.append(bin)
                            pass

                # elif (self.large_contigs[self.large_contigs['tid'].isin(tids)]["contigLen"].sum() <= 1e6) \
                #         and bin != 0 \
                #         and not reembed:
                #     for tid in tids:
                #         self.unbinned_tids.append(tid)
                #     bins_to_remove.append(bin)

        try:
            max_bin_id = max(self.bins.keys()) + 1
        except ValueError:
            max_bin_id = 1
        for k, v in new_bins.items():
            self.bins[max_bin_id + k] = list(set(np.sort(np.array(v))))
            
        for bin, force_new, reembed_cluster in zip(reembed_separately, force_new_clustering, reembed_if_no_cluster):
            tids = self.bins[bin]

            logging.debug("Checking bin %d..." % bin)
            try:
                max_bin_id = max(self.bins.keys()) + 1
            except ValueError:
                max_bin_id = 1

            if isinstance(max_bin_id, np.int64):
                max_bin_id = max_bin_id.item()

            if bin == problem_bin:
                debug = True
            else:
                debug = False
                
            plots, remove = self.reembed(tids, max_bin_id, plots,
                                    x_min, x_max, y_min, y_max, n,
                                    force=force_new,
                                    reembed=reembed_cluster, debug=debug)
            if debug:
                print("Problem bin result... removing: ", remove)

            if remove:
                if debug:
                    print("Removing bin %d..." % bin)
                bins_to_remove.append(bin)
            elif force_new:
                logging.debug("Removing bin %d through force..." % bin)
                big_tids = big_tids + self.bins[bin]
                bins_to_remove.append(bin)
            else:
                if debug:
                    print("Keeping bin %d..." % bin)
                self.survived.append(bin)

        for k in bins_to_remove:
            try:
                result = self.bins.pop(k)
            except KeyError:
                pass

        try:
            max_bin_id = max(self.bins.keys()) + 1
        except ValueError:
            max_bin_id = 1

        if isinstance(max_bin_id, np.int64):
            max_bin_id = max_bin_id.item()

        for idx in big_tids:
            if self.large_contigs[self.large_contigs['tid'] == idx]['contigLen'].iloc[0] >= self.min_bin_size:
                max_bin_id += 1
                self.bins[max_bin_id] = [idx]
            else:
                try:
                    self.bins[0].append(idx)
                except KeyError:
                    self.bins[0] = [idx]

        if bin_unbinned:
            for idx in self.unbinned_tids:
                if self.large_contigs[self.large_contigs['tid'] == idx]['contigLen'].iloc[0] >= self.min_bin_size:
                    max_bin_id += 1
                    self.bins[max_bin_id] = [idx]
                else:
                    try:
                        self.bins[0].append(idx)
                    except KeyError:
                        self.bins[0] = [idx]

        return plots, n

    def extract_contigs(self, tids):
        contigs = self.large_contigs[self.large_contigs['tid'].isin(tids)]
        contigs = contigs.drop(['tid'], axis=1)
        log_lengths = np.log(contigs['contigLen']) / np.log(max(sp_stats.mstats.gmean(self.large_contigs['contigLen']), 10000))
        tnfs = self.tnfs[self.tnfs['contigName'].isin(contigs['contigName'])]

        return contigs, log_lengths, tnfs

    def cluster(self, distances, metric='euclidean', binning_method='eom',
                allow_single_cluster=False, prediction_data=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ## Cluster on the UMAP embeddings and return soft clusters
            tuned = utils.hyperparameter_selection(distances, self.threads,
                                                   metric=metric,
                                                   method=binning_method,
                                                   allow_single_cluster=allow_single_cluster,
                                                   starting_size = self.min_cluster_size)
            best = utils.best_validity(tuned)
            # logging.debug(best['min_cluster_size'], best['min_samples'])
            # try:
            clusterer = hdbscan.HDBSCAN(
                algorithm='best',
                alpha=1.0,
                approx_min_span_tree=True,
                gen_min_span_tree=True,
                leaf_size=40,
                cluster_selection_method=binning_method,
                metric=metric,
                min_cluster_size=int(best['min_cluster_size']),
                min_samples=int(best['min_samples']),
                allow_single_cluster=allow_single_cluster,
                core_dist_n_jobs=self.threads,
                prediction_data=prediction_data
            )
            clusterer.fit(distances)
            if prediction_data:
                self.soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
            return clusterer.labels_

    def reload(self, old_binning):
        self.disconnected = old_binning.disconnected
        self.disconnected_intersected = old_binning.disconnected_intersected
        self.embeddings = old_binning.embeddings
        self.unbinned_tids = []

    def iterative_clustering(self,
                             distances,
                             metric='euclidean',
                             binning_method='eom',
                             allow_single_cluster=False,
                             prediction_data=False,
                             double=True):
        if metric != "precomputed":
            try:
                first_labels = self.cluster(distances, metric=metric,
                                            allow_single_cluster=allow_single_cluster,
                                            prediction_data=prediction_data)
            except TypeError:
                first_labels = np.array([-1 for i in range(distances.shape[0])])
            # if prediction_data is False:
            bool_arr = np.array([True if i == -1 else False for i in first_labels])
            if len(distances[bool_arr]) >= 10 and double: # try as it might fail with low numbers of leftover contigs
                try:
                    # Try to get unbinned super small clusters if they weren't binned
                    second_labels = self.cluster(distances[bool_arr],
                                                 metric=metric,
                                                 allow_single_cluster=False,
                                                prediction_data=prediction_data)

                except TypeError:
                    bool_arr = np.array([False for _ in first_labels])
            else:
                bool_arr = np.array([False for _ in first_labels])
        else:
            first_labels = self.precomputed(distances)
                # first_labels = np.array([-1 for i in range(distances.shape[0])])
            bool_arr = np.array([False for _ in first_labels])

        #
        #
        main_labels = []  # container for complete clustering
        max_label = max(first_labels) + 1  # value to shift second labels by
        second_idx = 0  # current index in second labels
        #
        for first, second_bool in zip(first_labels, bool_arr):
            if first != -1:  # use main label
                main_labels.append(first)
            elif second_bool:
                second_label = second_labels[second_idx]
                if second_label != -1:
                    main_labels.append(max_label + second_labels[second_idx] + 1)
                else:
                    main_labels.append(-1)
                second_idx += 1
            else:  # Should never get here but just have it here in case
                main_labels.append(-1)

        return main_labels


    def precomputed(self, distances):
        """
        Helper function to perform basic HDBSCAN on precomputed matrix.
        Hyperparameter selection does not work on precomputed matrices, so default params
        are set very low to try and force a result.
        """
        clusterer = hdbscan.HDBSCAN(
            algorithm='best',
            alpha=1.0,
            metric="precomputed",
            min_cluster_size=2,
            min_samples=1,
            allow_single_cluster=True,
            core_dist_n_jobs=self.threads,
        )
        clusterer.fit(distances)
        return clusterer.labels_

    @staticmethod
    def _validity(labels, distances):
        """
        Calculates cluster validity using Density Based Cluster Validity from HDBSCAN

        Params:
            :labels:
                Cluster labels to test
            :distances:
                Either a pairwise distance matrix or UMAP embedding values for the provided contig labels
        """
        if len(set(labels)) > 1:
            try:
                cluster_validity, validity_indices = hdbscan.validity.validity_index(distances.astype(np.float64), np.array(labels), per_cluster_scores=True)
            except ValueError:
                cluster_validity, validity_indices = -1, [-1 for x in set(labels)]
        else:
            cluster_validity, validity_indices = -1, [-1]

        if math.isnan(cluster_validity):
            cluster_validity = -1

        return cluster_validity, validity_indices

    # def quick_cluster(self, tids):

    def reembed(self, tids, max_bin_id, plots,
                           x_min=20, x_max=20, y_min=20, y_max=20, n=0,
                           max_n_neighbours = 50,
                           delete_unbinned = False,
                           bin_unbinned=False,
                           force=False,
                           reembed=False,
                           skip_clustering=False,
                           debug=False):
        """
        Recluster -> Re-embedding -> Reclustering on the specified set of contigs
        Any clusters that look better than current cluster are kept and old cluster is thrown out
        Anything that doesn't get binned is thrown in the unbinned_tids list

        Params:
            :tids:
                List of contig target ids to be reclustered
            :max_bin:
                The current large bin key
            :plots:
                List of plots resulting from all previous embeddings. Gets turned into a gif
            :x_min, x_max, y_max, y_min:
                Parameters for the plot to keep all plots at the same aspect ratio
            :n:
                The current iteration
            :force:
                Whether to force the results even if they look bad. This is used when a cluster looks
                looks especially heinous or is too big. Use this param lightly, it can bust your bins for sure
            :reembed:
                Whether to use the re-embedding feature. Setting to false will skip out on UMAP. Can make things
                faster if original clustering is easy to disentangle, but setting to false can miss things

        ignore the other shit
        """
        remove = False
        noise = False
        precomputed = False # Whether the precomputed clustering was the best result
        tids = list(np.sort(tids))
        contigs, log_lengths, tnfs = self.extract_contigs(tids)
        original_size = contigs['contigLen'].sum()
        min_validity = 1


        if original_size >= 14e6:
            force = True

        if len(set(tids)) > 1:

            if not skip_clustering:
                unbinned_array = self.large_contigs[~self.disconnected][~self.disconnected_intersected]['tid'].isin(tids)
                unbinned_embeddings = self.embeddings[unbinned_array]

                if reembed:
                    self.min_cluster_size = 2
                else:
                    self.min_cluster_size = 2
                try:
                    labels_single = self.iterative_clustering(unbinned_embeddings,
                                                            allow_single_cluster=True,
                                                            prediction_data=False,
                                                            double=False)
                    labels_multi = self.iterative_clustering(unbinned_embeddings,
                                                             allow_single_cluster=False,
                                                             prediction_data=False,
                                                             double=False)

                    # if max_validity == -1:
                    # Try out precomputed method, validity metric does not work here
                    # so we just set it to 1 and hope it ain't shit. Method for this is
                    # not accept a clustering result with noise. Not great, but

                    if self.n_samples > 0:
                        n_samples = self.n_samples
                        sample_distances = self.short_sample_distance
                    else:
                        n_samples = self.long_samples
                        sample_distances = self.long_sample_distance

                    distances = metrics.distance_matrix(np.concatenate((contigs.iloc[:, 3:].values,
                                                            log_lengths.values[:, None],
                                                            tnfs.iloc[:, 2:].values), axis=1),
                                                            n_samples,
                                                            sample_distances)

                    labels_precomputed = self.iterative_clustering(distances, metric="precomputed")

                    validity_single, _ = self._validity(labels_single, unbinned_embeddings)
                    validity_multi, _ = self._validity(labels_multi, unbinned_embeddings)
                    validity_precom, _ = self._validity(labels_precomputed, unbinned_embeddings)

                    # Calculate silhouette scores, will fail if only one label
                    # Silhouette scores don't work too well with HDBSCAN though since it
                    # usually requires pretty uniform clusters to generate a value of use
                    try:
                        silho_single = sk_metrics.silhouette_score(unbinned_embeddings, labels_single)
                    except ValueError:
                        silho_single = -1

                    try:
                        silho_multi = sk_metrics.silhouette_score(unbinned_embeddings, labels_multi)
                    except ValueError:
                        silho_multi = -1

                    try:
                        silho_precom = sk_metrics.silhouette_score(distances, labels_precomputed)
                    except ValueError:
                        silho_precom = -1

                    max_single = max(validity_single, silho_single)
                    max_multi = max(validity_multi, silho_multi)
                    max_precom = max(validity_precom, silho_precom)

                    if debug:
                        print('Allow single cluster validity: ', max_single)
                        print('Allow multi cluster validity: ', max_multi)
                        print('precom cluster validity: ', max_precom)


                    if max_single == -1 and max_multi == -1 and max_precom == -1:
                        self.labels = labels_single
                        max_validity = -1
                        min_validity = 1
                    elif max(max_single, max_multi, max_precom) == max_precom:
                        self.labels = labels_precomputed
                        max_validity = max_precom
                        precomputed = True
                        min_validity = 0.7
                    elif max(max_single, max_multi) == max_single:
                        self.labels = labels_single
                        max_validity = max_single
                        min_validity = 0.85
                    else:
                        self.labels = labels_multi
                        max_validity = max_multi
                        min_validity = 0.85

                    # get original size of bin


                    set_labels = set(self.labels)

                    if debug:
                        print("No. of Clusters:", len(set_labels), set_labels)

                except IndexError:
                    # Index error occurs when doing recluster after adding disconnected TNF
                    # contigs. Since the embedding array does not contain the missing contigs
                    # as such, new embeddings have to be calculated
                    max_validity = 0
            else:
                max_validity = -1

            if max_validity < 0.95 and reembed and len(tids) >= 5:
                # Generate new emebddings if clustering seems fractured
                contigs, log_lengths, tnfs = self.extract_contigs(tids)

                try:

                    self.fit_transform(tids, max_n_neighbours)

                    new_embeddings = self.intersection_mapper.embedding_
                    labels_single = self.iterative_clustering(new_embeddings,
                                                              allow_single_cluster=True,
                                                              prediction_data=False,
                                                              double=False)
                    labels_multi = self.iterative_clustering(new_embeddings,
                                                             allow_single_cluster=False,
                                                             prediction_data=False,
                                                             double=False)

                    validity_single, _ = self._validity(labels_single, new_embeddings)
                    validity_multi, _ = self._validity(labels_multi, new_embeddings)

                    logging.debug('Allow single cluster validity: ', validity_single)
                    logging.debug('Allow multi cluster validity: ', validity_multi)

                    if validity_single >= validity_multi:
                        if max_validity <= validity_single:
                            if all(label == -1 for label in labels_single):
                                if debug:
                                    print('using non re-embedded...')
                            else:
                                unbinned_embeddings = new_embeddings
                                self.labels = labels_single
                                max_validity = validity_single
                                min_validity = 0.85
                                if precomputed:
                                    precomputed = False # No longer the best results
                        else:
                            if debug:
                                print('using non re-embedded... %f' % max_validity)
                    else:
                        if max_validity <= validity_multi:
                            if all(label == -1 for label in labels_multi):
                                logging.debug('using non re-embedded...')
                            else:
                                unbinned_embeddings = new_embeddings
                                self.labels = labels_multi
                                max_validity = validity_multi
                                min_validity = 0.85
                                if precomputed:
                                    precomputed = False # No longer the best results
                        else:
                            if debug:
                                print('using non re-embedded... %f' % max_validity)




                except TypeError:
                    pass

            set_labels = set(self.labels)

            if debug:
                print("No. of Clusters:", len(set_labels), set_labels)
                print("Max and min valid and noise: ", max_validity, min_validity, noise)

            findem = [
                'contig_108_pilon', 'contig_1250_pilon',
                'scaffold_1715_pilon', 'contig_1687_pilon', 'contig_1719_pilon', 'contig_1718_pilon']

            names = list(contigs['contigName'])
            indices = []
            for to_find in findem:
                try:
                    indices.append(names.index(to_find))
                except ValueError:
                    indices.append(-1)

            plots.append(utils.plot_for_offset(unbinned_embeddings, self.labels, x_min, x_max, y_min, y_max, n))
            color_palette = sns.color_palette('husl', max(self.labels) + 1)
            cluster_colors = [
                color_palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in self.labels
            ]

            fig = plt.figure()
            ax = fig.add_subplot(111)

            ## Plot large contig membership
            ax.scatter(unbinned_embeddings[:, 0],
                       unbinned_embeddings[:, 1],
                       s=7,
                       linewidth=0,
                       c=cluster_colors,
                       alpha=0.7)
            found = False
            for i, index in enumerate(indices):
                if index != -1:
                    ax.annotate(findem[i], xy=(unbinned_embeddings[index, 0], unbinned_embeddings[index, 1]),
                                xycoords='data')
                    found = True

            total_new_bins = len(set(self.labels))

            plt.gca().set_aspect('equal', 'datalim')
            plt.title(format('UMAP projection of unbinned contigs - %d: %d clusters %f %d' %
                             (n, total_new_bins, max_validity, precomputed)), fontsize=24)
            plt.savefig(self.path + '/UMAP_projection_of_unbinned.png')

            if found and len(tids) < 100:
                plt.savefig(self.path + '/problem_cluster_closeup.png')

            if delete_unbinned:
                self.unbinned_tids = []

            big_contig_counter = 0
            if noise:
                # Clustering was a bit funky, so put back into unbinned and pull out again
                self.unbinned_tids = self.unbinned_tids + tids
                remove = True
            elif (len(set_labels) == 1) or (max_validity < min_validity) and not force:
                if debug:
                    print("Labels are bad")
                # Reclustering resulted in single cluster or all noise,
                # either case just use original bin
                remove = False

            else:
                new_bins = {}
                unbinned = []

                for (idx, label) in enumerate(self.labels):
                    if label != -1:
                        bin_key = max_bin_id + label.item() + 1
                        if isinstance(bin_key, np.int64):
                            bin_key = bin_key.item()
                        try:
                            new_bins[bin_key].append(tids[idx])  # inputs values as tid
                        except KeyError:
                            new_bins[bin_key] = [tids[idx]]
                    else:
                        unbinned.append(tids[idx])
                if debug:
                    print("No. of new bins:", new_bins.keys())
                    print("No. unbinned: ", len(unbinned))

                split = True
                if not force: # How much of original bin was binned?
                    not_recovered = 0
                    new_bin_ids = []

                    for bin, new_tids in new_bins.items():
                        contigs, log_lengths, tnfs = self.extract_contigs(new_tids)
                        bin_size = contigs['contigLen'].sum()
                        # if bin_size < self.min_bin_size:
                        #     if debug:
                        #         print("Didn't recover enough: %d of %d" % (bin_size, original_size))
                        #     not_recovered += bin_size
                        # else:
                        new_bin_ids.append(bin)
                        logging.debug("Recovered enough: %d of %d" % (bin_size, original_size))

                    contigs, _, _ = self.extract_contigs(unbinned)
                    not_recovered += contigs['contigLen'].sum()

                    if not_recovered > original_size // 2 and not force:
                        logging.debug("Didn't recover enough: %d of %d, %.3f percent" %
                              (not_recovered, original_size, not_recovered / original_size))
                        split = False
                        remove = False
                    elif ((len(new_bin_ids) < 2 and max_validity < 0.9)
                          or max_validity < min_validity) \
                            and not force:
                        split = False
                        remove = False
                if split:
                    # new_bins = []
                    # Half the original input has been binned if reembedding
                    for bin, new_tids in new_bins.items():
                        new_tids = list(np.sort(new_tids))
                        contigs, log_lengths, tnfs = self.extract_contigs(new_tids)
                        bin_size = contigs['contigLen'].sum()
                        if (bin_size >= 1e6 and reembed) \
                                or (not reembed and bin_size >= self.min_bin_size) \
                                or (force and bin_size >= self.min_bin_size):
                            #  Keep this bin
                            if debug:
                                print("Removing original bin, keeping bin: ", bin)
                                print("Length: ", bin_size)
                            remove = True # remove original bin
                            self.bins[bin] = new_tids
                            self.overclustered = True
                            if bin_size >= 14e6:
                                self.overclustered = True
                        else:
                            # put into unbinned
                            if debug:
                                print("Not adding new bin: ", bin, bin_size)
                            unbinned = unbinned + new_tids


                    if len(unbinned) != len(tids):
                        logging.debug("New bin(s) added... Total bins: ", len(self.bins.keys()))
                        contigs, log_lengths, tnfs = self.extract_contigs(unbinned)
                        bin_size = contigs['contigLen'].sum()
                        if self.n_samples > 0:
                            n_samples = self.n_samples
                            sample_distances = self.short_sample_distance
                        else:
                            n_samples = self.long_samples
                            sample_distances = self.long_sample_distance
                        try:
                            _, \
                            _, \
                            _, \
                            mean_agg, \
                            per_contig_avg = \
                                metrics.get_averages(np.concatenate((contigs.iloc[:, 3:].values,
                                                                        log_lengths.values[:, None],
                                                                        tnfs.iloc[:, 2:].values), axis=1),
                                                                        n_samples,
                                                                        sample_distances)
                        except ZeroDivisionError:
                            mean_agg = 0

                        bin_id = max(self.bins.keys()) + 1
                        if bin_size >= 2e6 and mean_agg < 0.25: # just treat it as a bin
                            if debug:
                                print("Unbinned contigs are bin: %d of size: %d" % (bin_id, bin_size))
                            self.bins[bin_id] = unbinned
                        else:
                            for contig in contigs.itertuples():
                                if contig.contigLen >= 2e6:
                                    self.bins[bin_id] = [self.assembly[contig.contigName]]
                                    bin_id += 1
                                else:
                                    self.unbinned_tids.append(self.assembly[contig.contigName])

                    else:
                        remove = False
                        if debug:
                            print("No new bin added.")

                else:
                    remove = False
                    if debug:
                        print("No new bin added")
        else:
           try:
               max_bin_id = max(self.bins.keys())
           except ValueError:
               max_bin_id = 1
               
           for idx in tids:
               if self.large_contigs[self.large_contigs['tid'] == idx]['contigLen'].iloc[0] >= self.min_bin_size:
                   max_bin_id += 1
                   self.bins[max_bin_id] = [idx]
           

        try:
            max_bin_id = max(self.bins.keys())
        except ValueError:
            max_bin_id = 1

        if isinstance(max_bin_id, np.int64):
            max_bin_id = max_bin_id.item()

        if bin_unbinned:
            for idx in self.unbinned_tids:
                if self.large_contigs[self.large_contigs['tid'] == idx]['contigLen'].iloc[0] >= self.min_bin_size:
                    max_bin_id += 1
                    self.bins[max_bin_id] = [idx]
                else:
                    try:
                        self.bins[0].append(idx)
                    except KeyError:
                        self.bins[0] = [idx]

        return plots, remove

    def read_bin_file(self, bin_json):
        with open(bin_json) as bin_file:
            self.bins = json.load(bin_file)
            self.bins = {int(k):v for k, v in self.bins.items()}


    def compare_contigs(self, contig1, contig2):
        tnf1 = np.concatenate((np.log(self.large_contigs[self.large_contigs['contigName']==contig1]['contigLen'].values[:, None]) / np.log(self.large_contigs['contigLen'].mean()),
                                    self.tnfs[self.tnfs['contigName']==contig1].iloc[:, 2:]),
                                    axis=1)
        tnf2 = np.concatenate((np.log(self.large_contigs[self.large_contigs['contigName']==contig2]['contigLen'].values[:, None]) / np.log(self.large_contigs['contigLen'].mean()),
                                            self.tnfs[self.tnfs['contigName']==contig2].iloc[:, 2:]),
                                            axis=1)
        t_euc = metrics.tnf_euclidean(tnf1[0], tnf2[0])

        all1 = np.concatenate((self.large_contigs[self.large_contigs['contigName']==contig1].iloc[:, 3:], tnf1), axis = 1)
        all2 = np.concatenate((self.large_contigs[self.large_contigs['contigName']==contig2].iloc[:, 3:], tnf2), axis = 1)
        
        
        d1 = self.large_contigs[self.large_contigs['contigName']==contig1].iloc[:, 3:].values
        d2 = self.large_contigs[self.large_contigs['contigName']==contig2].iloc[:, 3:].values

        if self.long_samples > 0:
            long1 = self.long_depths[self.long_depths['contigName'] == contig1].iloc[:, 3:].values
            long2 = self.long_depths[self.long_depths['contigName'] == contig2].iloc[:, 3:].values
            print("Metabat long: ", metrics.metabat_distance(long1[0], long2[0], self.long_samples, self.long_sample_distance))
            print("Hellinger normal long: ", metrics.hellinger_distance_normal(long1[0], long2[0], self.long_samples, self.long_sample_distance))
            print("Hellinger poisson long: ", metrics.hellinger_distance_poisson(long1[0], long2[0], self.long_samples, self.long_sample_distance))
        print("N samples: ", self.n_samples)
        print("TNF Euclidean distance: ", t_euc)
        print("TNF Correlation: ", metrics.tnf_correlation(tnf1[0], tnf2[0]))
        print("TNF Rho: ", metrics.rho(tnf1[0], tnf2[0]))
        print("Aggegate score: ", metrics.aggregate_tnf(all1[0], all2[0], self.n_samples, self.short_sample_distance))
        print("Metabat: ", metrics.metabat_distance(d1[0], d2[0], self.n_samples, self.short_sample_distance))
        print("Hellinger normal: ", metrics.hellinger_distance_normal(d1[0], d2[0], self.n_samples, self.short_sample_distance))
        print("Hellinger poisson: ", metrics.hellinger_distance_poisson(d1[0], d2[0], self.n_samples, self.short_sample_distance))

    
    def compare_bins(self, bin_id):        
        if self.n_samples > 0:
            n_samples = self.n_samples
            sample_distances = self.short_sample_distance
        else:
            n_samples = self.long_samples
            sample_distances = self.long_sample_distance
            
        tids = self.bins[bin_id]
        contigs, log_lengths, tnfs = self.extract_contigs(tids)

        mean_md, \
        mean_tnf, \
        mean_euc, \
        mean_agg, \
        per_contig_avg = \
            metrics.get_averages(np.concatenate((contigs.iloc[:, 3:].values,
                                                   log_lengths.values[:, None],
                                                   tnfs.iloc[:, 2:].values), axis=1),
                                                   n_samples,
                                                   sample_distances)

        print("MAG size: ", contigs['contigLen'].sum())
        print("Mean TNF Rho: ", mean_tnf)
        print("Mean Aitchinson: ", mean_euc)
        print("Mean Agg: ", mean_agg)
        print("Mean metabat: ", mean_md)



        return mean_tnf, mean_agg, mean_md, per_contig_avg
            
        
    def plot(self, findem=None, plot_bin_ids=False):

        if findem is None:
            findem = []
        logging.info("Generating UMAP plot with labels")

        names = list(self.large_contigs[~self.disconnected][~self.disconnected_intersected]['contigName'])
        indices = []
        for to_find in findem:
            try:
                indices.append(names.index(to_find))
            except ValueError:
                indices.append(-1)

        label_set = set(self.labels)
        color_palette = sns.color_palette('husl', max(self.labels) + 1)
        cluster_colors = [
            color_palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in self.labels
        ]
# 
        # cluster_member_colors = [
            # sns.desaturate(x, p) for x, p in zip(cluster_colors, self.clusterer.probabilities_)
        # ]
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ## Plot large contig membership
        ax.scatter(self.embeddings[:, 0],
                   self.embeddings[:, 1],
                   s=7,
                   linewidth=0,
                   c=cluster_colors,
                   # c = self.clusterer.labels_,
                   alpha=0.7)

        if plot_bin_ids:
            plotted_label = []
            for i, label in enumerate(self.labels):
                if label != -1 and label not in plotted_label:
                    ax.annotate(str(label), xy=(self.embeddings[i, 0] - 0.5,
                                               self.embeddings[i, 1] - 0.5),
                                xycoords='data')
                    plotted_label.append(label)

        for i, index in enumerate(indices):
            if index != -1:
                ax.annotate(findem[i], xy=(self.embeddings[index, 0],
                                           self.embeddings[index, 1]),
                            xycoords='data')

        plt.gca().set_aspect('equal', 'datalim')
        plt.title(format('UMAP projection of contigs - 0: %d clusters' % (len(label_set))), fontsize=24)
        plt.savefig(self.path + '/UMAP_projection_with_clusters.png')


    def labels(self):
        try:
            return self.soft_clusters_capped.astype('int32')
        except AttributeError:
            return self.clusterer.labels_.astype('int32')

    def use_soft_clusters(self, contigs):
        """"""
        for (idx, label) in enumerate(self.labels):
            if label == -1:
                # if contigs['contigLen'].iloc[idx] < 2e6:
                soft_values = self.soft_clusters[idx]
                max_value, best_label = metrics.get_best_soft_value(soft_values)
                if max_value >= 0.1:
                    self.labels[idx] = best_label
        # pass


    def bin_contigs(self, assembly_file, min_bin_size=200000):
        logging.info("Binning contigs...")
        self.bins = {}

        self.unbinned_tids = []
        # self.unbinned_embeddings = []

        set_labels = set(self.labels)
        max_bin_id = max(set_labels)

        for (idx, label) in enumerate(self.labels):
            if label != -1:

                try:
                    self.bins[label.item() + 1].append(
                        self.assembly[self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 0]]) # inputs values as tid
                except KeyError:
                    # self.bin_validity[label.item() + 1] = self.validity_indices[label]
                    self.bins[label.item() + 1] = [self.assembly[self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 0]]]
            else:
                self.unbinned_tids.append(self.assembly[self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 0]])
                # self.unbinned_embeddings.append(self.embeddings[idx, :])



    def bin_filtered(self,  min_bin_size=200000, keep_unbinned=False, unbinned_only=False):
        """
        Bins out any disconnected vertices if they are of sufficient size
        """
        try:
            max_bin_id = max(self.bins.keys()) + 1
        except ValueError:
            max_bin_id = 1
        # disconnected points end up in bin 0

        if isinstance(max_bin_id, np.int64):
            max_bin_id = max_bin_id.item()
        if not unbinned_only:
            for (idx, contig) in self.large_contigs[self.disconnected].iterrows():
                if contig["contigLen"] >= min_bin_size:
                    self.bins[max_bin_id] = [self.assembly[contig["contigName"]]]
                    max_bin_id += 1
                elif not keep_unbinned:
                    try:
                        self.bins[0].append(self.assembly[contig["contigName"]])
                    except KeyError:
                        self.bins[0] = [self.assembly[contig["contigName"]]]
                else:
                    self.unbinned_tids.append(contig['tid'])

            for (idx, contig) in self.large_contigs[~self.disconnected][self.disconnected_intersected].iterrows():
                if contig["contigLen"] >= min_bin_size:
                    self.bins[max_bin_id] = [self.assembly[contig["contigName"]]]
                    max_bin_id += 1
                elif not keep_unbinned:
                    try:
                        self.bins[0].append(self.assembly[contig["contigName"]])
                    except KeyError:
                        self.bins[0] = [self.assembly[contig["contigName"]]]
                else:
                    self.unbinned_tids.append(contig['tid'])


        unbinned_contigs, _, _ = self.extract_contigs(self.unbinned_tids)
        for contig in unbinned_contigs.itertuples():
            if contig.contigLen >= min_bin_size:
                self.bins[max_bin_id] = [self.assembly[contig.contigName]]
                self.unbinned_tids.remove(self.assembly[contig.contigName])
                max_bin_id += 1
            elif not keep_unbinned:
                try:
                    self.bins[0].append(self.assembly[contig.contigName])
                except KeyError:
                    self.bins[0] = [self.assembly[contig.contigName]]
            else:
                pass

        if not keep_unbinned:
            self.bins[0] = list(set(self.bins[0]))
        else:
            self.unbinned_tids = list(np.sort(self.unbinned_tids))
        

    def bin_big_contigs(self, min_bin_size=200000):
        """
        Bins out any disconnected vertices if they are of sufficient size
        """
        try:
            max_bin_id = max(self.bins.keys()) + 1
        except ValueError:
            max_bin_id = 1
        
        for (idx, contig) in self.genomes.iterrows():
            if contig["contigLen"] >= min_bin_size:
                self.bins[max_bin_id] = [self.assembly[contig["contigName"]]]
                max_bin_id += 1

    def rescue_contigs(self, min_bin_size=200000):
        """
        If entire projection was disconnected or left N < n_neighbors of points
        Then bin out any contigs greater than min bin size
        """
        try:
            max_bin_id = max(self.bins.keys()) + 1
        except ValueError:
            max_bin_id = 1
        
        for (idx, contig) in self.large_contigs.iterrows():
            if contig["contigLen"] >= min_bin_size:
                self.bins[max_bin_id] = [self.assembly[contig["contigName"]]]
                max_bin_id += 1
                
        for (idx, contig) in self.small_contigs.iterrows():
            if contig["contigLen"] >= min_bin_size:
                self.bins[max_bin_id] = [self.assembly[contig["contigName"]]]
                max_bin_id += 1


    def write_bins(self, min_bin_size=200000):
        logging.info("Writing bin JSON...")
        # self.bins = {k.item():v if isinstance(k, np.int64) else k:v for k,v in self.bins.items()}
        writing_bins = {}
        for key, value in self.bins.items():
            if isinstance(key, int64):
                writing_bins[key.item()] = value
            else:
                writing_bins[key] = value

        with open(self.path + '/rosella_bins.json', 'w') as fp:
            json.dump(writing_bins, fp, cls=NpEncoder)

        # self.small_contigs.to_csv(self.path + '/rosella_small_contigs.tsv', sep='\t')
        # self.large_contigs.to_csv(self.path + '/rosella_large_contigs.tsv', sep='\t')



class NpEncoder(json.JSONEncoder):
    """
    Numpy data type encoder for serializing bin dictionary
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
