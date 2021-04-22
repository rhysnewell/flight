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
from sklearn.preprocessing import RobustScaler

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
            # variant_rates,
            output_prefix,
            assembly,
            scaler="clr",
            n_neighbors=100,
            min_dist=0.1,
            n_components=2,
            random_state=42,
            min_cluster_size=100,
            min_contig_size=2500,
            min_samples=50,
            prediction_data=True,
            cluster_selection_method="eom",
            precomputed=False,
            hdbscan_metric="euclidean",
            metric = 'aggregate_tnf',
            threads=8,
            a=1.58,
            b=0.4,
            min_bin_size=200000,
            min_coverage=1,
            min_coverage_sum=1,
            initialization='spectral',
    ):
        # config.THREADING_LAYER = 'tbb'
        # config.NUMBA_NUM_THREADS = threads
        self.min_contig_size = min_contig_size
        self.min_bin_size = min_bin_size
        self.threads = threads
        set_num_threads(threads)
        self.checked_bins = [] # Used in the pdist function
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


            # if self.n_samples < 3 or self.long_samples < 3:
             # Combine tables to get better clustering
            self.large_contigs = pd.concat([self.large_contigs, self.long_depths.iloc[:, 3:]], axis = 1)
            self.n_samples = len(self.large_contigs.columns[3::2])
            # self.long_samples = len(self.long_depths.columns[3::2])
            self.long_samples = 0
            # else:
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
        # self.tnfs.iloc[:, 2:] = self.tnfs.iloc[:, 2:].div(self.tnfs.iloc[:, 2:].sum(axis=1), axis=0)
        self.tnfs.iloc[:, 2:] = skbio.stats.composition.clr(self.tnfs.iloc[:, 2:].astype(np.float64) + 1)
        ## Set custom log base change for lengths
        self.log_lengths = np.log(self.tnfs['contigLen']) / np.log(max(sp_stats.mstats.gmean(self.tnfs['contigLen']), 10000))
        
        ## Check the ordering of the contig names for sanity check
        if list(self.large_contigs['contigName']) != list(self.tnfs['contigName']):
            sys.exit("Contig ordering incorrect for kmer table or coverage table")


        if self.long_samples > 0:
            logging.info("Longread samples found, applying strict contig filtering...")
            filter_level = 1
            disconnect = 0.1
            self.binning_method = 'eom'
            self.min_cluster_size = 2
            b_long = b        
        else:
            filter_level = 0.025
            disconnect = 0.1
            self.binning_method = 'eom'
            self.min_cluster_size = 2
            b_long = b


        if self.n_samples >= 3 or self.long_samples >= 3:
            n_components = min(max(self.n_samples, self.long_samples), 5)
        else:
            n_components = 2

        self.tnf_reducer = umap.UMAP(
            metric=metrics.rho,
            n_neighbors=int(n_neighbors),
            n_components=10,
            min_dist=0,
            disconnection_distance=0.9,
            set_op_mix_ratio=1,
            spread=0.5,
            a=a,
            b=b,
            init=initialization,
        )

        
        self.depth_reducer = umap.UMAP(
            metric=metrics.aggregate_tnf,
            metric_kwds={"n_samples": self.n_samples, "sample_distances": self.short_sample_distance},
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            disconnection_distance=0.9,
            set_op_mix_ratio=1,
            spread=0.5,
            a=a,
            b=b,
            init=initialization
        )

        self.md_reducer = umap.UMAP(
            metric=metrics.aggregate_md,
            metric_kwds={"n_samples": self.n_samples, "sample_distances": self.short_sample_distance},
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            disconnection_distance=0.9,
            set_op_mix_ratio=1,
            spread=0.5,
            a=a,
            b=b,
            init=initialization
        )

        self.update_umap_params(self.large_contigs.shape[0])

    def update_umap_params(self, nrows):
        if nrows <= 10000: # high gear
            # Small datasets can have larger n_neighbors without being prohibitively slow
            if nrows <= 1000: # wheels fell off
                self.tnf_reducer.n_neighbors = nrows // 10
                self.depth_reducer.n_neighbors = nrows // 10
                # self.depth_reducer.n_neighbors = nrows // 10

            else:
                self.tnf_reducer.n_neighbors = 100
                self.depth_reducer.n_neighbors = 100
            self.tnf_reducer.n_epochs = 500
            self.depth_reducer.n_epochs = 500
        elif nrows <= 50000: # mid gear
            # Things start to get too slow around here, so scale back params
            self.tnf_reducer.n_neighbors = 100
            self.tnf_reducer.n_epochs = 400
            self.depth_reducer.n_neighbors = 100
            self.depth_reducer.n_epochs = 400
        else: # low gear
            # This is the super slow zone, but don't want to dip values below this
            # Hopefully pick out easy bins, then scale data down with each iterations
            # Allowing the params to bump up into other gears
            self.tnf_reducer.n_neighbors = 100
            self.tnf_reducer.n_epochs = 300
            self.depth_reducer.n_neighbors = 100
            self.depth_reducer.n_epochs = 300
            self.depth_reducer.n_epochs = 300

    def filter(self):

        try:
            # logging.info("Running UMAP Filter - %s" % self.tnf_reducer)
            # self.filterer = self.tnf_reducer.fit(np.concatenate((self.log_lengths.values[:, None], self.tnfs.iloc[:, 2:]), axis = 1))
            # self.disconnected = umap.utils.disconnected_vertices(self.filterer)
            self.disconnected = np.array([False for i in range(self.large_contigs.values.shape[0])])
            # self.disconnected_intersected = np.array([False for i in range(self.large_contigs.values.shape[0])])
        except ValueError: # Everything was disconnected
            self.disconnected = np.array([True for i in range(self.large_contigs.values.shape[0])])

    def update_parameters(self):
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
        if self.n_samples >= 0:
            disconnection_param = min(lognorm_cdf.mean(), 0.99)
        else:
            disconnection_param = min(0.5, 0.9)
        tnf_disconnect = max(min(lognorm_cdf.mean(), 0.9), 0.1)
        self.tnf_reducer.disconnection_distance = 0.05
        self.depth_reducer.disconnection_distance = 0.25
        self.md_reducer.disconnection_distance = 0.1

        self.depth_reducer.n_neighbors = 5
        self.tnf_reducer.n_neighbors = 5
        self.md_reducer.n_neighbors = 5


        self.filter_value = disconnection_stringent
 
    def fit_disconnect(self):
        ## Calculate the UMAP embeddings

        self.depths = np.nan_to_num(np.concatenate((self.large_contigs[~self.disconnected].iloc[:, 3:].drop(['tid'], axis=1),
                                                    self.log_lengths[~self.disconnected].values[:, None],
                                                    self.tnfs[~self.disconnected].iloc[:, 2:]), axis=1))
                                                    
        # Get all disconnected points, i.e. contigs that were disconnected in ANY mapping
        logging.info("Running UMAP Filter - %s" % self.depth_reducer)
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
                    
        logging.info("Found %d disconnected points..." % (sum(self.disconnected) + sum(self.disconnected_intersected)))
        # Print out all disconnected contigs
        pd.concat([self.large_contigs[self.disconnected],
                   self.large_contigs[~self.disconnected][self.disconnected_intersected]])\
            .to_csv(self.path + "/disconnected_contigs.tsv", sep="\t", header=True)

        
            
        # if not self.disconnected_intersected.any() == True:
        #     logging.info("Running UMAP - %s" % self.tnf_reducer)
        #     tnf_mapping = self.tnf_reducer.fit(
        #                     np.concatenate((self.log_lengths[~self.disconnected].values[:, None],
        #                     self.tnfs[~self.disconnected].iloc[:, 2:]),
        #                     axis=1))
        #     self.intersection_mapper = depth_mapping * tnf_mapping
        #     self.embeddings = self.intersection_mapper.embedding_
        #     return False
        # else:
        #     return True
        return True

    def fit_transform(self):
        # update parameters to artificially high values to avoid disconnected vertices in the final manifold
        self.tnf_reducer.disconnection_distance = 0.99
        self.depth_reducer.disconnection_distance = 0.99
        self.md_reducer.disconnection_distance = 0.99

        self.tnf_reducer.n_neighbors = 100
        self.depth_reducer.n_neighbors = 100
        self.md_reducer.n_neighbors = 100

        self.update_umap_params(self.large_contigs[~self.disconnected][~self.disconnected_intersected].shape[0])
        
        logging.info("Running UMAP - %s" % self.tnf_reducer)
        tnf_mapping = self.tnf_reducer.fit(
                    np.concatenate((self.log_lengths[~self.disconnected][~self.disconnected_intersected].values[:, None],
                                    self.tnfs[~self.disconnected][~self.disconnected_intersected].iloc[:, 2:]),
                                    axis=1))
    
        
        logging.info("Running UMAP - %s" % self.depth_reducer)
        self.depths = self.depths[~self.disconnected_intersected]
        try:
            depth_mapping = self.depth_reducer.fit(self.depths)
        except ValueError:  # Sparse or low coverage contigs can cause high n_neighbour values to kark it
            self.depth_reducer.n_neighbors = 30
            depth_mapping = self.depth_reducer.fit(self.depths)

        if self.n_samples >= 0:
            ## Intersect all of the embeddings
            self.intersection_mapper = depth_mapping * tnf_mapping
        else:
            self.intersection_mapper = depth_mapping * tnf_mapping

        self.embeddings = self.intersection_mapper.embedding_ 


    def pairwise_distances(self, plots, n, x_min, x_max, y_min, y_max,
                           bin_unbinned=False, reembed=False, small_only=False):

        if self.n_samples > 0:
            n_samples = self.n_samples
            sample_distances = self.short_sample_distance
        else:
            n_samples = self.long_samples
            sample_distances = self.long_sample_distance

        bins_to_remove = []
        new_bins = {}
        logging.debug("Checking bin internal distances...")
        big_tids = []
        reembed_separately = [] # container for bin ids that look like chimeras
        allow_single_cluster_vec = []
        bins = self.bins.keys()
        for bin in bins:
            tids = self.bins[bin]
            # validity = self.bin_validity[bin]
            if len(tids) != len(set(tids)):
                print("Duplicate contigs in: ", bin, " Exiting...")
                tids = set(tids)
                
            if len(tids) > 1 \
                and not (bin_unbinned or reembed or small_only) \
                and bin != 0 \
                and bin not in self.checked_bins:
                contigs, log_lengths, tnfs = self.extract_contigs(tids)

                if contigs['contigLen'].sum() < 5e5:
                    for tid in tids:
                        # remove this contig
                        self.unbinned_tids.append(tid)

                    bins_to_remove.append(bin)

                elif contigs['contigLen'].sum() >= 13e6:
                    # larger than most bacterial genomes, way larger than archaeal
                    # Likely strains getting bunched together. But they won't disentangle, so just dismantle the bin
                    # rescuing any large contigs. Unbinned contigs get put into self.unbinned_tids to be wholly re-embedded
                    reembed_separately.append(bin)
                    allow_single_cluster_vec.append(False)


                else:
                    try:
                        mean_md, \
                        mean_tnf, \
                        mean_agg, \
                        per_contig_avg = \
                            metrics.populate_matrix(np.concatenate((contigs.iloc[:, 3:].values,
                                                                    log_lengths.values[:, None],
                                                                    tnfs.iloc[:, 2:].values), axis=1),
                                                                    n_samples,
                                                                    sample_distances)
                    except ZeroDivisionError:
                        logging.info("Something broke - %s" % tids)
                        sys.exit()

                    removed = []

                    if mean_md >= 0.3 or mean_agg >= 0.4 \
                            or all(x > 0.15 for x in [mean_md, mean_tnf]) \
                            or contigs['contigLen'].sum() >= 13e6:
                        [(self.unbinned_tids.append(tid), removed.append(tid)) for tid in tids]

                        remove = True

                        if len(tids) == 0 or remove:
                            bins_to_remove.append(bin)
                    # else:
                    #     removed = []
                    #     for (tid, avgs) in zip(tids, per_contig_avg):
                    #         if (avgs[0] >= 0.2 and avgs[1] >= 0.01) or \
                    #                 (avgs[1] >= 0.5 and avgs[0] >= 0.1) or avgs[2] >= 0.4:
                    #             # remove this contig
                    #             # if self.large_contigs[self.large_contigs['tid'] == tid]['contigLen'].iloc[0] >= 1e6:
                    #             #     big_tids.append(tid)
                    #             #     removed.append(tid)
                    #             # else:
                    #             self.unbinned_tids.append(tid)
                    #             removed.append(tid)

                    if len(removed) >= 1:
                        [tids.remove(r) for r in removed]
                        current_contigs, current_lengths, current_tnfs = self.extract_contigs(tids)
                        remove = False
                        if current_contigs['contigLen'].sum() <= self.min_bin_size:
                            [self.unbinned_tids.append(tid) for tid in tids]
                            remove = True

                        if len(tids) == 0 or remove:
                            bins_to_remove.append(bin)
                    else:
                        self.checked_bins.append(bin)



            elif not small_only and \
                    (reembed or bin_unbinned) \
                and bin != 0 \
                and len(tids) > 1 \
                and bin not in self.checked_bins:
                
                contigs, log_lengths, tnfs = self.extract_contigs(tids)

                if contigs['contigLen'].sum() < 5e5:
                    for tid in tids:
                        # remove this contig
                        self.unbinned_tids.append(tid)

                    bins_to_remove.append(bin)

                elif contigs['contigLen'].sum() >= 13e6: # larger than most bacterial genomes, way larger than archaeal
                    # Likely strains getting bunched together. But they won't disentangle, so just dismantle the bin
                    # rescuing any large contigs. Only way I can think  of atm to deal with this.
                    # Besides perhaps looking at variation level?? But this seems to be a problem with
                    # the assembly being TOO good.
                    if reembed:
                        reembed_separately.append(bin)
                        allow_single_cluster_vec.append(False)
                    else:
                        removed = []
                        for tid in tids:
                            if self.large_contigs[self.large_contigs['tid'] == tid]['contigLen'].iloc[0] >= 2e6:
                                big_tids.append(tid)
                                removed.append(tid)
                            else:
                                self.unbinned_tids.append(tid)
                                removed.append(tid)
                        # This cluster is extra busted. Likely black hole contigs
                        # [tids.remove(r) for r in removed]
                        # current_contigs, current_lengths, current_tnfs = self.extract_contigs(tids)
                        remove = True
                        # if current_contigs['contigLen'].sum() <= self.min_bin_size:
                        #     [self.unbinned_tids.append(tid) for tid in tids]
                        #     remove = True

                        if len(tids) == 0 or remove:
                            bins_to_remove.append(bin)

                else:
                    try:
                        mean_md, \
                        mean_tnf, \
                        mean_agg, \
                        per_contig_avg = \
                            metrics.populate_matrix(np.concatenate((contigs.iloc[:, 3:].values,
                                                                    log_lengths.values[:, None],
                                                                    tnfs.iloc[:, 2:].values), axis=1),
                                                                    n_samples,
                                                                    sample_distances)
                    except ZeroDivisionError:
                        logging.info("Something broke - %s" % tids)
                        sys.exit()

                    # Slight higher thresholds since bins that break here are completely dismantled
                    if reembed:
                        f_level = 0.45
                        m_level = 0.3
                        shared_level = 0.15
                    else:
                        f_level = 0.4
                        m_level = 0.25
                        shared_level = 0.1

                    removed = []
                    if (any(x >= m_level for x in [mean_md])) or mean_agg >= f_level \
                                or all(x >= shared_level for x in [mean_md, mean_tnf]):
                        if reembed and len(tids) >= 2:
                            # print(bin, mean_md, mean_tnf, mean_agg, len(tids))
                            reembed_separately.append(bin)
                            allow_single_cluster_vec.append(True)
                        else:
                            for tid in tids:
                                if self.large_contigs[self.large_contigs['tid'] == tid]['contigLen'].iloc[0] >= 1e6:
                                    big_tids.append(tid)
                                    removed.append(tid)
                                # else:
                                #     self.unbinned_tids.append(tid)
                                #     removed.append(tid)

                    # else:
                    #     in_reembed_already = False
                    #     for (tid, avgs) in zip(tids, per_contig_avg):
                    #         if (avgs[0] >= 0.15 and avgs[1] >= 0.05) or \
                    #                 (avgs[1] >= 0.5 and avgs[0] >= 0.1) or avgs[2] >= 0.4:
                    #             # remove this contig
                    #             if self.large_contigs[self.large_contigs['tid'] == tid]['contigLen'].iloc[0] >= 2e6:
                    #                 big_tids.append(tid)
                    #                 removed.append(tid)
                    #             elif not in_reembed_already:
                    #                 reembed_separately.append(bin)
                    #                 allow_single_cluster_vec.append(True)
                    #                 in_reembed_already = True

                    [tids.remove(r) for r in removed]
                    current_contigs, current_lengths, current_tnfs = self.extract_contigs(tids)
                    remove = False
                    if current_contigs['contigLen'].sum() <= self.min_bin_size:
                        [self.unbinned_tids.append(tid) for tid in tids]
                        remove = True

                    if len(tids) == 0 or remove:
                        bins_to_remove.append(bin)

                    
            elif self.large_contigs[self.large_contigs['tid'].isin(tids)]["contigLen"].sum() <= 1e6 and bin != 0:
                for tid in tids:
                    self.unbinned_tids.append(tid)
                bins_to_remove.append(bin)

        for k, v in new_bins.items():
            self.bins[k] = list(set(v))
            
        for bin, allow_single in zip(reembed_separately, allow_single_cluster_vec):
            tids = self.bins[bin]
            try:
                max_bin_id = max(self.bins.keys()) + 1
            except ValueError:
                max_bin_id = 1

            if isinstance(max_bin_id, np.int64):
                max_bin_id = max_bin_id.item()
                
            plots, remove = self.recluster_unbinned(tids, max_bin_id, plots,
                                    x_min, x_max, y_min, y_max, n,
                                    allow_single_cluster=allow_single) # don't plot results
            if remove:
                bins_to_remove.append(bin)

        for k in bins_to_remove:
            try:
                result = self.bins.pop(k)
            except KeyError:
                pass
            # try:
            #     result = self.bin_validity.pop(k)
            # except KeyError:
            #     pass
            

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

    def cluster(self, distances, metric='euclidean', binning_method='eom', min_cluster_size=2, min_samples=2, allow_single_cluster=False, prediction_data=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ## Cluster on the UMAP embeddings and return soft clusters
            tuned = utils.hyperparameter_selection(distances, self.threads, metric=metric, method=binning_method, allow_single_cluster=allow_single_cluster, starting_size = self.min_cluster_size)
            best = utils.best_validity(tuned)
            # print(best['min_cluster_size'], best['min_samples'])
            try:
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
            except TypeError:
                return np.array([-1 for i in range(distances.shape[0])])


    def iterative_clustering(self, distances, metric='euclidean', binning_method='eom', allow_single_cluster=False, prediction_data=False):
        first_labels = self.cluster(distances, allow_single_cluster=allow_single_cluster, prediction_data=prediction_data)
        # if prediction_data is False:
        # bool_arr = np.array([True if i == -1 else False for i in first_labels])
        # second_labels = self.cluster(distances[bool_arr], allow_single_cluster=allow_single_cluster)
        # else:
        bool_arr = np.array([False for i in first_labels])


        main_labels = []  # container for complete clustering
        max_label = max(first_labels) + 1  # value to shift second labels by
        second_idx = 0  # current index in second labels

        for first, second_bool in zip(first_labels, bool_arr):
            if first != -1:  # use main label
                main_labels.append(first)
            # elif second_bool:  # check what the cluster is
            #     second = second_labels[second_idx]
            #     if second != -1:
            #         main_labels.append(max_label + second)
            #     else:
            #         main_labels.append(-1)
            #     second_idx += 1
            else:  # Should never get here but just have it here in case
                main_labels.append(-1)

        return np.array(main_labels)
        

    def validity(self, labels, distances):
        if len(set(labels)) > 1:
            try:
                self.cluster_validity, self.validity_indices = hdbscan.validity.validity_index(distances.astype(np.float64), labels, per_cluster_scores=True)
            except ValueError:
                self.cluster_validity, self.validity_indices = 0, [1 for x in set(labels)]
        else:
            self.cluster_validity, self.validity_indices = 0, [-1]


    def recluster_unbinned(self, tids, max_bin_id, plots,
                           x_min, x_max, y_min, y_max, n,
                           delete_unbinned = False, bin_unbinned=False,
                           allow_single_cluster=False, reembed=False):
        remove = False
        if len(set(tids)) > 1:
            if not reembed or len(set(tids)) <= 10: # Just use old embeddings for speed
                unbinned_array = self.large_contigs[~self.disconnected][~self.disconnected_intersected]['tid'].isin(tids)
                unbinned_embeddings = self.embeddings[unbinned_array]
                self.labels = self.iterative_clustering(unbinned_embeddings,
                                                        allow_single_cluster=allow_single_cluster)
                contigs = self.large_contigs[~self.disconnected][~self.disconnected_intersected][unbinned_array]
                # self.use_soft_clusters(contigs)

            else: # Generate new emebddings for left over contigs
                contigs, log_lengths, tnfs = self.extract_contigs(tids)
                tnf_mapping = self.tnf_reducer.fit(
                    np.concatenate(
                        (log_lengths.values[:, None],
                         tnfs.iloc[:, 2:]),
                        axis=1))

                depth_mapping = self.depth_reducer.fit(np.concatenate(
                    (contigs.iloc[:, 3:], log_lengths.values[:, None], tnfs.iloc[:, 2:]), axis=1))

                self.intersection_mapper = depth_mapping * tnf_mapping

                unbinned_embeddings = self.intersection_mapper.embedding_
                self.labels = self.iterative_clustering(unbinned_embeddings, allow_single_cluster=allow_single_cluster)


            # if len(set(self.labels)) == 1 and -1 in set(self.labels):
            #     self.labels = np.array([0 for i in self.labels])
            # self.validity(self.labels, unbinned_embeddings)


            findem = ['contig_371_pilon', 'contig_3132_pilon',
                      'contig_3901_pilon', 'contig_846_pilon',
                      'contig_941_pilon', 'scaffold_49_pilon',
                      'contig_591_pilon', 'contig_2054_pilon', 'contig_910_pilon']

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
            plt.title(format('UMAP projection of unbinned contigs - %d: %d clusters' % (n, total_new_bins)), fontsize=24)
            plt.savefig(self.path + '/UMAP_projection_of_unbinned.png')

            if found and len(tids) < 100:
                plt.savefig(self.path + '/problem_cluster_closeup.png')

            if delete_unbinned:
                self.unbinned_tids = []

            big_contig_counter = 0
            set_labels = set(self.labels)
            if len(set_labels) == 1:
                # Reclustering resulted in single cluster or all noise,
                # either case just use original bin
                pass
            else:
                new_bins = {}
                unbinned = []

                for (idx, label) in enumerate(self.labels):
                    if label != -1:
                        bin_key = max_bin_id + label.item() + 1
                        if isinstance(bin_key, np.int64):
                            bin_key = bin_key.item()
                        try:
                            new_bins[bin_key].append(self.assembly[contigs[
                                'contigName'].iloc[idx]])  # inputs values as tid
                        except KeyError:
                            new_bins[bin_key] = [
                                self.assembly[contigs['contigName'].iloc[idx]]]
                    elif contigs['contigLen'].iloc[idx] >= self.min_bin_size:
                        bin_key = max_bin_id + total_new_bins + big_contig_counter
                        if isinstance(bin_key, np.int64):
                            bin_key = bin_key.item()
                        new_bins[bin_key] = [
                            self.assembly[contigs['contigName'].iloc[idx]]]
                        big_contig_counter += 1

                    else:
                        unbinned.append(self.assembly[contigs['contigName'].iloc[idx]])

                for bin, new_tids in new_bins.items():
                    contigs, log_lengths, tnfs = self.extract_contigs(new_tids)
                    bin_size = contigs['contigLen'].sum()
                    if bin_size > 1e6 or bin_unbinned:
                        #  Keep this bin
                        remove = True # remove original bin
                        self.bins[bin] = new_tids
                    else:
                        # put into unbinned
                        unbinned = unbinned + new_tids

                if len(unbinned) != len(tids):
                    self.unbinned_tids = self.unbinned_tids + unbinned
                else:
                    pass

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
        mean_agg, \
        per_contig_avg = \
            metrics.populate_matrix(np.concatenate((contigs.iloc[:, 3:].values,
                                                   log_lengths.values[:, None],
                                                   tnfs.iloc[:, 2:].values), axis=1),
                                                   n_samples,
                                                   sample_distances)

        print("MAG size: ", contigs['contigLen'].sum())
        print("Mean TNF Rho: ", mean_tnf)
        print("Mean Agg: ", mean_agg)
        print("Mean metabat: ", mean_md)



        return mean_tnf, mean_agg, mean_md, per_contig_avg
            
        
    def plot(self):
        logging.info("Generating UMAP plot with labels")

        findem = ['contig_371_pilon', 'contig_3132_pilon',
                  'contig_3901_pilon', 'contig_846_pilon',
                  'contig_941_pilon', 'scaffold_49_pilon',
                  'contig_591_pilon', 'contig_2054_pilon']
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

        for i, index in enumerate(indices):
            if index != -1:
                ax.annotate(findem[i], xy=(self.embeddings[index, 0], self.embeddings[index, 1]), xycoords='data')

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
                if contigs['contigLen'].iloc[idx] < 2e6:
                    soft_values = self.soft_clusters[idx]
                    max_value, best_label = metrics.get_best_soft_value(soft_values)
                    if max_value >= 0.5:
                        self.labels[idx] = best_label
        # pass


    def bin_contigs(self, assembly_file, min_bin_size=200000):
        logging.info("Binning contigs...")
        self.bins = {}

        self.unbinned_tids = []
        # self.unbinned_embeddings = []

        set_labels = set(self.labels)
        max_bin_id = max(set_labels)

        if len(set_labels) > 5:
            for (idx, label) in enumerate(self.labels):
                if label != -1:

                    try:
                        self.bins[label.item() + 1].append(
                            self.assembly[self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 0]]) # inputs values as tid
                    except KeyError:
                        # self.bin_validity[label.item() + 1] = self.validity_indices[label]
                        self.bins[label.item() + 1] = [self.assembly[self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 0]]]
                elif self.large_contigs[~self.disconnected][~self.disconnected_intersected]['contigLen'].iloc[idx] >= 2e6:
                    # self.bin_validity[max_bin_id + 1] = 1
                    self.bins[max_bin_id + 1] = [self.assembly[self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 0]]]
                    max_bin_id += 1
                else:
                    self.unbinned_tids.append(self.assembly[self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 0]])
                    # self.unbinned_embeddings.append(self.embeddings[idx, :])



    def bin_filtered(self, min_bin_size=200000):
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
                    
        for (idx, contig) in self.large_contigs[self.disconnected].iterrows():
            if contig["contigLen"] >= min_bin_size:
                self.bins[max_bin_id] = [self.assembly[contig["contigName"]]]
                max_bin_id += 1
            try:
                self.bins[0].append(self.assembly[contig["contigName"]])
            except KeyError:
                self.bins[0] = [self.assembly[contig["contigName"]]]

        for (idx, contig) in self.large_contigs[~self.disconnected][self.disconnected_intersected].iterrows():
            if contig["contigLen"] >= min_bin_size:
                self.bins[max_bin_id] = [self.assembly[contig["contigName"]]]
                max_bin_id += 1
            try:
                self.bins[0].append(self.assembly[contig["contigName"]])
            except KeyError:
                self.bins[0] = [self.assembly[contig["contigName"]]]

        for (idx, contig) in self.small_contigs.iterrows():
            if contig["contigLen"] >= min_bin_size:
                self.bins[max_bin_id] = [self.assembly[contig["contigName"]]]
                max_bin_id += 1
            try:
                self.bins[0].append(self.assembly[contig["contigName"]])
            except KeyError:
                self.bins[0] = [self.assembly[contig["contigName"]]]
        

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
