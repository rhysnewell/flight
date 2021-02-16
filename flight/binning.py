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
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from Bio import SeqIO
import skbio.stats.composition
import umap
from itertools import product, combinations
import scipy.stats as sp_stats
from sklearn.preprocessing import RobustScaler

# self imports
import flight.metrics as metrics
import flight.utils as utils

# Set plotting style
sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

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


def phelp():
    print("""
Usage:
rosella.py [SUBCOMMAND] ..

Subcommands:
fit
""")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

def spawn_count(idx, seq):
    tetras = {''.join(p): 0 for p in product('ATCG', repeat=4)}
    forward = str(seq).upper()
    reverse = str(seq.reverse_complement()).upper()
    for s in [forward, reverse]:
        for i in range(len(s[:-4])):
            tetra = s[i:i + 4]
            if all(i in tetra for i in ("A", "T", "C", "G")):
                tetras[tetra] += 1
    return pd.Series(tetras, name=idx)


def spawn_merge_low_n(idx, soft_clusters):
    second_max = sorted(soft_clusters, reverse=True)[1]
    try:
        next_label = index(soft_clusters, second_max)[0]
    except IndexError:
        next_label = -1

    return next_label, idx



###############################################################################
################################ - Classes - ##################################


class CustomHelpFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        return text.splitlines()

    def _get_help_string(self, action):
        h = action.help
        if '%(default)' not in action.help:
            if action.default != '' and \
               action.default != [] and \
               action.default != None \
               and action.default != False:
                if action.default is not argparse.SUPPRESS:
                    defaulting_nargs = [
                        argparse.OPTIONAL, argparse.ZERO_OR_MORE
                    ]

                    if action.option_strings or action.nargs in defaulting_nargs:

                        if '\n' in h:
                            lines = h.splitlines()
                            lines[0] += ' (default: %(default)s)'
                            h = '\n'.join(lines)
                        else:
                            h += ' (default: %(default)s)'
        return h

    def _fill_text(self, text, width, indent):
        return ''.join([indent + line for line in text.splitlines(True)])


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
            b=0.5,
            min_bin_size=200000,
            min_coverage=1,
            min_coverage_sum=1,
    ):
        # config.THREADING_LAYER = 'tbb'
        # config.NUMBA_NUM_THREADS = threads
        self.min_contig_size = min_contig_size
        self.min_bin_size = min_bin_size
        self.threads = threads
        set_num_threads(threads)
        # Open up assembly
        self.assembly = {} 
        self.assembly_names = {}
        for (tid, rec) in enumerate(SeqIO.parse(assembly, "fasta")):
            self.assembly[rec.id] = tid
            self.assembly_names[tid] = rec.id

        # initialize bin dictionary Label: Vec<Contig>
        self.bins = {}

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

        
        ## Handle TNFs
        self.tnfs = pd.read_csv(kmer_frequencies, sep='\t')
        self.tnfs = self.tnfs[self.tnfs['contigName'].isin(self.large_contigs['contigName'])]
        ## Divide by row sums to get frequencies
        # self.tnfs.iloc[:, 2:] = self.tnfs.iloc[:, 2:].rank(axis=1)
        self.tnfs.iloc[:, 2:] = self.tnfs.iloc[:, 2:].div(self.tnfs.iloc[:, 2:].sum(axis=1), axis=0)
        self.tnfs.iloc[:, 2:] = skbio.stats.composition.clr(self.tnfs.iloc[:, 2:].astype(np.float64) + 1)
        ## Set custom log base change for lengths
        self.log_lengths = np.log(self.tnfs['contigLen']) / np.log(min_contig_size)
        # self.tnfs.iloc[:, 2:] = self.tnfs.iloc[:, 2:].div(self.tnfs.iloc[:, 2:].max(axis=1), axis=0)
        
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
        b_tnf = 0.5

        # self.filterer_tnf = umap.UMAP(
                        # metric=metrics.tnf_euclidean,
                        # n_neighbors=n_neighbors,
                        # n_components=10,
                        # min_dist=0,
                        # disconnection_distance=5.0,
                        # set_op_mix_ratio=0.01,
                        # random_state=random_state,
                        # n_epochs=500,
                        # spread=0.5,
                        # a=a,
                        # b=b,
                    # )
        if self.n_samples >= 3 or self.long_samples >= 3:
            n_components = min(max(self.n_samples, self.long_samples), 5)
        else:
            n_components = 2

        self.tnf_reducer = umap.UMAP(
                        metric=metrics.rho,
                        # metric = "correlation",
                        n_neighbors=int(n_neighbors),
                        n_components=n_components,
                        min_dist=0,
                        # local_connectivity=5,
                        disconnection_distance=3.0,
                        set_op_mix_ratio=0.01,
                        random_state=random_state,
                        n_epochs=500,
                        spread=0.5,
                        a=a,
                        b=b,
                    )

        self.inv_reducer = umap.UMAP(
                        metric=metrics.inverse_correlation,
                        # metric = "correlation",
                        n_neighbors=int(n_neighbors),
                        n_components=15,
                        min_dist=0,
                        # local_connectivity=5,
                        disconnection_distance=0.5,
                        set_op_mix_ratio=0.01,
                        random_state=random_state,
                        n_epochs=500,
                        spread=0.5,
                        a=a,
                        b=b,
                    )
        
        if self.n_samples > 0:
            self.depth_reducer = umap.UMAP(
                metric=metrics.aggregate_tnf,
                metric_kwds={"n_samples": self.n_samples, "sample_distances": self.short_sample_distance},
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                # local_connectivity=5,
                disconnection_distance=0.75,
                random_state=random_state,
                set_op_mix_ratio=0.01,
                n_epochs=500,
                spread=0.5,
                a=a,
                b=b,
            )


            self.metabat_reducer = umap.UMAP(
                    metric=metrics.metabat_distance,
                    metric_kwds={"n_samples": self.n_samples, "sample_distances": self.short_sample_distance},
                    n_neighbors=30,
                    n_components=n_components,
                    min_dist=min_dist,
                    # local_connectivity=5,
                    # disconnection_distance=0.75,
                    random_state=random_state,
                    set_op_mix_ratio=0.01,
                    n_epochs=500,
                    spread=0.5,
                    a=a,
                    b=b,
                )
            
            if self.n_samples < 3:
                self.coverage_reducer = umap.UMAP(
                    metric="euclidean",
                    # metric_kwds={"n_samples": self.n_samples, "sample_distances": self.short_sample_distance},
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    min_dist=min_dist,
                    # local_connectivity=5,
                    disconnection_distance=3,
                    random_state=random_state,
                    set_op_mix_ratio=0.01,
                    n_epochs=500,
                    spread=0.5,
                    a=a,
                    b=b,
                )
            else:
                self.coverage_reducer = umap.UMAP(
                    metric=metrics.rho,
                    # metric_kwds={"sample_distances": self.short_sample_distance},
                    n_neighbors=n_neighbors,
                    n_components=n_components,
                    min_dist=min_dist,
                    # local_connectivity=5,
                    disconnection_distance=1,
                    random_state=random_state,
                    set_op_mix_ratio=0.01,
                    n_epochs=500,
                    spread=0.5,
                    a=a,
                    b=b,
                )

            self.precomputed_reducer = umap.UMAP(
                    metric='precomputed',
                    # metric_kwds={"n_samples": self.n_samples, "sample_distances": self.short_sample_distance},
                    n_neighbors=10,
                    n_components=n_components,
                    min_dist=0.0,
                    # local_connectivity=5,
                    # disconnection_distance=1,
                    random_state=random_state,
                    set_op_mix_ratio=0.01,
                    n_epochs=500,
                    spread=0.5,
                    # a=a,
                    # b=b,
                )


    def filter(self):

        try:
            # logging.info("Running UMAP - %s" % self.tnf_reducer)
            self.filterer = self.tnf_reducer.fit(np.concatenate((self.log_lengths.values[:, None], self.tnfs.iloc[:, 2:]), axis = 1))
            self.disconnected = umap.utils.disconnected_vertices(self.filterer)
            # self.disconnected = np.array([False for i in range(self.large_contigs.values.shape[0])])
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
        except AttributeError:
            try:
                lognorm_cdf = sp_stats.lognorm.cdf(np.log(self.large_contigs[~self.disconnected]['contigLen']), 
                                                        np.log(self.large_contigs[~self.disconnected]['contigLen']).std(), # S - shape parameter
                                                        np.log(self.large_contigs[~self.disconnected]['contigLen']).mean(), # loc parameter
                                                        1) # scale parameter
            except AttributeError:
                lognorm_cdf = sp_stats.lognorm.cdf(np.log(self.large_contigs['contigLen']), 
                                                        np.log(self.large_contigs['contigLen']).std(), # S - shape parameter
                                                        np.log(self.large_contigs['contigLen']).mean(), # loc parameter
                                                        1) # scale parameter
            
        disconnection_stringent = max(lognorm_cdf.mean(), 0.05)
        disconnection_param = 0.9
        # disconnection_param = 1
        tnf_disconnect = min(lognorm_cdf.mean() * 3, 0.9)
        self.tnf_reducer.disconnection_distance = tnf_disconnect
        self.coverage_reducer.disconnection_distance = tnf_disconnect
        self.metabat_reducer.disconnection_distance = disconnection_stringent

        if self.n_samples > 0:
            self.depth_reducer.disconnection_distance = disconnection_param
 
    def fit_disconnect(self):
        ## Calculate the UMAP embeddings

        

        logging.info("Running UMAP - %s" % self.tnf_reducer)
        tnf_mapping = self.tnf_reducer.fit(
                    np.concatenate((self.log_lengths[~self.disconnected].values[:, None],
                                    self.tnfs[~self.disconnected].iloc[:, 2:]),
                                    axis=1))
                                                    
        self.depths = np.nan_to_num(np.concatenate((self.large_contigs[~self.disconnected].iloc[:, 3:],
                                                    self.log_lengths[~self.disconnected].values[:, None],
                                                    self.tnfs[~self.disconnected].iloc[:, 2:]), axis=1))
                                                    
        # Get all disconnected points, i.e. contigs that were disconnected in ANY mapping
        logging.info("Running UMAP - %s" % self.metabat_reducer)
        depth_mapping = self.metabat_reducer.fit(self.large_contigs[~self.disconnected].iloc[:, 3:].values)

        if self.n_samples >= 3:
        
            logging.info("Running UMAP - %s" % self.coverage_reducer)
            coverage_mapping = self.coverage_reducer.fit(
                        np.concatenate((self.log_lengths[~self.disconnected].values[:, None],
                                skbio.stats.composition.clr(self.large_contigs[~self.disconnected].iloc[:, 3::2].T + 1).T), axis = 1))

            logging.info("Finding diconnections...")
            self.disconnected_intersected = umap.utils.disconnected_vertices(depth_mapping) + \
                                            umap.utils.disconnected_vertices(tnf_mapping) + \
                                            umap.utils.disconnected_vertices(coverage_mapping)
        else:
            logging.info("Finding diconnections...")
            self.disconnected_intersected = umap.utils.disconnected_vertices(depth_mapping) + \
                                            umap.utils.disconnected_vertices(tnf_mapping)


        # Print out all disconnected contigs
        pd.concat([self.large_contigs[self.disconnected],
                   self.large_contigs[~self.disconnected][self.disconnected_intersected]])\
            .to_csv(self.path + "/disconnected_contigs.tsv", sep="\t", header=True)

        if not self.disconnected_intersected.any() == True:
            
            # union_mapper = tnf_mapping +
            if self.n_samples >= 3:
                self.intersection_mapper = depth_mapping * tnf_mapping * coverage_mapping
            else:
                self.intersection_mapper = depth_mapping * tnf_mapping
            self.embeddings = self.intersection_mapper.embedding_
            return False
        else:
            return True

    def fit_transform(self):
        # update parameters to artificially high values to avoid disconnected vertices in the final manifold
        # self.tnf_reducer.disconnection_distance = 1
        
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

        if self.n_samples >= 3:
            coverage_mapping = self.coverage_reducer.fit(
                                    np.concatenate((self.log_lengths[~self.disconnected][~self.disconnected_intersected].values[:, None],
                                        skbio.stats.composition.clr(self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[:, 3::2].T + 1).T), axis = 1))

            ## Intersect all of the embeddings
            # union_mapping = tnf_mapping # all significant hits
            self.intersection_mapper = depth_mapping * tnf_mapping * coverage_mapping # - union_mapping # intersect connections base on significance in at least one metric
        else:
            self.intersection_mapper = depth_mapping * tnf_mapping

        self.embeddings = self.intersection_mapper.embedding_ 


    def pairwise_distances(self):
        if self.n_samples > 0:
            n_samples = self.n_samples
            sample_distances = self.short_sample_distance
        else:
            n_samples = self.long_samples
            sample_distances = self.long_sample_distance



        bins_to_remove = []
        new_bins = {}
        logging.info("Checking bin internal distances...")

        for bin, tids in self.bins.items():
            if len(tids) > 1:
                contigs = self.large_contigs.loc[tids, :]
                log_lengths = np.log(contigs['contigLen']) / np.log(self.min_contig_size)
                tnfs = self.tnfs[self.tnfs['contigName'].isin(contigs['contigName'])]

                if contigs['contigLen'].sum() >= 3e6:
                    try:
                        max_bin_id = max(new_bins) + 1
                    except ValueError:
                        max_bin_id = 1
                    mean_md, mean_tnf, mean_agg, distances, per_contig_avg = metrics.populate_matrix(np.concatenate((contigs.iloc[:, 3:].values, log_lengths.values[:, None], tnfs.iloc[:, 2:].values), axis=1),
                                                                                                n_samples,
                                                                                                sample_distances)

                    
                    if contigs['contigLen'].sum() >= 10e6 or mean_md >= 0.3 or mean_tnf >= 0.3 or mean_agg >= 0.3 or contigs['contigLen'].sum() < 2e5:
                        tids_to_remove = []
                        for k, v in per_contig_avg.items():
                            if (v[0] / contigs.values.shape[0]) >= 0.3 \
                            or (v[1] / contigs.values.shape[0]) >= 0.3 \
                            or (v[2] / contigs.values.shape[0]) >= 0.3 \
                            or self.large_contigs['contigLen'].loc[tids[k]] >= 2e6:
                                # remove this contig
                                removed = tids[k]
                                tids_to_remove.append(removed)
                                if self.large_contigs['contigLen'].loc[removed] >= 2e6:
                                    max_bin_id += 1
                                    try:
                                        new_bins[max_bin_id].append(removed) # inputs values as tid
                                    except KeyError:
                                        new_bins[max_bin_id] = [removed]
                                else: # attempt to rebin
                                    self.unbinned_tids.append(removed)
                        for tid in tids_to_remove:
                            r = tids.remove(tid)
                        if len(tids) == 0:
                            bins_to_remove.append(bin)
                        
                        

        for k in bins_to_remove:
            try:
                del self.bins[k]
            except KeyError:
                pass
                
        try:
            max_bin_id = max(self.bins.keys()) + 1
        except ValueError:
            max_bin_id = 1
        for k, v in new_bins.items():
            self.bins[max_bin_id + k] = v


    
    def recluster(self, distances, metric='euclidean', binning_method='eom', min_cluster_size=5, min_samples=2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ## Cluster on the UMAP embeddings and return soft clusters
            tuned = utils.hyperparameter_selection(distances, self.threads, metric=metric, method=binning_method, allow_single_cluster=False, starting_size = self.min_cluster_size)
            best = utils.best_validity(tuned)
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
                    allow_single_cluster=False,
                    core_dist_n_jobs=self.threads,
                    prediction_data=False
                )
                return clusterer.fit(distances).labels_
            except TypeError:
                return np.array([-1 for i in range(distances.shape[0])])

                
    def reembed_unbinned(self, tids, max_bin_id, bin_unbinned = False):
        if len(tids) > 1:
            contigs = self.large_contigs.loc[tids, :]
            log_lengths = np.log(contigs['contigLen']) / np.log(self.min_contig_size)
            tnfs = self.tnfs[self.tnfs['contigName'].isin(contigs['contigName'])]
            # try:
            self.depth_reducer.n_neighors = min(contigs.values.shape[0] // 2, 50)


            self.tnf_reducer.n_neighbors = min(contigs.values.shape[0] // 2, 50)

            tnf_mapping = self.tnf_reducer.fit(
                                            np.concatenate((log_lengths.values[:, None],
                                                            tnfs.iloc[:, 2:]),
                                                            axis=1))
            agg_mapping = self.depth_reducer.fit(np.concatenate((contigs.iloc[:, 3:], log_lengths.values[:, None], tnfs.iloc[:, 2:]), axis=1))
            if self.n_samples >= 3:
                self.coverage_reducer.n_neighbors = min(contigs.values.shape[0] // 2, 50)
                coverage_mapping = self.coverage_reducer.fit(
                                            np.concatenate((log_lengths.values[:, None],
                                                            skbio.stats.composition.clr(contigs.iloc[:, 3::2].T + 1).T),
                                                            axis=1))
                intersection = agg_mapping * tnf_mapping * coverage_mapping
            else:
                intersection = agg_mapping * tnf_mapping         
            labels = self.recluster(intersection.embedding_) 


            self.unbinned_tids = []
            for (idx, label) in enumerate(labels):
                if label != -1:
                    try:
                        self.bins[max_bin_id + label.item() + 1].append(
                            self.assembly[contigs.iloc[idx, 0]]) # inputs values as tid
                    except KeyError:
                        self.bins[max_bin_id + label.item() + 1] = [self.assembly[contigs.iloc[idx, 0]]]

                else:
                    self.unbinned_tids.append(idx)

            if bin_unbinned:
                logging.info("Generating UMAP plot with labels for unbinned")

                color_palette = sns.color_palette('husl', max(labels) + 1)
                cluster_colors = [
                    color_palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels
                ]

                fig = plt.figure()
                ax = fig.add_subplot(111)

                ## Plot large contig membership
                ax.scatter(intersection.embedding_[:, 0],
                           intersection.embedding_[:, 1],
                           s=7,
                           linewidth=0,
                           c=cluster_colors,
                           alpha=0.7)

                plt.gca().set_aspect('equal', 'datalim')
                plt.title('UMAP projection of unbinned contigs', fontsize=24)
                plt.savefig(self.path + '/UMAP_projection_of_unbinned.png')
                try:
                    max_bin_id = max(self.bins.keys()) + 1
                except ValueError:
                    max_bin_id = 1
                for idx in self.unbinned_tids:
                    if contigs.iloc[idx, 1] >= self.min_bin_size:
                        max_bin_id += 1
                        self.bins[max_bin_id] = [self.assembly[contigs.iloc[idx, 0]]]


                    
    def cluster(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ## Cluster on the UMAP embeddings and return soft clusters
            logging.info("Clustering contigs...")
            tuned = utils.hyperparameter_selection(self.embeddings, self.threads, method=self.binning_method, allow_single_cluster=False, starting_size = self.min_cluster_size)
            best = utils.best_validity(tuned)
            self.clusterer = hdbscan.HDBSCAN(
                algorithm='best',
                alpha=1.0,
                approx_min_span_tree=True,
                gen_min_span_tree=True,
                leaf_size=40,
                cluster_selection_method=self.binning_method,
                metric='euclidean',
                min_cluster_size=int(best['min_cluster_size']),
                min_samples=int(best['min_samples']),
                allow_single_cluster=False,
                core_dist_n_jobs=self.threads,
                prediction_data=True
            )
            self.clusterer.fit(self.embeddings)
            try:
                self.validity, self.cluster_validity = hdbscan.validity.validity_index(self.embeddings.astype(np.float64), self.clusterer.labels_, per_cluster_scores=True)
            except ValueError:
                self.validity = 0
                self.cluster_validity = [0.5 for i in range(len(set(self.clusterer.labels_)))]
            self.soft_clusters = hdbscan.all_points_membership_vectors(
                self.clusterer)
            self.soft_clusters_capped = np.array([np.argmax(x) for x in self.soft_clusters])

    def cluster_unbinned(self):
        ## Cluster on the unbinned contigs, attempt to create fine grained clusters that were missed

        logging.info("Clustering unbinned contigs...")
        tuned = utils.hyperparameter_selection(self.unbinned_embeddings, self.threads, method=self.binning_method)
        best = utils.best_validity(tuned)
        if best is not None:
            self.unbinned_clusterer = hdbscan.HDBSCAN(
                algorithm='best',
                alpha=1.0,
                approx_min_span_tree=True,
                gen_min_span_tree=True,
                leaf_size=40,
                cluster_selection_method=self.binning_method,
                metric='euclidean',
                min_cluster_size=int(best['min_cluster_size']),
                min_samples=int(best['min_samples']),
                allow_single_cluster=False,
                core_dist_n_jobs=self.threads,
                prediction_data=True
            )
            self.unbinned_clusterer.fit(self.unbinned_embeddings)
        else:
            self.unbinned_clusterer = None

    def compare_contigs(self, contig1, contig2):
        tnf1 = np.concatenate((np.log(self.large_contigs[self.large_contigs['contigName']==contig1]['contigLen'].values[:, None]) / np.log(10000),
                                    self.tnfs[self.tnfs['contigName']==contig1].iloc[:, 2:]),
                                    axis=1)
        tnf2 = np.concatenate((np.log(self.large_contigs[self.large_contigs['contigName']==contig2]['contigLen'].values[:, None]) / np.log(10000),
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

    def recluster_bin(self, bin_json, bin_id, binning_method='leaf', min_cluster_size=5, min_samples=2):
        if self.n_samples > 0:
            n_samples = self.n_samples
            sample_distances = self.short_sample_distance
        else:
            n_samples = self.long_samples
            sample_distances = self.long_sample_distance
            
        with open(bin_json) as bin_file:
            bin_dict = json.load(bin_file)
            tids = bin_dict[bin_id]
            contigs = self.large_contigs.loc[tids, :]
            log_lengths = np.log(contigs['contigLen']) / np.log(self.min_contig_size)
            tnfs = self.tnfs[self.tnfs['contigName'].isin(contigs['contigName'])]
            mean_md, t, a, distances = metrics.populate_matrix(np.concatenate((contigs.iloc[:, 3:].values, log_lengths.values[:, None], tnfs.iloc[:, 2:].values), axis=1),
                                                        n_samples,
                                                        sample_distances)
            # tuned = utils.hyperparameter_selection(tnfs.iloc[:, 2:].values, self.threads, metric='euclidean', method=self.binning_method)
            # best = utils.best_validity(tuned)
            clusterer = hdbscan.HDBSCAN(
                                algorithm='best',
                                alpha=1.0,
                                approx_min_span_tree=True,
                                gen_min_span_tree=True,
                                leaf_size=40,
                                cluster_selection_method=binning_method,
                                metric='precomputed',
                                min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                allow_single_cluster=False,
                                core_dist_n_jobs=self.threads,
                                prediction_data=False
                            )
            return tids, clusterer.fit(distances).labels_
    
    def compare_bins(self, bin_json, bin_id):
        tnfs = pd.DataFrame(np.concatenate((self.large_contigs['contigName'].values[:, None],
                                self.log_lengths.values[:, None],
                                self.tnfs.iloc[:, 2:].values),
                                axis=1))
        depths = pd.DataFrame(np.concatenate([self.large_contigs['contigName'].values[:, None], 
                                                    self.large_contigs.iloc[:, 3:].values, 
                                                    tnfs.iloc[:, 1:]], axis = 1))
        depths_only = pd.concat([self.large_contigs['contigName'], self.large_contigs.iloc[:, 3:]], axis = 1)
        # multivar = pd.concat([self.large_contigs['contigName'], pd.DataFrame(RobustScaler().fit_transform(self.large_contigs.iloc[:, 3::2])).apply(lambda x: sp_stats.multivariate_normal.cdf(x, mean = x.mean(), cov = np.cov(x.values.astype(np.float32))))], axis = 1)

        agg_vec = []
        tnf_vec = []
        cor_vec = []
        inv_vec = []
        rho_vec = []
        md_vec = []
        hn_vec = []
        hp_vec = []
        kl_vec = []
        
        with open(bin_json) as bin_file:
            bin_dict = json.load(bin_file)
            bin = bin_dict[bin_id]
            for (id1, id2) in combinations(bin, 2):
                contig1 = self.assembly_names[id1]
                contig2 = self.assembly_names[id2]
                try:
                    tnf_vec.append(metrics.tnf_euclidean(tnfs[tnfs[0] == contig1].iloc[:, 1:].values[0].astype(np.float64), 
                                                        tnfs[tnfs[0] == contig2].iloc[:, 1:].values[0].astype(np.float64)))
                    cor_vec.append(metrics.tnf_correlation(tnfs[tnfs[0] == contig1].iloc[:, 1:].values[0].astype(np.float64), 
                                                        tnfs[tnfs[0] == contig2].iloc[:, 1:].values[0].astype(np.float64)))
                    inv_vec.append(metrics.inverse_correlation(tnfs[tnfs[0] == contig1].iloc[:, 1:].values[0].astype(np.float64), 
                                                        tnfs[tnfs[0] == contig2].iloc[:, 1:].values[0].astype(np.float64)))
                    rho_vec.append(metrics.rho(tnfs[tnfs[0] == contig1].iloc[:, 1:].values[0].astype(np.float64), 
                                                        tnfs[tnfs[0] == contig2].iloc[:, 1:].values[0].astype(np.float64)))
                    agg_vec.append(metrics.aggregate_tnf(depths[depths[0] == contig1].iloc[:, 1:].values[0].astype(np.float64), 
                                                        depths[depths[0] == contig2].iloc[:, 1:].values[0].astype(np.float64), 
                                                        self.n_samples, self.short_sample_distance))
                    md_vec.append(metrics.metabat_distance(depths_only[depths_only['contigName'] == contig1].iloc[:, 1:].values[0].astype(np.float64), 
                                                                        depths_only[depths_only['contigName'] == contig2].iloc[:, 1:].values[0].astype(np.float64), 
                                                                        self.n_samples, self.short_sample_distance))
                    hn_vec.append(metrics.hellinger_distance_normal(depths_only[depths_only['contigName'] == contig1].iloc[:, 1:].values[0].astype(np.float64), 
                                                                                        depths_only[depths_only['contigName'] == contig2].iloc[:, 1:].values[0].astype(np.float64), 
                                                                                        self.n_samples, self.short_sample_distance))
                    hp_vec.append(metrics.hellinger_distance_poisson(depths_only[depths_only['contigName'] == contig1].iloc[:, 1:].values[0].astype(np.float64), 
                                                                                        depths_only[depths_only['contigName'] == contig2].iloc[:, 1:].values[0].astype(np.float64),  
                                                                                        self.n_samples, self.short_sample_distance))
                    # kl_vec.append(metrics.symmetric_kl(multivar[multivar['contigName'] == contig1].iloc[:, 1:].values[0].astype(np.float64),
                                                                                                            # multivar[multivar['contigName'] == contig2].iloc[:, 1:].values[0].astype(np.float64),
                                                                                                            # self.short_sample_distance))
                except IndexError:
                    print(id1, contig1)
                    print(id2, contig2)

        print("Mean TNF Euc: ", np.mean(tnf_vec))
        print("Mean TNF Cor: ", np.mean(cor_vec))
        print("Mean TNF InvCor: ", np.mean(inv_vec))
        print("Mean TNF Rho: ", np.mean(rho_vec))
        print("Mean Agg: ", np.mean(agg_vec))
        print("Mean metabat: ", np.mean(md_vec))
        print("Mean hellinger normal: ", np.mean(hn_vec))
        print("Mean hellinger poisson: ", np.mean(hp_vec))
        # print("Mean KL: ", np.mean(kl_vec))


        return tnf_vec, cor_vec, inv_vec, rho_vec, agg_vec, md_vec, hn_vec, hp_vec
            
        
    def plot(self):
        logging.info("Generating UMAP plot with labels")

        label_set = set(self.clusterer.labels_)
        color_palette = sns.color_palette('husl', max(self.clusterer.labels_) + 1)
        cluster_colors = [
            color_palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in self.clusterer.labels_
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

        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of contigs', fontsize=24)
        plt.savefig(self.path + '/UMAP_projection_with_clusters.png')

    def plot_distances(self):
        label_set = set(self.clusterer.labels_)
        self.clusterer.condensed_tree_.plot(
            select_clusters=True,
            selection_palette=sns.color_palette('deep', len(label_set)),
        )
        plt.title('Hierarchical tree of clusters', fontsize=24)
        plt.savefig(self.path + '/cluster_hierarchy.png')

    def labels(self):
        try:
            return self.soft_clusters_capped.astype('int32')
        except AttributeError:
            return self.clusterer.labels_.astype('int32')

    def bin_contigs(self, assembly_file, min_bin_size=200000):
        logging.info("Binning contigs...")
        self.bins = {}
        redo_bins = {}

        self.unbinned_tids = []
        self.unbinned_embeddings = []

        set_labels = set(self.clusterer.labels_)
        max_bin_id = max(set_labels)

        if len(set_labels) > 5:
            for (idx, label) in enumerate(self.clusterer.labels_):
                if label != -1:
                    # if self.cluster_validity[label] > 0.0:
                    try:
                        self.bins[label.item() + 1].append(
                            self.assembly[self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 0]]) # inputs values as tid
                    except KeyError:
                        self.bins[label.item() + 1] = [self.assembly[self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 0]]]

                elif self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 1] >= self.min_bin_size * 3:
                    max_bin_id += 1
                    try:
                        self.bins[max_bin_id.item()].append(
                            self.assembly[self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 0]]) # inputs values as tid
                    except KeyError:
                        self.bins[max_bin_id.item()] = [self.assembly[self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 0]]]
                else:
                    self.unbinned_tids.append(self.assembly[self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 0]])
                    # self.unbinned_embeddings.append(self.embeddings[idx, :])
        else:
            redo_binning = {}
            for (idx, label) in enumerate(self.clusterer.labels_):
                soft_label = self.soft_clusters_capped[idx]
                try:
                    redo_binning[soft_label + 1]["embeddings"].append(
                        self.embeddings[idx, :]) # inputs values as tid
                    redo_binning[soft_label + 1]["indices"].append(idx)  # inputs values as tid
                except KeyError:
                    redo_binning[soft_label + 1] = {}
                    redo_binning[soft_label + 1]["embeddings"] = [self.embeddings[idx, :]]
                    redo_binning[soft_label + 1]["indices"] = [idx]

            for (original_label, values) in redo_binning.items():
                new_labels = utils.break_overclustered(np.array(values["embeddings"]), self.threads)
                inner_bin_id = max(set(new_labels))
                
                try:
                    max_bin_id = max(self.bins.keys()) + 1
                except ValueError:
                    max_bin_id = 1
                for (idx, label) in zip(values["indices"], new_labels):
                    # Update labels
                    if label != -1:
                        self.clusterer.labels_[idx] = label.item() + max_bin_id
                        self.soft_clusters_capped[idx] = label.item() + max_bin_id
                        try:
                            self.bins[label.item() + max_bin_id].append(
                                self.assembly[self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 0]])  # inputs values as tid
                        except KeyError:
                            self.bins[label.item() + max_bin_id] = [self.assembly[self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 0]]]
                    elif self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 1] >= self.min_bin_size:
                        inner_bin_id += 1

                        try:
                            self.bins[inner_bin_id.item() + max_bin_id].append(
                                self.assembly[self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 0]]) # inputs values as tid
                        except KeyError:
                            self.bins[inner_bin_id.item() + max_bin_id] = [self.assembly[self.large_contigs[~self.disconnected][~self.disconnected_intersected].iloc[idx, 0]]]
                    else:
                        self.unbinned_indices.append(idx)
                        self.unbinned_embeddings.append(self.embeddings[idx, :])

        removed_bins = redo_bins.keys()
        if len(removed_bins) > 0:
            self.clusterer.labels_[:] = [label - sum(i < label for i in removed_bins) if label not in removed_bins else label for label in self.clusterer.labels_]
            


        # self.unbinned_embeddings = np.array(self.unbinned_embeddings)


    def bin_filtered(self, min_bin_size=200000):
        """
        Bins out any disconnected vertices if they are of sufficient size
        """
        try:
            max_bin_id = max(self.bins.keys()) + 1
        except ValueError:
            max_bin_id = 1
        
        for (idx, contig) in self.large_contigs[self.disconnected].iterrows():
            if contig["contigLen"] >= min_bin_size:
                self.bins[max_bin_id] = [self.assembly[contig["contigName"]]]
                max_bin_id += 1

        for (idx, contig) in self.large_contigs[~self.disconnected][self.disconnected_intersected].iterrows():
            if contig["contigLen"] >= min_bin_size:
                self.bins[max_bin_id] = [self.assembly[contig["contigName"]]]
                max_bin_id += 1

        for (idx, contig) in self.small_contigs.iterrows():
            if contig["contigLen"] >= min_bin_size:
                self.bins[max_bin_id] = [self.assembly[contig["contigName"]]]
                max_bin_id += 1
        

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

        with open(self.path + '/rosella_bins.json', 'w') as fp:
            json.dump(self.bins, fp)

        # self.small_contigs.to_csv(self.path + '/rosella_small_contigs.tsv', sep='\t')
        # self.large_contigs.to_csv(self.path + '/rosella_large_contigs.tsv', sep='\t')
