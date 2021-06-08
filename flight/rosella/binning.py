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

import json
import logging
###############################################################################
# System imports
import sys

import matplotlib
import matplotlib.pyplot as plt
# Function imports
import numpy as np
import pandas as pd
import scipy.stats as sp_stats
import seaborn as sns
import skbio.stats.composition
import umap
from Bio import SeqIO
from numba import njit
from numpy import int64

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
################################ - Functions - ################################

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

###############################################################################
################################ - Classes - ##################################

class Binner:
    def __init__(
            self,
            count_path,
            kmer_frequencies,
            output_prefix,
            assembly,
            long_count_path=None,
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


        # Embedder options
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.a = a
        self.b = b
        self.initialization = initialization
        self.random_seed = random_seed
        self.disconnected = None
        self.disconnected_intersected = None
        self.intersection_mapper = None
        self.embeddings = None

        # Clusterer options
        self.unbinned_tids = []
        self.labels = None
        self.soft_clusters = None

        # Validator options
        self.overclustered = False  # large cluster


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
        self.depth_reducer.disconnection_distance = 0.25
        self.md_reducer.disconnection_distance = 0.15

        self.depth_reducer.n_neighbors = 5
        self.tnf_reducer.n_neighbors = 5
        self.md_reducer.n_neighbors = 5


    def sort_bins(self):
        """
        Helper functiont that sorts bin tids
        """
        bins = self.bins.keys()
        for bin_id in bins:
            tids = self.bins[bin_id]
            self.bins[bin_id] = list(np.sort(tids))

    def extract_contigs(self, tids):
        contigs = self.large_contigs[self.large_contigs['tid'].isin(tids)]
        contigs = contigs.drop(['tid'], axis=1)
        log_lengths = np.log(contigs['contigLen']) / np.log(max(sp_stats.mstats.gmean(self.large_contigs['contigLen']), 10000))
        tnfs = self.tnfs[self.tnfs['contigName'].isin(contigs['contigName'])]

        return contigs, log_lengths, tnfs

    def reload(self, old_binning):
        self.disconnected = old_binning.disconnected
        self.disconnected_intersected = old_binning.disconnected_intersected
        self.embeddings = old_binning.embeddings
        self.unbinned_tids = []

    def get_labels_from_bins(self):
        """
        Takes bin ids from dictionary and turns them back into labels, keeping the labels in the same order
        as self.embeddings
        """
        contigs = self.large_contigs[~self.disconnected][~self.disconnected_intersected]
        self.labels = np.array([-1 for _ in range(contigs.shape[0])])
        for bin_id, tids in self.bins.items():
            if bin_id != 0:
                truth_array = contigs['tid'].isin(tids)
                self.labels[truth_array] = bin_id

    def add_plot(self, plots, unbinned_embeddings, contigs,
                 n = 0, x_min = 20, x_max = 20, y_min = 20, y_max = 20, max_validity = -1, precomputed = False):

        findem = [
            'contig_1357_pilon', 'contig_1570_pilon', 'contig_810_pilon', 'scaffold_1358_pilon']

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

        if found:
            plt.savefig(self.path + '/problem_cluster_closeup.png')

        return plots


    def read_bin_file(self, bin_json):
        with open(bin_json) as bin_file:
            self.bins = json.load(bin_file)
            self.bins = {int(k):v for k, v in self.bins.items()}


    def compare_contigs(self, tid1, tid2, n_samples, sample_distances, debug=False):
        depth1, log_length1, tnfs1 = self.extract_contigs([tid1])
        depth2, log_length2, tnfs2 = self.extract_contigs([tid2])
        w = (n_samples) / (n_samples + 1)  # weighting by number of samples same as in metabat2

        contig1 = np.concatenate((depth1.iloc[:, 3:].values,
                                     log_length1.values[:, None],
                                     tnfs1.iloc[:, 2:].values), axis=1)

        contig2 = np.concatenate((depth2.iloc[:, 3:].values,
                                  log_length2.values[:, None],
                                  tnfs2.iloc[:, 2:].values), axis=1)

        md = metrics.metabat_distance(
            contig1[0, :n_samples * 2],
            contig2[0, :n_samples * 2],
            n_samples, sample_distances
        )

        rho = metrics.rho(
            contig1[0, n_samples * 2:],
            contig2[0, n_samples * 2:],
        )

        euc = metrics.tnf_euclidean(
            contig1[0, n_samples * 2:],
            contig2[0, n_samples * 2:],
        )

        agg = np.sqrt((md ** w) * (rho ** (1 - w)))

        if debug:
            print("Tid compared to other tid: ", tid1, tid2, md, rho, euc, agg)

        return md, rho, euc, agg

    def compare_contig_to_bin(self, tid, current_depths, bin_id, n_samples, sample_distances, debug=False):
        """
        Compares a given contig to given bin. If the bin has one contig, return the distances of current bin
        to that bin. If the bin has multiple contigs, only return the distances if the contig being compared
        had a mean distance less than or equal to the mean distance of all points in the bin already

        :returns: distances to bin or None
        """
        logging.debug("Beginning check on bin: ", bin_id)
        tids = self.bins[bin_id]
        if tid not in tids:
            if len(tids) == 1:
                return self.compare_contigs(tid, tids[0], n_samples, sample_distances, debug)

            elif bin_id == 0:
                return None
            else:
                contigs, log_lengths, tnfs = self.extract_contigs(tids)

                try:
                    other_depths = np.concatenate((contigs.iloc[:, 3:].values,
                                                    log_lengths.values[:, None],
                                                    tnfs.iloc[:, 2:].values), axis=1)
                    mean_md, \
                    mean_rho, \
                    mean_euc, \
                    mean_agg,\
                        _ = \
                        metrics.get_averages(other_depths,
                                             n_samples,
                                             sample_distances)

                    other_md, \
                    other_rho, \
                    other_euc, \
                    other_agg = \
                        metrics.get_single_contig_averages(
                                            current_depths,
                                            other_depths,
                                            n_samples,
                                            sample_distances
                                        )
                    if debug:
                        print("Tid compared to other bin: ", tid, bin_id, other_md, other_rho, other_euc, other_agg)
                    if other_md <= mean_md and (other_rho <= mean_rho or other_euc <= mean_euc):
                        if (other_md <= 0.3 and (other_rho <= 0.2 or other_euc <= 4)) and other_agg <= 0.5:
                            if debug:
                                print("Moving contig to: ", bin_id)
                            return other_md, other_rho, other_euc, other_agg
                        else:
                            return None
                    else:
                        return None

                except ZeroDivisionError:
                    # Only one contig left, break out
                    return None
        else:
            return None


    def find_best_bin_for_contig(self, tid, current_bin, current_depths, n_samples, sample_distances, debug=False):
        """
        Finds the bin for which the contig had the best concordance with depending on the the ADP and
        rho or euclidean distance values
        :params:
            @tid - contig id to look at
            @current_bin - the contigs current bin_id
            @current_depths - the current depth matrix of the current bin

        :returns: bin_id or None
        """
        bins = self.bins.keys()
        result = None

        if current_bin != 0 and current_depths.shape[0] > 1:
            result = self.compare_contig_to_bin(
                tid,
                current_depths,
                current_bin,
                n_samples,
                sample_distances
            )
            if result is None:
                # Original bin had only one contig
                current_md, current_rho, current_euc, current_agg = -1, -1, -1, -1
            else:
                if debug:
                    print("Tid compared to own bin in finding best bin: ", tid, result)
                current_md, current_rho, current_euc, current_agg = result
        else:
            # original bin was unbinned contigs
            current_md, current_rho, current_euc, current_agg = -1, -1, -1, -1

        best_md = -1
        best_rho = -1
        best_euc = -1
        best_agg = -1
        best_bin_id = -1

        if debug:
            print("Before testing: ", tid, result)

        for bin_id in bins:
            if bin_id == 0 or bin_id == current_bin:
                continue
            else:
                result = self.compare_contig_to_bin(
                    tid,
                    current_depths,
                    bin_id,
                    n_samples,
                    sample_distances,
                    debug
                )
                if result is None:
                    if debug:
                        print("Found nothing: ", tid, result)
                    continue
                else:
                    if debug:
                        print("Found something: ", tid, result)
                    other_md, other_rho, other_euc, other_agg = result
                    if best_md == -1:
                        best_md, best_rho, best_euc, best_agg = other_md, other_rho, other_euc, other_agg
                        best_bin_id = bin_id
                    elif other_md < best_md and (other_rho < best_rho or other_euc < best_euc):
                        best_md, best_rho, best_euc, best_agg = other_md, other_rho, other_euc, other_agg
                        best_bin_id = bin_id

        if best_bin_id != -1:
            if current_md == -1:
                if best_md <= 0.3 and (best_rho <= 0.15 or best_euc <= 3):
                    return best_bin_id

            elif best_md <= current_md and (best_rho <= current_rho or best_euc <= current_euc):
                # if best_md <= 0.3 and (best_rho <= 0.15 or best_euc <= 3):
                return best_bin_id

        return None


    def get_n_samples_and_distances(self):
        if self.n_samples > 0:
            n_samples = self.n_samples
            sample_distances = self.short_sample_distance
        else:
            n_samples = self.long_samples
            sample_distances = self.long_sample_distance

        return n_samples, sample_distances

    def bin_stats(self, bin_id):
        n_samples, sample_distances = self.get_n_samples_and_distances()
            
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

        for i, idx in enumerate(indices):
            if idx != -1:
                ax.annotate(findem[i], xy=(self.embeddings[idx, 0],
                                           self.embeddings[idx, 1]),
                            xycoords='data')

        plt.gca().set_aspect('equal', 'datalim')
        plt.title(format('UMAP projection of contigs - 0: %d clusters' % (len(label_set))), fontsize=24)
        plt.savefig(self.path + '/UMAP_projection_with_clusters.png')


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
        
        for (idx, contig) in self.large_contigs.iterrows():
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
