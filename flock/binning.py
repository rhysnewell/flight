#!/usr/bin/env python
###############################################################################
# binning.py - A fast binning algorithm spinning off of the methodology of
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
__version__ = "0.0.1"
__maintainer__ = "Rhys Newell"
__email__ = "rhys.newell near hdr.qut.edu.au"
__status__ = "Development"

###############################################################################
# System imports
import sys
import argparse
import logging
import os
import datetime
from operator import itemgetter

# Function imports
import numpy as np
from numba import njit, set_num_threads
import multiprocessing as mp
import pandas as pd
import hdbscan
import seaborn as sns
import json
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from Bio import SeqIO
import skbio.stats.composition
from sklearn.preprocessing import RobustScaler
import umap
from itertools import product
import pynndescent

# self imports
import flock.metrics as metrics
import flock.utils as utils

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


def write_contig(contig, assembly, f):
    seq = assembly[contig]
    fasta = ">" + seq.id + '\n'
    fasta += str(seq.seq) + '\n'
    f.write(fasta)

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


def read_variant_rates(variant_rates, contig_names, min_contig_size=1000):
    rates = pd.read_csv(variant_rates, sep='\t')
    rates = rates[rates["contigLen"] >= min_contig_size]
    if list(rates['contigName']) != list(contig_names):
        sys.exit("Contig ordering incorrect for variant rate table")
    snv_rates = rates.iloc[:, 2::2]
    sv_rates = rates.iloc[:, 3::2]
    return snv_rates, sv_rates


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
            kmer_frequencies,
            # variant_rates,
            output_prefix,
            assembly,
            scaler="clr",
            n_neighbors=100,
            min_dist=0.0,
            n_components=2,
            random_state=42,
            min_cluster_size=100,
            min_contig_size=2500,
            min_samples=50,
            prediction_data=True,
            cluster_selection_method="eom",
            precomputed=False,
            hdbscan_metric="euclidean",
            metric = 'aggregate_variant_tnf',
            threads=8,
    ):
        self.threads = threads
        set_num_threads(threads)
        # Open up assembly
        self.assembly = SeqIO.to_dict(SeqIO.parse(assembly, "fasta"))

        # initialize bin dictionary Label: Vec<Contig>
        self.bins = {}

        ## Set up clusterer and UMAP
        self.path = output_prefix

        ## These tables should have the same ordering as each other if they came from rosella.
        ## I.e. all the rows match the same contig
        self.coverage_table = pd.read_csv(count_path, sep='\t')
        self.tnfs = pd.read_csv(kmer_frequencies, sep='\t')
        self.tnfs = self.tnfs[self.tnfs["contigLen"] >= min_contig_size]

        self.large_contigs = self.coverage_table[self.coverage_table["contigLen"] >= min_contig_size]
        self.small_contigs = self.coverage_table[self.coverage_table["contigLen"] < min_contig_size]

        # self.snv_rates, self.sv_rates = read_variant_rates(variant_rates, self.large_contigs['contigName'], min_contig_size)

        ## Check the ordering of the contig names for sanity check
        if list(self.large_contigs['contigName']) != list(self.tnfs['contigName']):
            sys.exit("Contig ordering incorrect for kmer table or coverage table")


        # If there are enough contigs of that size
        if self.large_contigs.shape[0] > 100:
            self.depths = self.large_contigs.iloc[:,3::2]
            self.variance = self.large_contigs.iloc[:,4::2]
            # self.small_depths = self.small_contigs.iloc[:,3:]
        else: # Otherwise we'll just use a smaller value
            self.large_contigs = self.coverage_table[self.coverage_table["contigLen"] >= 1000]
            self.small_contigs = self.coverage_table[self.coverage_table["contigLen"] < 1000]
            self.depths = self.large_contigs.iloc[:,3::2]
            self.variance = self.large_contigs.iloc[:,4::2]
            # self.small_depths = self.small_contigs.iloc[:,3:]

        # if self.depths.shape[1] > 2:
        self.n_samples = self.depths.shape[1]
        # self.small_depths = self.small_depths[self.small_depths.columns[::2]]

        ## Scale the data but first check if we have an appropriate amount of samples
        if scaler.lower() == "clr" and self.n_samples < 3:
            scaler = "minmax"


        # clr transformations
        self.tnfs = skbio.stats.composition.clr(self.tnfs[[name for name in self.tnfs.columns if utils.special_match(name)]]
                                                .iloc[:, 1:].astype(np.float64) + 1)

        if self.n_samples < 3:
            # self.depths = self.depths.values.reshape(1, -1)
            # self.variances = self.variances.values.reshape(1, -1)
            # self.snv_rates = self.snv_rates.values.reshape(1, -1)
            # self.depths = skbio.stats.composition.clr(self.depths.T.astype(np.float64) + 1).T

            self.depths = np.nan_to_num(RobustScaler().fit_transform(self.depths), nan=0.0, posinf=0.0, neginf=0.0)
            self.variance = RobustScaler().fit_transform(np.nan_to_num(self.variance, nan=0.0, posinf=0.0, neginf=0.0))
            self.depths = np.nan_to_num(np.concatenate((self.large_contigs.iloc[:, 1].values[:, None], self.depths, self.tnfs), axis=1))

            # self.snv_rates = RobustScaler().fit_transform(np.nan_to_num(self.snv_rates, nan=0.0, posinf=0.0, neginf=0.0))
            # self.variance = RobustScaler().fit_transform(np.nan_to_num(self.variance, nan=0.0, posinf=0.0, neginf=0.0))
        else:
            self.depths = skbio.stats.composition.clr(self.depths.T.astype(np.float64) + 1).T
            # self.snv_rates = RobustScaler().fit_transform(np.nan_to_num(self.snv_rates, nan=0.0, posinf=0.0, neginf=0.0))
            self.variance = RobustScaler().fit_transform(np.nan_to_num(self.variance, nan=0.0, posinf=0.0, neginf=0.0))


        
        if self.n_samples >= 3:
            # Three UMAP reducers for each input type
            self.tnf_reducer = umap.UMAP(
                metric=metrics.rho,
                metric_kwds={"n_samples": self.n_samples},
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0,
                random_state=random_state,
                spread=1,
            )
            self.depth_reducer = umap.UMAP(
                metric=metrics.rho,
                metric_kwds={"n_samples": self.n_samples},
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                random_state=random_state,
                spread=1,
            )
            
            self.variance_reducer = umap.UMAP(
                metric='correlation',
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                random_state=random_state,
                spread=1,
            )
            
        else:
            self.depth_reducer = umap.UMAP(
                metric=metrics.aggregate_tnf,
                metric_kwds={"n_samples": self.n_samples},
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                random_state=random_state,
                spread=1,
            )


        self.snv_reducer = umap.UMAP(
            metric='euclidean',
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            random_state=random_state,
            spread=1,
        )


    def fit_transform(self):
        ## Calculate the UMAP embeddings
        # logging.info("Running UMAP - %s" % self.snv_reducer)
        # snv_mapping = self.snv_reducer.fit(self.snv_rates)
        
        logging.info("Running UMAP - %s" % self.depth_reducer)
        depth_mapping = self.depth_reducer.fit(self.depths)
        
        if self.n_samples >=3:
            logging.info("Running UMAP - %s" % self.tnf_reducer)
            tnf_mapping = self.tnf_reducer.fit(self.tnfs)
            logging.info("Running UMAP - %s" % self.variance_reducer)
            variance_mapping = self.variance_reducer.fit(self.variance)
            ## Contrast all reducers
            contrast_mapper = (tnf_mapping - variance_mapping) + depth_mapping
        else:
            contrast_mapper = depth_mapping #- (depth_mapping - variance_mapping)
        self.embeddings = contrast_mapper.embedding_

    def cluster(self):
        ## Cluster on the UMAP embeddings and return soft clusters
        logging.info("Running HDBSCAN")
        tuned = utils.hyperparameter_selection(self.embeddings, self.threads)
        best = utils.best_validity(tuned)
        self.clusterer = hdbscan.HDBSCAN(
            algorithm='best',
            alpha=1.0,
            approx_min_span_tree=True,
            gen_min_span_tree=True,
            leaf_size=40,
            cluster_selection_method='eom',
            metric='euclidean',
            min_cluster_size=int(best['min_cluster_size']),
            min_samples=int(best['min_samples']),
            allow_single_cluster=False,
            core_dist_n_jobs=self.threads,
            prediction_data=True
        )
        self.clusterer.fit(self.embeddings)

        self.validity, self.cluster_validity = hdbscan.validity.validity_index(self.embeddings.astype(np.float64), self.clusterer.labels_, per_cluster_scores=True)
        self.soft_clusters = hdbscan.all_points_membership_vectors(
            self.clusterer)
        self.soft_clusters_capped = np.array([np.argmax(x) for x in self.soft_clusters])

    def cluster_unbinned(self):
        ## Cluster on the unbinned contigs, attempt to create fine grained clusters that were missed

        logging.info("Running HDBSCAN")
        tuned = utils.hyperparameter_selection(self.unbinned_embeddings, self.threads)
        best = utils.best_validity(tuned)
        self.unbinned_clusterer = hdbscan.HDBSCAN(
            algorithm='best',
            alpha=1.0,
            approx_min_span_tree=True,
            gen_min_span_tree=True,
            leaf_size=40,
            cluster_selection_method='eom',
            metric='euclidean',
            min_cluster_size=int(best['min_cluster_size']),
            min_samples=int(best['min_samples']),
            allow_single_cluster=False,
            core_dist_n_jobs=self.threads,
            prediction_data=True
        )
        self.unbinned_clusterer.fit(self.unbinned_embeddings)

    def plot(self):
        logging.info("Generating UMAP plot with labels")

        # label_set = set(self.clusterer.labels_)
        # color_palette = sns.color_palette('Paired', len(label_set))
        # cluster_colors = [
            # color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in self.clusterer.labels_ if x != -1
        # ]
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
                   c=self.clusterer.labels_,
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
            return self.soft_clusters_capped.astype('int8')
        except AttributeError:
            return self.clusterer.labels_.astype('int8')

    def bin_contigs(self, assembly_file, min_bin_size=200000):
        logging.info("Binning contigs...")
        self.bins = {}
        redo_bins = {}

        self.unbinned_indices = []
        self.unbinned_embeddings = []

        if len(set(self.clusterer.labels_)) > 5:
            for (idx, label) in enumerate(self.clusterer.labels_):
                if label != -1:
                    if self.cluster_validity[label] >= 0.5:
                        try:
                            self.bins[label.item() + 1].append(
                                self.large_contigs.iloc[idx, 0:2].name.item()) # inputs values as tid
                        except KeyError:
                            self.bins[label.item() + 1] = [self.large_contigs.iloc[idx, 0:2].name.item()]
                    else:
                        try:
                            redo_bins[label.item() + 1]["size"] += self.large_contigs.iloc[idx, 1]
                            redo_bins[label.item() + 1]["embeddings"].append(self.embeddings[idx, :])
                            redo_bins[label.item() + 1]["indices"].append(idx)
                        except KeyError:
                            redo_bins[label.item() + 1] = {}
                            redo_bins[label.item() + 1]["size"] = self.large_contigs.iloc[idx, 1]
                            redo_bins[label.item() + 1]["embeddings"] = [self.embeddings[idx, :]]
                            redo_bins[label.item() + 1]["indices"] = [idx]
                        
                else:
                    self.unbinned_indices.append(idx)
                    self.unbinned_embeddings.append(self.embeddings[idx, :])
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
                try:
                    max_bin_id = max(self.bins.keys()) + 1
                except ValueError:
                    max_bin_id = 1
                for (idx, label) in zip(values["indices"], new_labels):
                    # Update labels
                    if label != -1:
                        self.clusterer.labels_[idx] = label + max_bin_id - 1
                        self.soft_clusters_capped[idx] = label + max_bin_id - 1
                        try:
                            self.bins[label.item() + max_bin_id].append(
                                self.large_contigs.iloc[idx, 0:2].name.item())  # inputs values as tid
                        except KeyError:
                            self.bins[label.item() + max_bin_id] = [self.large_contigs.iloc[idx, 0:2].name.item()]
                    else:
                        self.unbinned_indices.append(idx)
                        self.unbinned_embeddings.append(self.embeddings[idx, :])

        # break up very large bins. Not sure how to threshold this
        for (bin, values) in redo_bins.items():
            new_labels = utils.break_overclustered(np.array(values["embeddings"]), self.threads)
            try:
                max_bin_id = max(self.bins.keys()) + 1
            except ValueError:
                max_bin_id = 1
            for (idx, label) in zip(values["indices"], new_labels):
                if label != -1:
                    # Update labels
                    self.clusterer.labels_[idx] = label + max_bin_id - 1
                    self.soft_clusters_capped[idx] = label + max_bin_id - 1
                    try:
                        self.bins[label.item() + max_bin_id].append(
                            self.large_contigs.iloc[idx, 0:2].name.item())  # inputs values as tid
                    except KeyError:
                        self.bins[label.item() + max_bin_id] = [self.large_contigs.iloc[idx, 0:2].name.item()]
                else:
                    # unbinned contigs
                    try:
                        self.bins[label.item() + 1].append(
                            self.large_contigs.iloc[idx, 0:2].name.item())  # inputs values as tid
                    except KeyError:
                        self.bins[label.item() + 1] = [self.large_contigs.iloc[idx, 0:2].name.item()]


        self.unbinned_embeddings = np.array(self.unbinned_embeddings)

    def bin_unbinned_contigs(self):
        logging.info("Binning unbinned contigs...")
        max_bin_id = max(self.bins.keys()) + 1


        for (unbinned_idx, label) in enumerate(self.unbinned_clusterer.labels_):
            idx = self.unbinned_indices[unbinned_idx]
            self.clusterer.labels_[idx] = label.item() + max_bin_id
            if label != -1:
                try:
                    self.bins[label.item() + max_bin_id].append(
                        self.large_contigs.iloc[idx, 0:2].name.item()) # inputs values as tid
                except KeyError:
                    self.bins[label.item() + max_bin_id] = [self.large_contigs.iloc[idx, 0:2].name.item()]
            # elif self.n_samples < 3:
                # soft_label = self.soft_clusters_capped[idx]
                # try:
                    # self.bins[soft_label.item() + 1].append(self.large_contigs.iloc[idx, 0:2].name.item())
                # except KeyError:
                    # self.bins[soft_label.item() + 1] = [self.large_contigs.iloc[idx, 0:2].name.item()]
            else:
                ## bin out the unbinned contigs again as label 0. Rosella will try to rescue them if there
                ## are enough samples
                try:
                    self.bins[label.item() + 1].append(
                        self.large_contigs.iloc[idx, 0:2].name.item())  # inputs values as tid
                except KeyError:
                    self.bins[label.item() + 1] = [self.large_contigs.iloc[idx, 0:2].name.item()]

    def merge_bins(self, min_bin_size=200000):
        pool = mp.Pool(self.threads)

        if self.n_samples < 3:
            logging.info("Merging bins...")
            for bin in list(self.bins):
                if bin != 0:
                    contigs = self.bins[bin]
                    bin_length = sum([len(self.assembly[self.large_contigs.iloc[idx, 0]].seq) for idx in contigs])
                    if bin_length < min_bin_size:
                        results = [pool.apply_async(spawn_merge_low_n, args=(idx, self.soft_clusters[idx])) for idx in contigs]
                        for result in results:
                            result = result.get()
                            try:
                                self.bins[result[0].item()].append(self.large_contigs.iloc[result[1], 0:2].name)
                                # self.bins[bin].remove(idx)
                            except KeyError:
                                self.bins[result[0].item()] = [self.large_contigs.iloc[result[1], 0:2].name]

        pool.close()
        pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

    def write_bins(self, min_bin_size=200000):
        logging.info("Writing bin JSON...")

        with open(self.path + '/rosella_bins.json', 'w') as fp:
            json.dump(self.bins, fp)

        # self.small_contigs.to_csv(self.path + '/rosella_small_contigs.tsv', sep='\t')
        # self.large_contigs.to_csv(self.path + '/rosella_large_contigs.tsv', sep='\t')
