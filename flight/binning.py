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
import os
import datetime
from operator import itemgetter

# Function imports
import numpy as np
from numba import njit, config, set_num_threads
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
        self.min_bin_size = min_bin_size
        self.threads = threads
        set_num_threads(threads)
        # Open up assembly
        self.assembly = {} 
        for (tid, rec) in enumerate(SeqIO.parse(assembly, "fasta")):
            self.assembly[rec.id] = tid

        # initialize bin dictionary Label: Vec<Contig>
        self.bins = {}

        ## Set up clusterer and UMAP
        self.path = output_prefix

        ## These tables should have the same ordering as each other if they came from rosella.
        ## I.e. all the rows match the same contig
        self.coverage_table = pd.read_csv(count_path, sep='\t')
        self.tnfs = pd.read_csv(kmer_frequencies, sep='\t')
        # self.tnfs = self.tnfs[self.tnfs["contigLen"] >= min_contig_size]

        self.large_contigs = self.coverage_table[(self.coverage_table["contigLen"] >= min_contig_size)]
        self.small_contigs = self.coverage_table[(self.coverage_table["contigLen"] < min_contig_size)]

        self.tnfs = self.tnfs[self.tnfs['contigName'].isin(self.large_contigs['contigName'])]

        # self.snv_rates, self.sv_rates = read_variant_rates(variant_rates, self.large_contigs['contigName'], min_contig_size)

        ## Check the ordering of the contig names for sanity check
        if list(self.large_contigs['contigName']) != list(self.tnfs['contigName']):
            sys.exit("Contig ordering incorrect for kmer table or coverage table")

        # if self.depths.shape[1] > 2:
        self.n_samples = len(self.large_contigs.columns[3::2])
        # self.small_depths = self.small_depths[self.small_depths.columns[::2]]

        ## Scale the data but first check if we have an appropriate amount of samples
        if scaler.lower() == "clr" and self.n_samples < 3:
            scaler = "minmax"


        # clr transformations
        self.tnfs = skbio.stats.composition.clr(self.tnfs[[name for name in self.tnfs.columns if utils.special_match(name)]]
                                                .iloc[:, 1:].astype(np.float64) + 1)

        if self.n_samples < 3:
            # self.depths = skbio.stats.composition.clr(self.depths.T.astype(np.float64) + 1).T
            self.depths = np.nan_to_num(np.concatenate((self.large_contigs.iloc[:, 1].values[:, None], self.large_contigs.iloc[:, 3:], self.tnfs), axis=1))

        else:
            # self.depths = skbio.stats.composition.clr(self.depths.T.astype(np.float64) + 1).T
            # self.depths = np.nan_to_num(np.concatenate((self.large_contigs.iloc[:, 1].values[:, None], self.large_contigs.iloc[:, 3:], self.tnfs), axis=1))
            self.depths = self.large_contigs.iloc[:, 3:]
            # self.depths = RobustScaler().fit_transform(np.nan_to_num(self.depths, nan=0.0, posinf=0.0, neginf=0.0))

       
        if self.n_samples >= 3:
            # Three UMAP reducers for each input type
            self.tnf_reducer = umap.UMAP(
                metric='cosine',
                # metric_kwds={"n_samples": self.n_samples},
                n_neighbors=int(n_neighbors),
                n_components=n_components,
                min_dist=0,
                random_state=random_state,
                n_epochs=500,
                spread=0.5,
                a=a,
                b=b,
            )
            # 
            self.depth_reducer = umap.UMAP(
                metric=metrics.metabat_distance,
                metric_kwds={"n_samples": self.n_samples},
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                random_state=random_state,
                n_epochs=500,
                spread=0.5,
                a=a,
                b=b,
            )

            self.correlation_reducer = umap.UMAP(
                            metric='cosine',
                            # metric_kwds={"n_samples": self.n_samples},
                            n_neighbors=n_neighbors,
                            n_components=n_components,
                            min_dist=0,
                            random_state=random_state,
                            n_epochs=500,
                            spread=0.5,
                            a=a,
                            b=b,
                        )

        elif self.n_samples < 3:
            
            self.depth_reducer = umap.UMAP(
                metric=metrics.aggregate_tnf,
                metric_kwds={"n_samples": self.n_samples},
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                random_state=random_state,
                n_epochs=500,
                spread=0.5,
                a=a,
                b=b,
            )


    def fit_transform(self):
        ## Calculate the UMAP embeddings

        if self.n_samples >= 3:
            logging.info("Running UMAP - %s" % self.depth_reducer)
            try:
                depth_mapping = self.depth_reducer.fit(self.depths)
            except ValueError: # Sparse or low coverage contigs can cause high n_neighbour values to kark it
                self.depth_reducer.n_neighbors = 30
                depth_mapping = self.depth_reducer.fit(self.depths)               
            logging.info("Running UMAP - %s" % self.tnf_reducer)
            tnf_mapping = self.tnf_reducer.fit(self.tnfs)
            if self.n_samples >= 3:
                logging.info("Running UMAP - %s" % self.correlation_reducer)
                correlation_mapping = self.correlation_reducer.fit(skbio.stats.composition.clr(self.depths.iloc[:, 0::2].T.astype(np.float64) + 1).T)
                intersection_mapper = (depth_mapping * correlation_mapping) * tnf_mapping
            else:
                intersection_mapper = depth_mapping * tnf_mapping

            ## Embeddings to cluster against
            self.embeddings = intersection_mapper.embedding_

        else:
            logging.info("Running UMAP - %s" % self.depth_reducer)
            try:
                depth_mapping = self.depth_reducer.fit(self.depths)
            except ValueError: # Sparse or low coverage contigs can cause high n_neighbour values to kark it
                self.depth_reducer.n_neighbors = 30
                depth_mapping = self.depth_reducer.fit(self.depths)

            self.embeddings = depth_mapping.embedding_

    def cluster(self):
        ## Cluster on the UMAP embeddings and return soft clusters
        logging.info("Clustering contigs...")
        tuned = utils.hyperparameter_selection(self.embeddings, self.threads, allow_single_cluster=False)
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
        tuned = utils.hyperparameter_selection(self.unbinned_embeddings, self.threads)
        best = utils.best_validity(tuned)
        if best is not None:
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
        else:
            self.unbinned_clusterer = None

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

        self.unbinned_indices = []
        self.unbinned_embeddings = []

        set_labels = set(self.clusterer.labels_)
        max_bin_id = max(set_labels)

        if len(set_labels) > 5:
            for (idx, label) in enumerate(self.clusterer.labels_):
                if label != -1:
                    # if self.cluster_validity[label] > 0.0:
                    try:
                        self.bins[label.item() + 1].append(
                            self.assembly[self.large_contigs.iloc[idx, 0]]) # inputs values as tid
                    except KeyError:
                        self.bins[label.item() + 1] = [self.assembly[self.large_contigs.iloc[idx, 0]]]

                    # elif self.large_contigs.iloc[idx, 1] >= self.min_bin_size:
                        # max_bin_id += 1
                        # try:
                            # self.bins[max_bin_id.item()].append(
                                # self.large_contigs.iloc[idx, 0:2].name.item()) # inputs values as tid
                        # except KeyError:
                            # self.bins[max_bin_id.item()] = [self.large_contigs.iloc[idx, 0:2].name.item()]
                    # else:
                        # try:
                            # redo_bins[label.item()]["embeddings"].append(self.embeddings[idx, :]) # inputs values as tid
                            # redo_bins[label.item()]["indices"].append(idx)  # inputs values as tid
                        # except KeyError:
                            # redo_bins[label.item()] = {}
                            # redo_bins[label.item()]["embeddings"] = [self.embeddings[idx, :]]
                            # redo_bins[label.item()]["indices"] = [idx]
                                             
                elif self.large_contigs.iloc[idx, 1] >= self.min_bin_size:
                    max_bin_id += 1
                    try:
                        self.bins[max_bin_id.item()].append(
                            self.assembly[self.large_contigs.iloc[idx, 0]]) # inputs values as tid
                    except KeyError:
                        self.bins[max_bin_id.item()] = [self.assembly[self.large_contigs.iloc[idx, 0]]]
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
                                self.assembly[self.large_contigs.iloc[idx, 0]])  # inputs values as tid
                        except KeyError:
                            self.bins[label.item() + max_bin_id] = [self.assembly[self.large_contigs.iloc[idx, 0]]]
                    elif self.large_contigs.iloc[idx, 1] >= self.min_bin_size:
                        inner_bin_id += 1

                        try:
                            self.bins[inner_bin_id.item() + max_bin_id].append(
                                self.assembly[self.large_contigs.iloc[idx, 0]]) # inputs values as tid
                        except KeyError:
                            self.bins[inner_bin_id.item() + max_bin_id] = [self.assembly[self.large_contigs.iloc[idx, 0]]]
                    else:
                        self.unbinned_indices.append(idx)
                        self.unbinned_embeddings.append(self.embeddings[idx, :])

        removed_bins = redo_bins.keys()
        if len(removed_bins) > 0:
            self.clusterer.labels_[:] = [label - sum(i < label for i in removed_bins) if label not in removed_bins else label for label in self.clusterer.labels_]
            
        # break up very large bins. Not sure how to threshold this
        for (bin, values) in redo_bins.items():
            new_labels = utils.break_overclustered(np.array(values["embeddings"]), self.threads)
            inner_bin_id = max(set(new_labels))
            try:
                max_bin_id = max([label.item() for label in set(self.clusterer.labels_) if label not in removed_bins]) + 1
            except ValueError:
                max_bin_id = 1
            for (idx, label) in zip(values["indices"], new_labels):
                if label != -1:
                    # Update labels
                    self.clusterer.labels_[idx] = label.item() + max_bin_id
                    self.soft_clusters_capped[idx] = label.item() + max_bin_id
                    try:
                        self.bins[label.item() + max_bin_id].append(
                            self.assembly[self.large_contigs.iloc[idx, 0]])  # inputs values as tid
                    except KeyError:
                        self.bins[label.item() + max_bin_id] = [self.assembly[self.large_contigs.iloc[idx, 0]]]
                elif self.large_contigs.iloc[idx, 1] >= self.min_bin_size:
                    inner_bin_id += 1

                    self.clusterer.labels_[idx] = inner_bin_id + max_bin_id
                    self.soft_clusters_capped[idx] = inner_bin_id + max_bin_id
                    try:
                        self.bins[inner_bin_id.item() + max_bin_id].append(
                            self.assembly[self.large_contigs.iloc[idx, 0]]) # inputs values as tid
                    except KeyError:
                        self.bins[inner_bin_id.item() + max_bin_id] = [self.assembly[self.large_contigs.iloc[idx, 0]]]
                else:
                    # unbinned contigs
                    try:
                        self.bins[label.item() + 1].append(
                            self.assembly[self.large_contigs.iloc[idx, 0]])  # inputs values as tid
                    except KeyError:
                        self.bins[label.item() + 1] = [self.assembly[self.large_contigs.iloc[idx, 0]]]


        self.unbinned_embeddings = np.array(self.unbinned_embeddings)

    def bin_unbinned_contigs(self):
        if self.unbinned_clusterer is not None:
            logging.info("Binning unbinned contigs...")
            max_bin_id = max(self.bins.keys()) + 1


            for (unbinned_idx, label) in enumerate(self.unbinned_clusterer.labels_):
                idx = self.unbinned_indices[unbinned_idx]
                self.clusterer.labels_[idx] = label.item() + max_bin_id
                if label != -1:
                    try:
                        self.bins[label.item() + max_bin_id].append(
                            self.assembly[self.large_contigs.iloc[idx, 0]]) # inputs values as tid
                    except KeyError:
                        self.bins[label.item() + max_bin_id] = [self.assembly[self.large_contigs.iloc[idx, 0]]]
                else:
                    ## bin out the unbinned contigs again as label 0. Rosella will try to rescue them if there
                    ## are enough samples
                    try:
                        self.bins[label.item() + 1].append(
                            self.assembly[self.large_contigs.iloc[idx, 0]])  # inputs values as tid
                    except KeyError:
                        self.bins[label.item() + 1] = [self.assembly[self.large_contigs.iloc[idx, 0]]]

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
                                self.bins[result[0].item()].append(self.assembly[self.large_contigs.iloc[result[1], 0]])
                                # self.bins[bin].remove(idx)
                            except KeyError:
                                self.bins[result[0].item()] = [self.assembly[self.large_contigs.iloc[result[1], 0]]]

        pool.close()
        pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

    def write_bins(self, min_bin_size=200000):
        logging.info("Writing bin JSON...")

        with open(self.path + '/rosella_bins.json', 'w') as fp:
            json.dump(self.bins, fp)

        # self.small_contigs.to_csv(self.path + '/rosella_small_contigs.tsv', sep='\t')
        # self.large_contigs.to_csv(self.path + '/rosella_large_contigs.tsv', sep='\t')
