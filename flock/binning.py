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
from numba import njit
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
import umap
from itertools import product

# self imports
import flock.metrics as metrics

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
            variant_rates,
            output_prefix,
            assembly,
            scaler="clr",
            n_neighbors=20,
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
            metric = 'aggregate_variant_tnf',
            threads=8,
    ):
        self.threads = threads
        # Open up assembly
        self.assembly = SeqIO.to_dict(SeqIO.parse(assembly, "fasta"))

        ## Set up clusterer and UMAP
        self.path = output_prefix

        ## These tables should have the same ordering as each other if they came from rosella.
        ## I.e. all the rows match the same contig
        self.coverage_table = pd.read_csv(count_path, sep='\t')
        self.tnfs = pd.read_csv(kmer_frequencies, sep='\t')
        self.tnfs = self.tnfs[self.tnfs["contigLen"] >= min_contig_size]

        self.large_contigs = self.coverage_table[self.coverage_table["contigLen"] >= min_contig_size]
        self.small_contigs = self.coverage_table[self.coverage_table["contigLen"] < min_contig_size]

        self.snv_rates, self.sv_rates = read_variant_rates(variant_rates, self.large_contigs['contigName'], min_contig_size)

        ## Check the ordering of the contig names for sanity check
        if list(self.large_contigs['contigName']) != list(self.tnfs['contigName']):
            sys.exit("Contig ordering incorrect for kmer table or coverage table")

        # self.tnfs = self.tnfs.iloc[:, 2:] + 1
        # self.tnfs = self.tnfs.div(self.tnfs.sum(axis=1), axis=0)  # convert counts to frequencies along rows
        # self.tnfs = skbio.stats.composition.clr(self.tnfs + 1)


        # If there are enough contigs of that size
        if self.large_contigs.shape[0] > 100:
            self.depths = self.large_contigs.iloc[:,3:]
            # self.small_depths = self.small_contigs.iloc[:,3:]
        else: # Otherwise we'll just use a smaller value
            self.large_contigs = self.coverage_table[self.coverage_table["contigLen"] >= 1000]
            self.small_contigs = self.coverage_table[self.coverage_table["contigLen"] < 1000]
            self.depths = self.large_contigs.iloc[:,3:]
            # self.small_depths = self.small_contigs.iloc[:,3:]

        # if self.depths.shape[1] > 2:
        self.depths = self.depths[self.depths.columns[::2]]
        self.n_samples = self.depths.shape[1]
        # self.small_depths = self.small_depths[self.small_depths.columns[::2]]

        ## Scale the data but first check if we have an appropriate amount of samples
        if scaler.lower() == "clr" and self.n_samples < 3:
            scaler = "minmax"

        # if scaler.lower() == "minmax":
        #     self.depths = MinMaxScaler().fit_transform(self.depths)
        #     # self.small_depths = MinMaxScaler().fit_transform(self.small_depths)
        # elif scaler.lower() == "clr":
        #     # Need to merge small and large together for CLR transform to work properly
        #     # large_count = self.depths.shape[0]
        #     # concatenated = np.concatenate((self.depths, self.small_depths))
        #     # concatenated = concatenated.T + 1
        #     # concatenated = skbio.stats.composition.clr(concatenated).T
        #     # self.depths = concatenated[:large_count, ]
        #     # self.small_depths = concatenated[large_count:, ]
        #     self.depths = skbio.stats.composition.clr(self.depths.T + 1).T
        # elif scaler.lower() == "none":
        #     pass

        # Normalize all table values
        # self.tnfs = (self.tnfs.iloc[:, 2:] / np.sqrt(np.square(self.tnfs.iloc[:, 2:] + 1).sum(axis=1)))

        # clr transformations
        self.tnfs = skbio.stats.composition.clr(self.tnfs[[name for name in self.tnfs.columns if 'N' not in name]].iloc[:, 1:] + 1)
        self.depths = skbio.stats.composition.clr(self.depths.T + 1).T
        self.snv_rates = skbio.stats.composition.clr(self.snv_rates + 1)
        self.sv_rates = skbio.stats.composition.clr(self.sv_rates + 1)


        # if self.depths.shape[1] > 1:
            # pass
        # else:
        self.depths = np.concatenate((self.depths, self.snv_rates, self.sv_rates, self.tnfs), axis=1) # Add extra dimension so concatenation works
            
        if n_neighbors >= int(self.depths.shape[0] * 0.5):
            n_neighbors = max(int(self.depths.shape[0] * 0.5), 2)

        if n_components > self.depths.shape[1]:
            n_components = self.depths.shape[1]

        if metric in ['aggregate', 'aggregate_variant_tnf', 'aggregate_tnf', 'rho', 'phi', 'phi_dist']:
            self.reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                random_state=random_state,
                spread=1,
                metric=getattr(metrics, metric),
                metric_kwds={'n_samples': self.n_samples}
            )
        else:
            self.reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                random_state=random_state,
                spread=1,
                metric=metric
            )


        if min_cluster_size > self.depths.shape[0] * 0.1:
            min_cluster_size = max(int(self.depths.shape[0] * 0.1), 2)
            min_samples = max(int(min_cluster_size * 0.1), 2)

        if precomputed:
            metric = "precomputed"
            prediction_data = False

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            # min_samples=min_samples,
            prediction_data=prediction_data,
            cluster_selection_method=cluster_selection_method,
            metric=hdbscan_metric,
        )

    def fit_transform(self):
        ## Calculate the UMAP embeddings
        logging.info("Running UMAP - %s" % self.reducer)
        self.embeddings = self.reducer.fit_transform(self.depths)
        # self.small_embeddings = self.reducer.transform(self.small_depths)

    def cluster(self):
        ## Cluster on the UMAP embeddings and return soft clusters
        try:
            logging.info("Running HDBSCAN - %s" % self.clusterer)
            self.clusterer.fit(self.embeddings)
            self.soft_clusters = hdbscan.all_points_membership_vectors(
                self.clusterer)
            self.soft_clusters_capped = np.array([np.argmax(x) for x in self.soft_clusters])
            # self.small_labels, self.small_strengths = hdbscan.approximate_predict(self.clusterer, self.small_embeddings)
        except:
            ## Likely integer overflow in HDBSCAN
            ## Try reduce min samples
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(int(self.depths.shape[0] * 0.01), 2),
                min_samples=max(int(self.depths.shape[0] * 0.005), 2),
                prediction_data=True,
                cluster_selection_method="eom",
            )
            logging.info("Retrying HDBSCAN - %s" % self.clusterer)
            self.clusterer.fit(self.embeddings)
            self.soft_clusters = hdbscan.all_points_membership_vectors(
                self.clusterer)
            self.soft_clusters_capped = np.array([np.argmax(x) for x in self.soft_clusters])
            # self.small_labels, self.small_strengths = hdbscan.approximate_predict(self.clusterer, self.small_embeddings)


    def cluster_distances(self):
        ## Cluster on the UMAP embeddings and return soft clusters
        try:
            logging.info("Running HDBSCAN - %s" % self.clusterer)
            self.clusterer.fit(self.depths)
        except:
            ## Likely integer overflow in HDBSCAN
            ## Try reduce min samples
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(int(self.depths.shape[0] * 0.01), 2),
                min_samples=max(int(self.depths.shape[0] * 0.005), 2),
                prediction_data=True,
                cluster_selection_method="precomputed",
            )
            logging.info("Retrying HDBSCAN - %s" % self.clusterer)

            self.clusterer.fit(self.depths)

    def plot(self):
        logging.info("Generating UMAP plot with labels")

        label_set = set(self.clusterer.labels_)
        color_palette = sns.color_palette('Paired', len(label_set))
        cluster_colors = [
            color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in self.soft_clusters_capped
        ]

        # small_cluster_colors = [
        #     color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in self.small_labels
        # ]

        cluster_member_colors = [
            sns.desaturate(x, p) for x, p in zip(cluster_colors, self.clusterer.probabilities_)
        ]
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ## Plot large contig membership
        ax.scatter(self.embeddings[:, 0],
                   self.embeddings[:, 1],
                   s=7,
                   linewidth=0,
                   c=cluster_member_colors,
                   alpha=0.7)

        # ## Plot small contig membership
        # ax.scatter(self.small_embeddings[:, 0],
        #            self.small_embeddings[:, 1],
        #            s=5,
        #            linewidth=0,
        #            c=small_cluster_colors,
        #            alpha=0.7)
        # ax.add_artist(legend)
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
            return self.soft_clusters.astype('int8')
        except AttributeError:
            return self.clusterer.labels_.astype('int8')

    def bin_contigs(self, assembly_file, min_bin_size=200000):
        logging.info("Binning contigs...")

        # initialize bin dictionary Label: Vec<Contig>
        self.bins = {}
        for (idx, label) in enumerate(self.clusterer.labels_):
            if label != -1:
                try:
                    self.bins[label.item()].append(self.large_contigs.iloc[idx, 0:2].name.item()) # inputs values as tid
                except KeyError:
                    self.bins[label.item()] = [self.large_contigs.iloc[idx, 0:2].name.item()]
            # elif len(self.assembly[self.large_contigs.iloc[idx, 0]].seq) >= min_bin_size:
            #     try:
            #         self.bins[label].append(idx)
            #     except KeyError:
            #         self.bins[label] = [idx]
            else:
                soft_label = self.soft_clusters_capped[idx]
                try:
                    self.bins[soft_label.item()].append(self.large_contigs.iloc[idx, 0:2].name.item())
                except KeyError:
                    self.bins[soft_label.item()] = [self.large_contigs.iloc[idx, 0:2].name.item()]

        # ## Bin out small contigs
        # for (idx, label) in enumerate(self.small_labels):
        #     if label != -1:
        #         try:
        #             self.bins[label].append(idx)
        #         except KeyError:
        #             self.bins[label] = [idx]

    def merge_bins(self, min_bin_size=200000):
        pool = mp.Pool(self.threads)

        if self.n_samples < 3:
            logging.info("Merging bins...")
            for bin in list(self.bins):
                if bin != -1:
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
        # else:
            # for bin in list(self.bins):
            #     if bin != -1:
            #         contigs = self.bins[bin]
            #         lengths = [pool.apply_async(len, args=(self.assembly[self.large_contigs.iloc[idx, 0]].seq)) for idx in contigs]
            #         bin_length = 0
            #         [bin_length := bin_length + r.get() for r in lengths]
            #         # bin_length = sum([len(self.assembly[self.large_contigs.iloc[idx, 0]].seq) for idx in contigs])
            #         if bin_length < min_bin_size:
            #             for idx in contigs:
            #                 results = [self.spawn_merge_high_n(idx, self.depths, other_bin, other_ids, self.n_samples, pool)
            #                                               for other_bin, other_ids in self.bins.items() if
            #                                               other_bin != bin and other_bin != -1]
            #
            #                 # results = [r.get() for r in results]
            #                 max_concordance = max(results, key=itemgetter(0))
            #                 if max_concordance[0] > 0.8:
            #                     try:
            #                         self.bins[max_concordance[1]].append(idx)
            #                         # self.bins[bin].remove(idx)
            #                     except KeyError:
            #                         self.bins[max_concordance[1]] = [idx]
            # Let rosella handle this

        pool.close()
        pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

    def spawn_merge_high_n(self, idx, depths, other_bin, other_ids, n_samples, pool):
        current_depths = depths[idx,]
        result = [pool.apply_async(metrics.concordance, args=(current_depths, depths[other_id,], n_samples))
                                     for other_id in other_ids]
        result = [r.get() for r in result]
        average_rho = sum(result) / n_samples

        return average_rho, other_bin

    def spawn_merge_small_contigs(self, idx, small_depths, depths, other_bin, other_ids, n_samples):
        result = [metrics.concordance(small_depths, depths[other_id,], n_samples)
                                     for other_id in other_ids]

        average_rho = sum(result) / n_samples

        return average_rho, other_bin

    def rescue_small_contigs(self):
        logging.info("Rescuing contigs...")
        #
        # pool = mp.Pool(self.threads)
        # for (contig_id, small_depth) in enumerate(self.small_depths):
        #     results = [self.pool.apply_async(self.spawn_merge_small_contigs, args=(contig_id, small_depth, self.depths, other_bin, other_ids, self.n_samples)) for
        #                                   other_bin, other_ids
        #                                   in self.bins.items() if
        #                                   other_bin != bin and other_bin != -1]
        #     results = [r.get() for r in results]
        #     max_concordance = max(results, key=itemgetter(0))
        #     if max_concordance[0] > 0.8:
        #         try:
        #             self.bins[max_concordance[1]].append(contig_id)
        #             # self.bins[bin].remove(idx)
        #         except KeyError:
        #             self.bins[max_concordance[1]] = [contig_id]
        #
        # pool.close()
        # pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

    def write_bins(self, min_bin_size=200000):
        logging.info("Writing bin JSON...")

        with open(self.path + '/rosella_bins.json', 'w') as fp:
            json.dump(self.bins, fp)

        # self.small_contigs.to_csv(self.path + '/rosella_small_contigs.tsv', sep='\t')
        # self.large_contigs.to_csv(self.path + '/rosella_large_contigs.tsv', sep='\t')
