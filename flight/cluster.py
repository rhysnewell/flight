#!/usr/bin/env python
###############################################################################
# cluster.py - A program which handles the UMAP and HDBSCAN python components
#              of lorikeet
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
import shutil
import datetime

# Function imports
import numpy as np
import hdbscan
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import skbio.stats.composition
import umap
from numba import set_num_threads
import pynndescent

# self imports
import flight.utils as utils
import flight.metrics as metrics

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
################################ - Functions - ################################


def phelp():
    print("""
Usage:
cluster.py [SUBCOMMAND] ..

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


class Cluster():
    def __init__(
        self,
        count_path,
        output_prefix,
        scaler="clr",
        n_neighbors=100,
        min_dist=0.1,
        n_components=2,
        random_state=42,
        min_cluster_size=100,
        min_samples=50,
        prediction_data=True,
        cluster_selection_method="eom",
        precomputed=False,
        metric='euclidean',
        hdbscan_metric="euclidean",
        threads=8
    ):
        set_num_threads(threads)
        self.threads = threads
        ## Set up clusterer and UMAP
        self.path = output_prefix
        self.depths = np.load(count_path)

        ## Scale the data
        if scaler.lower() == "minmax":
            self.depths = MinMaxScaler().fit_transform(self.depths)
        elif scaler.lower() == "clr":
            self.depths = skbio.stats.composition.clr((self.depths + 1).T).T
        elif scaler.lower() == "none":
            pass

        self.n_samples = self.depths.shape[1]


        if n_neighbors >= int(self.depths.shape[0] * 0.5):
            n_neighbors = max(int(self.depths.shape[0] * 0.5), 2)

        if n_components > self.depths.shape[1]:
            n_components = self.depths.shape[1]

        if metric in ['rho', 'phi', 'phi_dist']:
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
            self.metric = "precomputed"
            prediction_data = False
        else:
            self.metric = "euclidean"


    def fit_transform(self):
        ## Calculate the UMAP embeddings
        self.embeddings = self.reducer.fit_transform(self.depths)

    def cluster(self):
        ## Cluster on the UMAP embeddings and return soft clusters
        tuned = utils.hyperparameter_selection(self.embeddings, self.threads, metric=self.metric)
        best = utils.best_validity(tuned)
        self.clusterer = hdbscan.HDBSCAN(
            algorithm='best',
            alpha=1.0,
            approx_min_span_tree=True,
            gen_min_span_tree=True,
            leaf_size=40,
            cluster_selection_method='eom',
            metric=self.metric,
            min_cluster_size=int(best['min_cluster_size']),
            min_samples=int(best['min_samples']),
            allow_single_cluster=False,
            core_dist_n_jobs=self.threads,
            prediction_data=True
        )
        logging.info("Running HDBSCAN - %s" % self.clusterer)
        self.clusterer.fit(self.embeddings)
        try:
            self.validity, self.cluster_validity = hdbscan.validity.validity_index(self.embeddings.astype(np.float64),
                                                                                   self.clusterer.labels_,
                                                                                   per_cluster_scores=True)
        except ValueError:
            self.validity = None
            self.cluster_validity = [0.5 for i in range(len(set(self.clusterer.labels_)))]
        self.soft_clusters = hdbscan.all_points_membership_vectors(
            self.clusterer)
        self.soft_clusters_capped = np.array([np.argmax(x) for x in self.soft_clusters])

    def cluster_separation(self):
        dist_mat = utils.cluster_distances(self.embeddings, self.clusterer, self.threads)
        return dist_mat


    def cluster_distances(self):
        ## Cluster on the UMAP embeddings and return soft clusters
        
        tuned = utils.hyperparameter_selection(self.depths, self.threads, metric=self.metric)
        best = utils.best_validity(tuned)
        self.clusterer = hdbscan.HDBSCAN(
            algorithm='best',
            alpha=1.0,
            approx_min_span_tree=True,
            gen_min_span_tree=True,
            leaf_size=40,
            cluster_selection_method='eom',
            metric=self.metric,
            min_cluster_size=int(best['min_cluster_size']),
            min_samples=int(best['min_samples']),
            allow_single_cluster=False,
            core_dist_n_jobs=self.threads,
        )
        logging.info("Running HDBSCAN - %s" % self.clusterer)
        self.clusterer.fit(self.embeddings)

    def plot(self):
        color_palette = sns.color_palette('Paired', max(self.clusterer.labels_) + 1)
        cluster_colors = [
            color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in self.clusterer.labels_
        ]
        # cluster_member_colors = [
            # sns.desaturate(x, p) for x, p in zip(cluster_colors, self.clusterer.probabilities_)
        # ]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.embeddings[:, 0],
                   self.embeddings[:, 1],
                   s=7,
                   linewidth=0,
                   c=cluster_colors,
                   alpha=0.7)
        # ax.add_artist(legend)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of variants', fontsize=24)
        plt.savefig(self.path + '_UMAP_projection_with_clusters.png')

    def plot_distances(self):
        self.clusterer.condensed_tree_.plot(
            select_clusters=True,
            selection_palette=sns.color_palette('deep', len(set(self.clusterer.labels_))))
        plt.title('Hierarchical tree of clusters', fontsize=24)
        plt.savefig(self.path + '_UMAP_projection_with_clusters.png')

    def labels(self):
        try:
            return self.soft_clusters_capped.astype('int32')
        except AttributeError:
            return self.clusterer.labels_.astype('int32')

    def break_clusters(self):
        redo_bins = {}

        for (idx, label) in enumerate(self.clusterer.labels_):
            if label != -1:
                if self.cluster_validity[label] < 0.0:
                    try:
                        redo_bins[label.item()]["embeddings"].append(self.embeddings[idx, :])
                        redo_bins[label.item()]["indices"].append(idx)
                    except KeyError:
                        redo_bins[label.item()] = {}
                        redo_bins[label.item()]["embeddings"] = [self.embeddings[idx, :]]
                        redo_bins[label.item()]["indices"] = [idx]

        removed_labels = redo_bins.keys()
        self.clusterer.labels_[:] = [label - sum(i < label for i in removed_labels) if label not in removed_labels else label for label in self.clusterer.labels_]
        
        
        # break up very large bins. Not sure how to threshold this
        max_bin_id = max([label for label in set(self.clusterer.labels_) if label not in removed_labels]) + 1
        for (bin, values) in redo_bins.items():
            new_labels = utils.break_overclustered(np.array(values["embeddings"]), self.threads)
            for (idx, label) in zip(values["indices"], new_labels):
                if label != -1:
                    # Update labels
                    self.clusterer.labels_[idx] = label + max_bin_id
                    self.soft_clusters_capped[idx] = label + max_bin_id
                else:
                    self.clusterer.labels_[idx] = label
                    self.soft_clusters_capped[idx] = label

