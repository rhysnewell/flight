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
import argparse
import logging

# Function imports
import numpy as np
import hdbscan
import seaborn as sns
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
import skbio.stats.composition
from sklearn.metrics import pairwise_distances
import umap
import scipy.spatial.distance as sp_distance
# import pacmap
# import phate

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


class Cluster:
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
        metric='hellinger_distance_poisson',
        hdbscan_metric="euclidean",
        threads=8,
        b=0.5,
        a=1.48,
        random_seed=42069,
    ):
        # set_num_threads(threads)
        self.embeddings = []
        self.labels = None
        self.cluster_means = None
        self.separation = None
        self.threads = threads
        ## Set up clusterer and UMAP
        self.path = output_prefix
        self.depths = np.load(count_path)

        if self.depths.shape[1] == 1:
            self.single_sample = True
        else:
            self.single_sample = False
        ## Scale the data
        # self.sample_distance = utils.sample_distance(self.depths)

        self.clr_depths = skbio.stats.composition.clr((self.depths[:, 2:] + 1).T).T
        if self.single_sample:
            # Have to reshape after clr transformation
            self.clr_depths = self.clr_depths.reshape((-1, 1))

        # self.clr_depths = skbio.stats.composition.clr((self.depths + 1).T).T

        # self.depths[:, 2:] = self.clr_depths
        try:
            self.n_samples = (self.depths.shape[1] - 2) // 2
        except IndexError:
            self.n_samples = (self.depths.shape[0] - 2) // 2

        n_components = min(max(self.n_samples, 2), 10)
        # n_components = 2
        if n_neighbors > self.depths.shape[0]:
            n_neighbors = self.depths.shape[0] - 1

        self.rho_reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            # min_dist=min_dist,
            n_components=n_components,
            random_state=random_seed,
            # spread=1,
            metric=metrics.rho_variants,
            a=a,
            b=b,
            init="spectral"
        )
        self.distance_reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            # min_dist=min_dist,
            n_components=n_components,
            random_state=random_seed,
            # spread=1,
            # metric=metrics.euclidean_variant,
            a=a,
            b=b,
            init="spectral"
        )

        self.precomputed_reducer_low = umap.UMAP(
            metric="precomputed",
            densmap=False,
            dens_lambda=2.5,
            # output_dens=True,
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            set_op_mix_ratio=1,
            a=1.48,
            b=0.3,
            n_jobs=self.threads,
            random_state=random_seed
        )

        self.precomputed_reducer_mid = umap.UMAP(
            metric="precomputed",
            densmap=False,
            dens_lambda=2.5,
            # output_dens=True,
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            set_op_mix_ratio=1,
            a=1.58,
            b=0.4,
            n_jobs=self.threads,
            random_state=random_seed
        )

        self.precomputed_reducer_high = umap.UMAP(
            metric="precomputed",
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            set_op_mix_ratio=1,
            a=1.68,
            b=0.5,
            n_jobs=self.threads,
            random_state=random_seed
        )

        if precomputed:
            self.metric = "precomputed"
        else:
            self.metric = "euclidean"

    def filter(self):
        # Not sure to include this
        pass

    def filter(self):
        # Not sure to include this
        pass

    def fit_transform(self, stat, second_pass=False):
        ## Calculate the UMAP embeddings
        try:
            if self.depths.shape[0] >= 5:
                # dist_embeddings = self.distance_reducer.fit(self.clr_depths)
                # rho_embeddings = self.rho_reducer.fit(self.clr_depths)
                # intersect = dist_embeddings * rho_embeddings
                self.precomputed_reducer_low.fit(sp_distance.squareform(stat))
                self.precomputed_reducer_mid.fit(sp_distance.squareform(stat))
                self.precomputed_reducer_high.fit(sp_distance.squareform(stat))
                self.embeddings = self.precomputed_reducer_low.embedding_
                # self.embeddings = self.distance_reducer.fit_transform(self.clr_depths)
            else:
                self.precomputed_reducer_low.embedding_ = self.clr_depths
                self.precomputed_reducer_mid.embedding_ = self.clr_depths
                self.precomputed_reducer_high.embedding_ = self.clr_depths
                self.embeddings = self.clr_depths
        except TypeError as e:
            if not second_pass:
                ## TypeError occurs here on sparse input. So need to lower the number of components
                ## That are trying to be embedded to. Choose minimum of 2
                self.precomputed_reducer_low.n_components = 2
                self.precomputed_reducer_mid.n_components = 2
                self.precomputed_reducer_high.n_components = 2
                self.fit_transform(stat, True)
            else:
                raise e

    def cluster(self, embeddings):
        if embeddings.shape[0] >= 5 and len(embeddings.shape) >= 2:
            try:
                ## Cluster on the UMAP embeddings and return soft clusters
                tuned = utils.hyperparameter_selection(embeddings, self.threads, metric=self.metric, starting_size=max(2, round(embeddings.shape[0] * 0.05)), use_multi_processing=False)
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
                # logging.info("Running HDBSCAN - %s" % self.clusterer)
                self.clusterer.fit(embeddings)
                try:
                    self.validity, self.cluster_validity = hdbscan.validity.validity_index(embeddings.astype(np.float64),
                                                                                           self.clusterer.labels_,
                                                                                           per_cluster_scores=True)
                except (ValueError, SystemError):
                    self.validity = None
                    self.cluster_validity = [0.5 for i in range(len(set(self.clusterer.labels_)))]

                return self.clusterer.labels_
            except TypeError:
                return np.array([-1 for _ in range(embeddings.shape[0])])
        else:
            return np.array([-1 for _ in range(embeddings.shape[0])])

    """
    Reclusters unclustered elements and updates the labels array with the potential new label making sure to make the label
    at least 1 value higher than the previous max label value
    """
    def recover_unbinned(self):
        unclustered_truth_array = self.labels == -1
        unclustered_embeddings = self.embeddings[unclustered_truth_array]
        if unclustered_embeddings.shape[0] > 5:
            unclustered_labels = self.cluster(unclustered_embeddings)

            if unclustered_labels is not None:
                previous_max_label = np.max(self.labels)

                unclustered_idx = 0
                for (idx, label) in enumerate(self.labels):
                    if label == -1:
                        new_label = unclustered_labels[unclustered_idx]
                        if new_label != -1:
                            new_label += previous_max_label + 1
                            self.labels[idx] = new_label
                        unclustered_idx += 1

    def recluster(self):
        unique_labels = set(self.labels)
        logging.info("Refining clusters...")
        if len(unique_labels) == 1 and -1 in unique_labels:
            self.labels = self.labels + 1
        else:
            for label in unique_labels:
                if label != -1:
                    truth_array = self.labels == label
                    embeddings_for_label = self.embeddings[truth_array]
                    recluster_attempt = self.cluster(embeddings_for_label)
                    if recluster_attempt is not None:
                        try:
                            cluster_validity = hdbscan.validity.validity_index(embeddings_for_label.astype(np.float64), np.array(recluster_attempt), per_cluster_scores=False)
                        except (ValueError, SystemError):
                            cluster_validity = -1

                        if cluster_validity >= 0.9:
                            # print("reclustering %d validity %.3f" % (label, cluster_validity))
                            if not np.any(recluster_attempt == -1):
                                # shift all labels greater than current label down by one since this label is fully
                                # removed
                                self.labels[self.labels >= label] = self.labels[self.labels >= label] - 1

                            previous_max_label = np.max(self.labels)

                            new_labels_idx = 0
                            for (idx, label) in enumerate(truth_array):
                                if label:
                                    new_label = recluster_attempt[new_labels_idx]
                                    if new_label != -1:
                                        new_label += previous_max_label + 1
                                        self.labels[idx] = new_label
                                    new_labels_idx += 1

    def cluster_separation(self):
        # dist_mat = utils.cluster_distances(self.embeddings, self.labels, self.threads)

        labels_no_unlabelled = set(self.labels[self.labels != -1])
        if len(labels_no_unlabelled) > 1:
            cluster_centres = [[] for _ in range(len(labels_no_unlabelled))]
            for label in labels_no_unlabelled:
                cluster_centres[label] = self.cluster_means[label]

            dist_mat = pairwise_distances(cluster_centres)

            return dist_mat
        else:
            return np.zeros((1, 1))

    def combine_bins(self):
        not_neg_labs = self.labels[self.labels != -1]
        # recscale the labels so that they increment by one
        for (i, previous_label) in enumerate(set(not_neg_labs)):
            not_neg_labs[not_neg_labs == previous_label] = i
        self.labels[self.labels != -1] = not_neg_labs

        self.cluster_means = self.get_cluster_means()
        self.separation = self.cluster_separation()

        clocked = set()
        combine_these = {}

        for i in range(self.separation.shape[0]):
            if i not in clocked:
                for j in range(self.separation.shape[1]):
                    if j not in combine_these.keys() and i != j:
                        if self.separation[i, j] <= 0.1:
                            try:
                                combine_these[i].append(j)
                            except KeyError:
                                combine_these[i] = [j]
                                clocked.add(j)

        if len(combine_these.keys()) >= 1:
            for (base_label, other_labels) in combine_these.items():
                # change the labels over to the base label
                for other_label in other_labels:
                    self.labels[self.labels == other_label] = base_label

            self.combine_bins()


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
        color_palette = sns.color_palette('Paired', max(self.labels) + 1)
        cluster_colors = [
            color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in self.labels
        ]
        # cluster_member_colors = [
        # sns.desaturate(x, p) for x, p in zip(cluster_colors, self.clusterer.probabilities_)
        # ]
        try:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.scatter(self.embeddings[:, 0],
                       self.embeddings[:, 1],
                       s=7,
                       linewidth=0,
                       c=cluster_colors,
                       alpha=0.7)

            for label, coords in self.cluster_means.items():
                if label != -1:
                    plt.annotate(
                        label,
                        coords,
                        size = 14,
                        weight = 'bold',
                        color = color_palette[label]
                    )

            # ax.add_artist(legend)
            plt.gca().set_aspect('equal', 'datalim')
            plt.title('UMAP projection of variants - %d Clusters' % len(self.cluster_means), fontsize=24)
            plt.savefig(self.path + '_UMAP_projection_with_clusters.png')

        except IndexError:
            pass

    def get_cluster_means(self):
        result = {}
        cluster_size = {}
        for (i, label) in enumerate(self.labels):
            try:
                label_val = result[label]
                try:
                    label_val[0] += self.embeddings[i, 0]
                    label_val[1] += self.embeddings[i, 1]
                except IndexError:
                    label_val[0] += self.embeddings[0]
                    label_val[1] += self.embeddings[1]
                cluster_size[label] += 1
            except KeyError:
                try:
                    result[label] = list(self.embeddings[i, :2])
                except IndexError:
                    result[label] = list(self.embeddings[:2]) # when only one variant

                cluster_size[label] = 1

        new_result = {}
        for (key, value) in result.items():
            new_values = [val / cluster_size[key] for val in value]
            new_result[key] = new_values

        return new_result

    def plot_distances(self):
        self.clusterer.condensed_tree_.plot(
            select_clusters=True,
            selection_palette=sns.color_palette('deep', len(set(self.clusterer.labels_))))
        plt.title('Hierarchical tree of clusters', fontsize=24)
        plt.savefig(self.path + '_UMAP_projection_with_clusters.png')

    def labels_for_printing(self):
        try:
            return self.labels.astype('int32')
        except AttributeError:
            return self.labels.astype('int32')

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
        self.clusterer.labels_[:] = [
            label - sum(i < label for i in removed_labels) if label not in removed_labels else label for label in
            self.clusterer.labels_]

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

