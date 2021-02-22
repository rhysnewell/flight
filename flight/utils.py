#!/usr/bin/env python
###############################################################################
# utils.py - File containing shared utility functions for binning.py and
#            cluster.py
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
import warnings
import logging
import re

# Function imports
import numpy as np
import pandas as pd
import multiprocessing as mp
import hdbscan
import itertools
import threadpoolctl
import imageio
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

###############################################################################                                                                                                                      [44/1010]
################################ - Functions - ################################


def plot_for_offset(embeddings, labels, x_min, x_max, y_min, y_max, n):
    matplotlib.use("agg")
    label_set = set(labels)
    color_palette = sns.color_palette('husl', max(labels) + 1)
    cluster_colors = [
        color_palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels
    ]
    #
    # cluster_member_colors = [
    # sns.desaturate(x, p) for x, p in zip(cluster_colors, self.clusterer.probabilities_)
    # ]
    fig, ax = plt.subplots(figsize=(10, 5))

    ## Plot large contig membership
    ax.scatter(embeddings[:, 0],
               embeddings[:, 1],
               s=7,
               linewidth=0,
               c=cluster_colors,
               alpha=0.7)
    ax.set(xlabel = 'UMAP dimension 1', ylabel = 'UMAP dimension 2',
           title=format("UMAP projection and HDBSCAN clustering of contigs. Iteration = %d" % (n)))

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    # plt.gca().set_aspect('equal', 'datalim')

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

def mp_cluster(df, n, gamma, ms, method='eom', metric='euclidean', allow_single_cluster=False):
    """
    Asynchronous parallel function for use with hyperparameter_selection function
    """
    clust_alg = hdbscan.HDBSCAN(algorithm='best', alpha=1.0,
                                approx_min_span_tree=True,
                                gen_min_span_tree=True,
                                leaf_size=40,
                                cluster_selection_method=method,
                                metric=metric,
                                min_cluster_size=int(gamma),
                                min_samples=ms,
                                allow_single_cluster=allow_single_cluster,
                                core_dist_n_jobs=20).fit(df)

    min_cluster_size = clust_alg.min_cluster_size
    min_samples = clust_alg.min_samples
    validity_score = clust_alg.relative_validity_
    n_clusters = np.max(clust_alg.labels_)

    return (min_cluster_size, min_samples, validity_score, n_clusters)


def hyperparameter_selection(df, cores=10, method='eom', metric='euclidean', allow_single_cluster=False, starting_size = 2):
    """
    Input:
    df - embeddings from UMAP
    cores - number of cores to run in parallel
    method - hdbscan cluster selection method

    Output:
    Quality metrics for multiple HDBSCAN clusterings
    """
    pool = mp.Pool(cores)
    warnings.filterwarnings('ignore')
    results = []
    n = df.shape[0]
    for gamma in range(starting_size, int(np.log(max(n, 3)))):
        mp_results = [pool.apply_async(mp_cluster, args=(df, n, gamma, ms, method, metric, allow_single_cluster)) for ms in
                      range(1, int(2 * np.log(n)))]
        for result in mp_results:
            result = result.get()
            results.append(result)

    pool.close()
    pool.join()

    return results


def best_validity(source):
    """
    Retrieves best clustering result based on the relative validity metric
    """
    try:
        cols = ['min_cluster_size', 'min_samples', 'validity_score', 'n_clusters']
        df =  pd.DataFrame(source, columns = cols)
        df['validity_score'] = df['validity_score'].fillna(0)
        best_validity = df.loc[df['validity_score'].idxmax()]
    except TypeError:
        best_validity = None
        
    return best_validity


# Calculates distances between clusters using minimum spanning trees
def cluster_distances(embeddings, cluster_result, threads):
    with threadpoolctl.threadpool_limits(limits=threads, user_api='blas'):
        pool = mp.Pool(max(int(threads / 5), 1))
        labels = set(cluster_result.labels_)
        logging.info(labels)
        try:
            labels.remove(-1)
        except KeyError:
            None

        dist_mat = np.zeros((len(labels), len(labels)))

        dist_results = [pool.apply_async(get_dist, args=(first, second, embeddings, cluster_result)) for (first, second) in
                        itertools.combinations(labels, 2)]

        for result in dist_results:
            result = result.get()
            dist_mat[result[0], result[1]] = result[2]
            dist_mat[result[1], result[0]] = result[2]

        pool.close()
        pool.join()

        return dist_mat


def get_dist(first, second, embeddings, cluster_result):
    try:
        # Calculate within core mutual reachability and all core distance
        (first_mr, first_core) = hdbscan.validity.all_points_mutual_reachability(embeddings, cluster_result.labels_, first)
        # Calcualtes the internal minimum spanning tree for a cluster
        (first_nodes, first_edges) = hdbscan.validity.internal_minimum_spanning_tree(first_mr.astype(np.float64))

        (second_mr, second_core) = hdbscan.validity.all_points_mutual_reachability(embeddings, cluster_result.labels_,
                                                                                   second)
        (second_nodes, second_edges) = hdbscan.validity.internal_minimum_spanning_tree(second_mr.astype(np.float64))

        # Calculates the density separation between two clusters using the above results
        sep = hdbscan.validity.density_separation(embeddings, cluster_result.labels_, first, second, first_nodes,
                                                  second_nodes, first_core, second_core)
    except ValueError:
        sep = 1.0

    return first, second, sep


def break_overclustered(embeddings, threads):
    ## Break up suspected regions of overclustering
    tuned = hyperparameter_selection(embeddings, threads)
    best = best_validity(tuned)
    if best is not None:
        clusterer = hdbscan.HDBSCAN(
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
            core_dist_n_jobs=threads,
            prediction_data=True
        )
        clusterer.fit(embeddings)
        return clusterer.labels_

    else:
        return np.array([-1 for i in range(len(embeddings))])


def special_match(strg, search=re.compile(r'[^ATGC]').search):
    return not bool(search(strg))


def sample_distance(coverage_table):
    """
    Input:
    coverage_table - a coverage and variance table for all contigs passing initial size and min coverage filters

    Output:
    A sample distance matrix - shared content between samples
    """

    # Convert coverage table to presence absence table of 1s and 0s
    presence_absence = (coverage_table.iloc[:, 3::2].values > 0).astype(int).T
    return 1 - pairwise_distances(presence_absence, metric='hamming')
