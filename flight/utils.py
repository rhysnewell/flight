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
import io
from os.path import dirname, join

# Function imports
import numpy as np
import pandas as pd
import multiprocessing as mp
import hdbscan
import itertools
import threadpoolctl


###############################################################################                                                                                                                      [44/1010]
################################ - Functions - ################################


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


def hyperparameter_selection(df, cores=10, method='eom', metric='euclidean', allow_single_cluster=False):
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
    for gamma in range(2, int(np.log(max(n, 3)))):
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
