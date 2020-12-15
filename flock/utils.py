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
__version__ = "0.0.1"
__maintainer__ = "Rhys Newell"
__email__ = "rhys.newell near hdr.qut.edu.au"
__status__ = "Development"

###############################################################################
# System imports
import warnings
import logging

# Function imports
import numpy as np
import pandas as pd
import multiprocessing as mp
import hdbscan


###############################################################################                                                                                                                      [44/1010]
################################ - Functions - ################################

def mp_cluster(df, n, gamma, ms, method='eom', metric='euclidean'):
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
                                allow_single_cluster=False,
                                core_dist_n_jobs=20).fit(df)

    min_cluster_size = clust_alg.min_cluster_size
    min_samples = clust_alg.min_samples
    validity_score = clust_alg.relative_validity_
    n_clusters = np.max(clust_alg.labels_)

    return (min_cluster_size, min_samples, validity_score, n_clusters)


def hyperparameter_selection(df, cores=10, method='eom', metric='euclidean'):
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
    for gamma in range(2, int(np.log(n))):
        mp_results = [pool.apply_async(mp_cluster, args=(df, n, gamma, ms, method, metric)) for ms in
                      range(1, int(2 * np.log(n)))]
        for result in mp_results:
            result = result.get()
            results.append(result)
            if result[2] >= .5:
                logging.info(
                    f'min_cluster_size = {result[0]},  min_samples = {result[1]}, validity_score = {result[2]} n_clusters = {result[3]}')


    return results


def best_validity(source):
    """
    Retrieves best clustering result based on the relative validity metric
    """
    cols = ['min_cluster_size', 'min_samples', 'validity_score', 'n_clusters']
    df =  pd.DataFrame(source, columns = cols)
    df['validity_score'] = df['validity_score'].fillna(0)
    best_validity = df.loc[df['validity_score'].idxmax()]
    return best_validity
