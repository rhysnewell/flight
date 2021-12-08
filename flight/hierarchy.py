#!/usr/bin/env python
###############################################################################
#                                                                             #
#    hierarchy.py                                                             #
#                                                                             #
#    Working with hierarchical clusterings                                    #
#                                                                             #
#    Copyright (C) Tim Lamberton                                              #
#                                                                             #
###############################################################################
#                                                                             #
#          .d8888b.                                    888b     d888          #
#         d88P  Y88b                                   8888b   d8888          #
#         888    888                                   88888b.d88888          #
#         888        888d888 .d88b.   .d88b.  88888b.  888Y88888P888          #
#         888  88888 888P"  d88""88b d88""88b 888 "88b 888 Y888P 888          #
#         888    888 888    888  888 888  888 888  888 888  Y8P  888          #
#         Y88b  d88P 888    Y88..88P Y88..88P 888 d88P 888   "   888          #
#          "Y8888P88 888     "Y88P"   "Y88P"  88888P"  888       888          #
#                                             888                             #
#                                             888                             #
#                                             888                             #
#                                                                             #
###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

__author__ = "Tim Lamberton"
__copyright__ = "Copyright 2016"
__credits__ = ["Tim Lamberton"]
__license__ = "GPL3"
__maintainer__ = "Tim Lamberton"
__email__ = "t.lamberton@uq.edu.au"

###############################################################################

import numpy as np
import scipy.cluster.hierarchy as sp_hierarchy
import scipy.spatial.distance as sp_distance
import operator

np.seterr(all='raise')


###############################################################################
###############################################################################
###############################################################################
###############################################################################

def maxscoresbelow(Z, scores, fun=np.maximum):
    """Compute maximum cumulative score of disjoint cluster sets below
    cluster hierarchy nodes.

    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    scores : ndarray
        `scores[i]` for `i<n` is defines the quality score for the `i`th
        singleton node, and for `i>=n` is the score for the cluster encoded
        by the `(i-n)`-th row in `Z`.
    fun : function, optional
        `fun(a, b)` is used to compute the accumulative score of the
        child nodes to propogate. By default, the scores for child nodes are
        added together.

    Returns
    -------
    maxscores : ndarray
        `maxscores[i]` is the maximum coefficient of any disjoint set of
        clusters below cluster `i`.
    """
    Z = np.asarray(Z)
    n = Z.shape[0] + 1
    max_scores = np.copy(scores)
    max_below_scores = np.zeros(n - 1, dtype=max_scores.dtype)

    # Bottom-up traversal
    for i in range(n - 1):
        left_child = int(Z[i, 0])
        right_child = int(Z[i, 1])
        current_node = n + i
        current_score = max_scores[current_node]
        max_below_scores[i] = fun(max_scores[left_child], max_scores[right_child])
        max_scores[current_node] = np.maximum(current_score, max_below_scores[i])

    return max_below_scores


def iterlinkage(Z):
    """Bottom-up iteration over leaf node sets for cluster hierarchy nodes"""
    Z = np.asarray(Z)
    n = Z.shape[0] + 1

    # Store cluster leaves
    leaves_dict = dict([(i, [i]) for i in range(n)])

    # Bottom-up traversal
    for i in range(n - 1):
        left_child = int(Z[i, 0])
        right_child = int(Z[i, 1])
        current_node = n + i

        # update leaf cache
        current_leaves = leaves_dict[left_child] + leaves_dict[right_child]
        del leaves_dict[left_child]
        del leaves_dict[right_child]
        leaves_dict[current_node] = current_leaves

        yield current_leaves


def fcluster_merge(Z, merge, return_nodes=False):
    """Partition a hierarchical clustering by flattening clusters.

    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    merge : ndarray
        Boolean array. `merge[i]` indicates whether the cluster represented by
        `Z[i, :]` should be flattened.
    return_nodes : bool
        If True, also return array of flat cluster root nodes.

    Returns
    -------
    T : ndarray
        1-D array. `T[i]` is the flat cluster number to which original
        observation `i` belongs.
    nodes : ndarray
        1-D array. `nodes[i]` is the cluster index corresponding to the flat
        cluster of the `i`th original obseration. Only provided if
        `return_nodes` is True.
    """
    Z = np.asarray(Z)
    n = Z.shape[0] + 1

    # Compute leaf clusters
    leaders = np.arange(n)

    for (i, leaves) in enumerate(iterlinkage(Z)):
        if merge[i]:
            leaders[leaves] = n + i

    (_, bids) = np.unique(leaders, return_inverse=True)

    if not return_nodes:
        return bids

    out = (bids,)
    if return_nodes:
        out += (leaders,)
    return out


def flatten_nodes(Z):
    """Map nested cluster nodes to their earliest equal height ancestor
    """
    Z = np.asarray(Z)
    n = Z.shape[0] + 1

    node_ids = np.arange(n - 1)
    for i in range(n - 2, -1, -1):
        children = Z[i, :2].astype(int)
        for c in children:
            if c >= n and Z[i, 2] == Z[c - n, 2]:
                node_ids[c - n] = node_ids[i]

    return node_ids


def embed_nodes(Z, leaves):
    """Map cluster nodes to an embedded tree generated by a set of leaf nodes

    Nodes of the embedded tree have more than one child with descendents in
    the generating set of leaves.

    Returns a list of for corresponding nodes in the original tree of the earliest
    embedded descendent, or -1 if no descendent is embedded.
    """
    Z = np.asarray(Z)
    n = Z.shape[0] + 1

    is_leaf = np.zeros(2 * n - 1, dtype=bool)
    is_leaf[leaves] = True
    num_leaves_below = is_leaf.astype(int)
    num_leaves_below[n:] = maxscoresbelow(Z, is_leaf.astype(int), operator.add)

    is_embedded_parent = is_leaf.copy()
    is_embedded_parent[n:] = maxscoresbelow(Z, num_leaves_below, max) < num_leaves_below[n:]

    node_id_if_embedded = np.full(2 * n - 1, -1, dtype=int)
    node_id_if_embedded[is_embedded_parent] = np.flatnonzero(is_embedded_parent)
    node_ids = maxscoresbelow(Z, node_id_if_embedded, max)
    node_ids[is_embedded_parent[n:]] = np.flatnonzero(is_embedded_parent[n:]) + n

    return node_ids


def linkage_from_reachability(o, d):
    """Hierarchical clustering from reachability ordering and distances

    Paramters
    ---------
    o : ndarray
        1-D array. `o[i]` is the index of the original observation reached `i`th
        in the reachability traversal. Must be a permutation array.
    d : ndarray
        1-D array. `d[i]` is the distance to the `o[i]`th observation in the
        reachabililty traversal.

    Returns
    -------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
        See `scipy.cluster.hierarchy.linkage` for information on encoding.
    """
    o = np.asarray(o)
    d = np.asarray(d)
    # check if o is a permutation array
    if np.all(np.argsort(o) != o):
        raise ValueError("Argument for parameter `o` must be a valid permutation array.")

    n = len(o)
    Z = np.empty((n - 1, 4), dtype=d.dtype)

    # observations ordered from smallest to largest density distance. clusters
    # will be formed by repeatedly splitting the parent cluster containing the
    # largest distance.
    splits = reachability_splits(d)
    # dict of { node_id: (range_from, range_to) }
    # this encodes the range of `o` of observations below the node with `node_id` in the hierarchy
    # the root node with id `2*n-2` contains the whole dataset
    indices_dict = dict([(2 * n - 2, (0, n))])

    for i in range(n - 2, -1, -1):
        # ordering positions of observations in current cluster
        (low, high) = indices_dict.pop(n + i)
        # determine the next largest observation at which to split the cluster
        split = splits[i]
        # the split clusters will occupy ordering positions in the ranges
        # (low, split) and (split, high)

        # determine the iterations at which the split clusters will themselves
        # be split by finding the descendent observations with the largest
        # density distance in the `splits` list. The rows in the linkage matrix
        # encoding are in reverse splitting order (i.e. last row is first split,
        # etc.).
        if split == low + 1:
            # singleton left cluster
            left_node = o[low]
        else:
            # look in range (low, split) for position of observation with
            # largest density distance.
            left_node = np.flatnonzero(np.logical_and(low <= splits[:i], splits[:i] < split))[-1] + n
            indices_dict[left_node] = (low, split)

        if split == high - 1:
            # singleton right cluster
            right_node = o[split]
        else:
            # look in range (split, high) for position of observation with
            # largest density distance.
            right_node = np.flatnonzero(np.logical_and(split <= splits[:i], splits[:i] < high))[-1] + n
            indices_dict[right_node] = (split, high)

        if left_node < right_node:
            Z[i, :2] = np.array([left_node, right_node])
        else:
            Z[i, :2] = np.array([right_node, left_node])
        Z[i, 2] = d[split]
        Z[i, 3] = high - low

    return Z


def reachability_splits(d):
    """Returns array of reachability indices which divide clusters at each level"""
    return np.concatenate((np.asarray(d)[1:].argsort() + 1, [0]))  # pretend first observation is largest


def descendents(Z, indices, inclusive=False):
    """Compute descendent nodes of indices

    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    indices : ndarray
        1-D array of node indices.
    inclusive : boolean, optional
        If `True`, indices are counted as their own ancestors.

    Returns
    -------
    descendents : ndarray
        1-D array of node indices of the union of the sets of descendents of input nodes.
    """
    Z = np.asarray(Z)
    n = Z.shape[0] + 1
    is_descendent = np.zeros(2 * n - 1, dtype=bool)
    is_descendent_or_index = is_descendent.copy()
    is_descendent_or_index[indices] = True
    for i in range(n - 2, -1, -1):
        left_child = int(Z[i, 0])
        is_descendent[left_child] = is_descendent[left_child] or is_descendent_or_index[n + i]
        is_descendent_or_index[left_child] = is_descendent_or_index[left_child] or is_descendent[left_child]

        right_child = int(Z[i, 1])
        is_descendent[right_child] = is_descendent[right_child] or is_descendent_or_index[n + i]
        is_descendent_or_index[right_child] = is_descendent_or_index[right_child] or is_descendent[right_child]

    if inclusive:
        return np.flatnonzero(is_descendent_or_index)
    else:
        return np.flatnonzero(is_descendent)


def ancestors(Z, indices, inclusive=False):
    """Compute ancestor node indices.

    Parameters
    ----------
    Z : ndarray
        Linkage matrix encoding hierarchical clustering.
    indices : ndarray
        1-D array of node indices.
    inclusive : boolean, optional
        If `True`, indices are counted as their own ancestors.

    Returns
    -------
    ancestors : ndarray
        1-D array of node indices of the union of the sets of ancestors of input nodes.
    """
    Z = np.asarray(Z)
    n = Z.shape[0] + 1
    isancestor = np.zeros(2 * n - 1, dtype=bool)
    isancestor_or_index = isancestor.copy()
    isancestor_or_index[indices] = True
    for i in range(n - 1):
        isancestor[i + n] = isancestor[i + n] or isancestor_or_index[Z[i, :2].astype(int)].any()
        isancestor_or_index[i + n] = isancestor_or_index[i + n] or isancestor[i + n]

    if inclusive:
        return np.flatnonzero(isancestor_or_index)
    else:
        return np.flatnonzero(isancestor)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
