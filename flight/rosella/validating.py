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
import copy
import logging

# Function imports
import numpy as np
from numba import njit, set_num_threads
import umap
import seaborn as sns
import matplotlib
import sklearn.metrics as sk_metrics
import concurrent.futures
import threadpoolctl
import random

# self imports
import flight.metrics as metrics
import flight.utils as utils
from flight.rosella.clustering import Clusterer, iterative_clustering_static
from flight.rosella.embedding import Embedder, fit_transform_static, multi_transform_static, switch_intersector_static
from flight.DBCV import DBCV

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

class Validator(Clusterer, Embedder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def validate_bins(self, plots, n, x_min, x_max, y_min, y_max,
                      bin_unbinned=False, reembed=False,
                      size_only=False, big_only=False,
                      quick_filter=False, size_filter=False, debug=False,
                      force=False, truth_array=None, dissolve=False):
        """
        Function for deciding whether a bin needs to be reembedded or split up
        Uses internal bin statistics, mainly mean ADP and Rho values
        """

        n_samples, sample_distances = self.get_n_samples_and_distances()

        bins_to_remove = []
        new_bins = {}
        new_bin_counter = 0
        logging.debug("Checking bin internal distances...")
        big_tids = []
        reembed_separately = []  # container for bin ids that look like chimeras
        force_new_clustering = []
        lower_thresholds = []
        reembed_if_no_cluster = []
        switches = []
        bins = self.bins.keys()
        for bin in bins:
            logging.debug("Beginning check on bin: ", bin)
            tids = self.bins[bin]
            if len(tids) == 1:
                continue
            elif bin == 0:
                continue

            if quick_filter or size_filter:
                ## Remove stuff that is obviously wrong
                contigs, log_lengths, tnfs = self.extract_contigs(tids)
                bin_size = contigs['contigLen'].sum()
                if bin_size < self.min_bin_size:
                    self.unbinned_tids = self.unbinned_tids + tids
                    bins_to_remove.append(bin)
                else:
                    try:
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

                        per_contig_avg = np.array(per_contig_avg)

                        removed = []

                        if debug:
                            print('before check for distant contigs: ', len(tids))
                            _, _, _, _ = self.bin_stats(bin)

                        md_median = np.median(per_contig_avg[:, 0])
                        agg_median = np.median(per_contig_avg[:, 3])
                        rho_median = np.median(per_contig_avg[:, 1])
                        euc_median = np.median(per_contig_avg[:, 2])

                        if mean_md >= 0.15 or mean_agg >= 0.25:
                            # Simply remove
                            if quick_filter:
                                for (tid, avgs) in zip(tids, per_contig_avg):
                                    if ((avgs[0] >= 0.65 or avgs[3] >= 0.5) and
                                        (avgs[1] > 0.15 or avgs[2] >= 3)) or \
                                            ((avgs[0] >= 0.5 or avgs[3] >= 0.5) and
                                             (avgs[1] >= 0.5 or avgs[2] >= 6)):
                                        removed.append(tid)
                            elif size_filter:
                                # check internal connections within this bin
                                # disconnected_tids = self.large_contigs[self.check_contigs(tids)]['tid']
                                # removed += list(disconnected_tids.values)
                                for (tid, avgs) in zip(tids, per_contig_avg):
                                    if ((avgs[0] >= md_median + 0.15
                                         or avgs[3] >= agg_median + 0.15) and
                                            ((avgs[1] >= rho_median + 0.05
                                              or avgs[2] >= euc_median + 0.5)
                                             or (avgs[1] > 0.1 or avgs[2] >= 3.5)
                                            or (avgs[0] >= md_median + 0.3
                                         or avgs[3] >= agg_median + 0.3)
                                            )):
                                        removed.append(tid)

                        remove = False
                        if len(removed) > 0 and len(removed) != len(tids):
                            new_bins[new_bin_counter] = []
                            [(tids.remove(r), new_bins[new_bin_counter].append(r)) for r in removed]
                            new_bin_counter += 1

                            current_contigs, current_lengths, current_tnfs = self.extract_contigs(tids)
                            if current_contigs['contigLen'].sum() <= self.min_bin_size:
                                [self.unbinned_tids.append(tid) for tid in tids]
                                remove = True

                            if len(tids) == 0 or remove:
                                bins_to_remove.append(bin)
                    except ZeroDivisionError:
                        # Only one contig left, break out
                        continue

            elif big_only:

                # filtering = True
                removed_single = []  # Single contig bin
                contigs, log_lengths, tnfs = self.extract_contigs(tids)

                bin_size = contigs['contigLen'].sum()

                if bin_size >= 3e6 and len(tids) >= 2:

                    # Extract current contigs and get statistics
                    try:
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

                        per_contig_avg = np.array(per_contig_avg)
                    except ZeroDivisionError:
                        # Only one contig left, break out
                        break


                    if len(tids) == 2:
                        # Lower thresholds for fewer contigs
                        md_filt = max(0.2, mean_md)
                        agg_filt = max(0.35, mean_agg)
                        euc_filt = 2
                        rho_filt = 0.05
                        # Two contigs by themselves that are relatively distant. Remove them separately
                    else:
                        md_filt = max(0.35, mean_md)
                        agg_filt = max(0.4, mean_agg)
                        euc_filt = 2
                        rho_filt = 0.05

                    f_level = 0.4
                    m_level = 0.35

                    if ((round(mean_md, 2) >= md_filt
                         or round(mean_agg, 2) >= agg_filt)
                        and bin_size > 1e6) or bin_size >= 15e6:
                        if debug:
                            print("Checking big contigs for bin: ", bin)
                        for max_idx in range(per_contig_avg.shape[0]):
                            max_values = per_contig_avg[max_idx, :]
                            contig_length = contigs['contigLen'].iloc[max_idx]
                            if debug:
                                print("Contig size and tid: ", contig_length, tids[max_idx])

                            if contig_length >= 2e6:
                                if debug:
                                    print("Found large contig: ", max_idx, tids[max_idx])
                                if (max_values[3] >= agg_filt or max_values[0] >= md_filt) and \
                                        (max_values[1] >= rho_filt
                                         or max_values[2] >= euc_filt) or bin_size >= 15e6:
                                    if debug:
                                        print("Removing contig: ", max_idx, tids[max_idx])
                                    removed_single.append(tids[max_idx])

                    if len(removed_single) > 0:
                        [(big_tids.append(r), tids.remove(r)) for r in removed_single]

            elif reembed:

                contigs, log_lengths, tnfs = self.extract_contigs(tids)
                bin_size = contigs['contigLen'].sum()

                if debug:
                    print(bin, bin_size)

                if bin not in self.survived:

                    try:
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
                        per_contig_avg = np.array(per_contig_avg)

                    except ZeroDivisionError:
                        continue

                    if debug:
                        print('before check for distant contigs: ', len(tids))
                        _, _, _, _ = self.bin_stats(bin)

                    f_level = 0.25
                    m_level = 0.15
                    shared_level = 0.1

                    # Always check bins with bad bin stats or if they are large, just for sanity check
                    if bin_size >= 8e6 or len(tids) >= 250 or \
                            (((mean_agg >= f_level or mean_md >= m_level) and (mean_tnf >= 0.1 or round(mean_euc, 2) >= 2.5))
                        and bin_size > 1e6) \
                        or (round(per_contig_avg[:, 0].std(), 2) >= 0.1
                            or round(per_contig_avg[:, 3].std(), 2) >= 0.1
                            or round(per_contig_avg[:, 1].std(), 2) >= 0.1
                            or round(per_contig_avg[:, 2].std(), 2) >= 1):
                        logging.debug(bin, mean_md, mean_tnf, mean_agg, len(tids))
                        reembed_separately.append(bin)
                        if debug:
                            print("Reclustering bin %d" % bin)
                        # if (mean_md >= m_level
                        #     or mean_agg >= f_level) and (mean_tnf >= 0.1 or mean_euc >= 3):
                        #     lower_thresholds.append(0.85)
                        if mean_tnf >= 0.15 or mean_euc >= 4.5:
                            lower_thresholds.append(0.5)
                        elif (mean_md >= 0.25 or mean_agg >= 0.35) and (mean_tnf >= 0.1 or mean_euc >= 3.0):
                            lower_thresholds.append(0.5)
                        elif (mean_md >= 0.2 or mean_agg >= 0.3) and (mean_tnf >= 0.1 or mean_euc >= 3.0):
                            lower_thresholds.append(0.85)
                        else:
                            lower_thresholds.append(0.85)
                        force_new_clustering.append(False)  # send it to regular hell
                        reembed_if_no_cluster.append(True)
                    else:
                        reembed_separately.append(bin)
                        force_new_clustering.append(False)  # send it to regular hell
                        reembed_if_no_cluster.append(False)  # take it easy, okay?
                        lower_thresholds.append(0.85)
                        # if debug:
                        #     print("bin survived %d" % bin)
                        #     self.bin_stats(bin)
                        # self.survived.append(bin)

                    local_switches = [] # store which umap projections to use 0 = md, 1 = rho/tnf, 2 = euc
                    # if mean_md <= 0.25:
                    local_switches.append(0)
                    if mean_tnf <= 0.1:
                        local_switches.append(1)
                    if (mean_euc <= 3 and not self.use_euclidean) or (mean_euc <= 4 and self.use_euclidean):
                        local_switches.append(2)

                    if len(local_switches) == 0:
                        if self.use_euclidean:
                            local_switches = [0, 1, 2]
                        else:
                            local_switches = [0, 1]
                    #
                    # if reembed_if_no_cluster[-1]:
                    #     print("Using this switch [%s] for bin %d with "
                    #           "{ mean_md %.3f, mean_tnf %.3f, mean_euc %.3f, size %d } " %
                    #           (",".join(map(str, local_switches)),
                    #            bin, mean_md, mean_tnf, mean_euc, bin_size))

                    switches.append(local_switches)
                else:
                    logging.debug(bin, self.survived)

        try:
            max_bin_id = max(self.bins.keys()) + 1
        except ValueError:
            max_bin_id = 1
        for k, v in new_bins.items():
            self.bins[max_bin_id + k] = list(np.sort(np.array(v)))


        numpy_thread_limit = max(self.threads // 5, 1)
        if numpy_thread_limit == 1:
            worker_limit = self.threads
        else:
            worker_limit = max(self.threads // numpy_thread_limit, 1)

        set_num_threads(numpy_thread_limit)
        with threadpoolctl.threadpool_limits(limits=numpy_thread_limit, user_api='blas'):
            with concurrent.futures.ProcessPoolExecutor(max_workers=worker_limit) as executor:
                if self.n_samples > 0:
                    n_samples = self.n_samples
                    sample_distances = self.short_sample_distance
                else:
                    n_samples = self.long_samples
                    sample_distances = self.long_sample_distance

                results = [executor.submit(reembed_static,
                                           bin,
                                           self.bins[bin],
                                           self.extract_contigs(self.bins[bin]),
                                           self.embeddings[
                                               self.large_contigs[
                                                   ~self.disconnected][
                                                   ~self.disconnected_intersected
                                               ]['tid'].isin(self.bins[bin])
                                           ],
                                           n_samples, sample_distances,
                                           self.a, self.b,
                                           self.n_neighbors,
                                           min_validity,
                                           reembed_cluster,
                                           force_new,
                                           False,
                                           switch,
                                           False
                                           ) for
                           (bin, force_new, min_validity, reembed_cluster, switch)
                           in zip(
                        reembed_separately,
                        force_new_clustering,
                        lower_thresholds,
                        reembed_if_no_cluster,
                        switches
                    )]

                for f in concurrent.futures.as_completed(results):
                    result = f.result()
                    plots, remove = self.handle_new_embedding(
                        result[0], result[1], result[2], result[3], result[4],
                        plots, n, x_min, x_max, y_min, y_max, result[5], result[6], debug=False
                    )
                    if debug:
                        print("Problem bin result... removing: ", remove)

                    if remove:
                        if debug:
                            print("Removing bin %d..." % result[0])
                        bins_to_remove.append(result[0])
                        self.overclustered = True
                    elif result[5]:
                        logging.debug("Removing bin %d through force..." % result[0])
                        big_tids = big_tids + self.bins[result[0]]
                        bins_to_remove.append(result[0])
                    else:
                        if debug:
                            print("Keeping bin %d..." % result[0])
                        self.survived.append(result[0])
        #
        # if self.n_samples > 0:
        #     n_samples = self.n_samples
        #     sample_distances = self.short_sample_distance
        # else:
        #     n_samples = self.long_samples
        #     sample_distances = self.long_sample_distance
        #
        # results = [reembed_static(
        #                            bin,
        #                            self.bins[bin],
        #                            self.extract_contigs(self.bins[bin]),
        #                            self.embeddings[
        #                                self.large_contigs[
        #                                    ~self.disconnected][
        #                                    ~self.disconnected_intersected
        #                                ]['tid'].isin(self.bins[bin])
        #                            ],
        #                            n_samples, sample_distances,
        #                            self.a, self.b,
        #                            50,
        #                            min_validity,
        #                            reembed_cluster,
        #                            force_new,
        #                            False,
        #                            switch,
        #                            False
        #                            ) for
        #            (bin, force_new, min_validity, reembed_cluster, switch)
        #            in zip(
        #         reembed_separately,
        #         force_new_clustering,
        #         lower_thresholds,
        #         reembed_if_no_cluster,
        #         switches
        #     )]
        #
        # for result in results:
        #     # result = f.result()
        #     plots, remove = self.handle_new_embedding(
        #         result[0], result[1], result[2], result[3], result[4],
        #         plots, n, x_min, x_max, y_min, y_max, result[5], result[6]
        #     )
        #     if debug:
        #         print("Problem bin result... removing: ", remove)
        #
        #     if remove:
        #         if debug:
        #             print("Removing bin %d..." % result[0])
        #         bins_to_remove.append(result[0])
        #         self.overclustered = True
        #     elif result[5]:
        #         logging.debug("Removing bin %d through force..." % result[0])
        #         big_tids = big_tids + self.bins[result[0]]
        #         bins_to_remove.append(result[0])
        #     else:
        #         if debug:
        #             print("Keeping bin %d..." % result[0])
        #         self.survived.append(result[0])

        #
        #
        # for bin, force_new, min_validity, reembed_cluster, switch in zip(
        #     reembed_separately,
        #     force_new_clustering,
        #     lower_thresholds,
        #     reembed_if_no_cluster,
        #     switches
        # ):
        #     tids = self.bins[bin]
        #
        #     logging.debug("Checking bin %d..." % bin)
        #     try:
        #         max_bin_id = max(self.bins.keys()) + 1
        #     except ValueError:
        #         max_bin_id = 1
        #
        #     if isinstance(max_bin_id, np.int64):
        #         max_bin_id = max_bin_id.item()
        #
        #     plots, remove = self.reembed(tids, max_bin_id, plots,
        #                                  x_min, x_max, y_min, y_max, n,
        #                                  default_min_validity=min_validity,
        #                                  force=force_new,
        #                                  reembed=reembed_cluster,
        #                                  switch=switch,
        #                                  truth_array=truth_array, debug=debug)
        #     if debug:
        #         print("Problem bin result... removing: ", remove)
        #
        #     if remove:
        #         if debug:
        #             print("Removing bin %d..." % bin)
        #         bins_to_remove.append(bin)
        #         self.overclustered = True
        #     elif force_new:
        #         logging.debug("Removing bin %d through force..." % bin)
        #         big_tids = big_tids + self.bins[bin]
        #         bins_to_remove.append(bin)
        #     else:
        #         if debug:
        #             print("Keeping bin %d..." % bin)
        #         self.survived.append(bin)

        for k in bins_to_remove:
            try:
                _ = self.bins.pop(k)
            except KeyError:
                pass

        try:
            max_bin_id = max(self.bins.keys()) + 1
        except ValueError:
            max_bin_id = 1

        if isinstance(max_bin_id, np.int64):
            max_bin_id = max_bin_id.item()

        for idx in big_tids:
            if self.large_contigs[self.large_contigs['tid'] == idx]['contigLen'].iloc[0] >= self.min_bin_size:
                max_bin_id += 1
                self.bins[max_bin_id] = [idx]
            else:
                try:
                    self.bins[0].append(idx)
                except KeyError:
                    self.bins[0] = [idx]

        if bin_unbinned:
            for idx in self.unbinned_tids:
                if self.large_contigs[self.large_contigs['tid'] == idx]['contigLen'].iloc[0] >= self.min_bin_size:
                    max_bin_id += 1
                    self.bins[max_bin_id] = [idx]
                else:
                    try:
                        self.bins[0].append(idx)
                    except KeyError:
                        self.bins[0] = [idx]

        return plots, n

    def combine_bins(self, threshold = 0.001):
        """
        Takes bins and embeddings calculates the distances between clusters, combining clusters that are within
        a certain threshold
        """
        distance_matrix = utils.cluster_distances(self.embeddings, self.labels, self.threads)

        # Sort original labels, lowest labels id is idx 0
        sorted_labels = list(np.sort(np.array(list(set(self.labels)))))
        try:
            sorted_labels.remove(-1)
        except KeyError:
            pass

        sorted_labels = np.array(sorted_labels)

        try:
            max_bin_id = max(self.bins.keys())
        except ValueError:
            max_bin_id = 0

        # Clustered bins
        bin_sets = {}
        for idx, i in enumerate(range(distance_matrix.shape[0])):
            truth_array = distance_matrix[i, :] <= threshold
            if truth_array.sum() > 1:
                close_labels = sorted_labels[np.argwhere(truth_array)[0]]
                current_label = sorted_labels[idx]
                if len(bin_sets.keys()) > 0:
                    close_sets = []
                    do_nothing = False
                    for (key, bin_set) in bin_sets.items():
                        if current_label in bin_set:
                            do_nothing = True
                            break
                        elif any(label in bin_set for label in close_labels):
                            close_sets.append(key)

                    if not do_nothing:
                        if len(close_sets) > 1:
                            continue
                        elif len(close_sets) == 1:
                            bin_sets[close_sets[0]].add(current_label)
                        else:
                            # new set
                            bin_sets[idx] = set(close_labels)
                else:
                    bin_sets[idx] = set(close_labels)
            else:
                bin_sets[idx] = {sorted_labels[idx]}

        reclustered_bins = {}
        for label in sorted_labels:
            if label != -1:
                for key, bin_set in bin_sets.items():
                    if label in bin_set:
                        try:
                            reclustered_bins[key + max_bin_id + 1] += copy.deepcopy(self.bins[label])
                        except KeyError:
                            reclustered_bins[key + max_bin_id + 1] = copy.deepcopy(self.bins[label])

        self.bins = reclustered_bins

    def reembed(self, tids, max_bin_id, plots,
                x_min=20, x_max=20, y_min=20, y_max=20, n=0,
                max_n_neighbours=50,
                default_min_validity=0.85,
                delete_unbinned=False,
                bin_unbinned=False,
                force=False,
                relaxed=False,
                reembed=False,
                skip_clustering=False,
                update_embeddings=False,
                truth_array=None,
                switch=None,
                debug=False):
        """
        Recluster -> Re-embedding -> Reclustering on the specified set of contigs
        Any clusters that look better than current cluster are kept and old cluster is thrown out
        Anything that doesn't get binned is thrown in the unbinned_tids list

        Params:
            :tids:
                List of contig target ids to be reclustered
            :max_bin:
                The current large bin key
            :plots:
                List of plots resulting from all previous embeddings. Gets turned into a gif
            :x_min, x_max, y_max, y_min:
                Parameters for the plot to keep all plots at the same aspect ratio
            :n:
                The current iteration
            :force:
                Whether to force the results even if they look bad. This is used when a cluster looks
                looks especially heinous or is too big. Use this param lightly, it can bust your bins for sure
            :reembed:
                Whether to use the re-embedding feature. Setting to false will skip out on UMAP. Can make things
                faster if original clustering is easy to disentangle, but setting to false can miss things

        ignore the other shit
        """
        if switch is None:
            switch = [0, 1, 2]
        remove = False
        noise = False
        precomputed = False  # Whether the precomputed clustering was the best result
        tids = list(np.sort(tids))
        contigs, log_lengths, tnfs = self.extract_contigs(tids)
        original_size = contigs['contigLen'].sum()
        min_validity = default_min_validity

        if original_size >= 14e6:
            force = True

        if len(set(tids)) > 1:
            if not skip_clustering:

                # Keep embeddings size consistent if they have been updated
                if truth_array is not None:
                    unbinned_array = \
                    self.large_contigs[~self.disconnected][~self.disconnected_intersected][truth_array]['tid'].isin(
                        tids)
                else:
                    unbinned_array = \
                        self.large_contigs[~self.disconnected][~self.disconnected_intersected]['tid'].isin(
                            tids)
                unbinned_embeddings = self.embeddings[unbinned_array]

                if reembed:
                    self.min_cluster_size = 2
                else:
                    self.min_cluster_size = 2
                try:
                    # + 1 because we don't want unlabelled
                    labels_single = self.iterative_clustering(unbinned_embeddings,
                                                              allow_single_cluster=True,
                                                              prediction_data=False,
                                                              double=False)
                    labels_multi = self.iterative_clustering(unbinned_embeddings,
                                                             allow_single_cluster=False,
                                                             prediction_data=False,
                                                             double=False)


                    # Try out precomputed method, validity metric does not work here
                    # so we just set it to 1 and hope it ain't shit. Method for this is
                    # not accept a clustering result with noise. Not great, but

                    if self.n_samples > 0:
                        n_samples = self.n_samples
                        sample_distances = self.short_sample_distance
                    else:
                        n_samples = self.long_samples
                        sample_distances = self.long_sample_distance

                    distances = metrics.distance_matrix(np.concatenate((contigs.iloc[:, 3:].values,
                                                                        log_lengths.values[:, None],
                                                                        tnfs.iloc[:, 2:].values), axis=1),
                                                        n_samples,
                                                        sample_distances)

                    distances = np.nan_to_num(distances)

                    labels_precomputed = self.iterative_clustering(distances, metric="precomputed")

                    validity_single = self.validity(labels_single, unbinned_embeddings)
                    validity_multi = self.validity(labels_multi, unbinned_embeddings)
                    validity_precom = self.validity(labels_precomputed, unbinned_embeddings)


                    # Calculate silhouette scores, will fail if only one label
                    # Silhouette scores don't work too well with HDBSCAN though since it
                    # usually requires pretty uniform clusters to generate a value of use
                    try:
                        silho_single = sk_metrics.silhouette_score(unbinned_embeddings, labels_single)
                    except ValueError:
                        silho_single = -1

                    try:
                        silho_multi = sk_metrics.silhouette_score(unbinned_embeddings, labels_multi)
                    except ValueError:
                        silho_multi = -1

                    try:
                        silho_precom = sk_metrics.silhouette_score(distances, labels_precomputed)
                    except ValueError:
                        silho_precom = -1

                    max_single = max(validity_single, silho_single)
                    max_multi = max(validity_multi, silho_multi)
                    max_precom = max(validity_precom, silho_precom)

                    if debug:
                        print('Allow single cluster validity: ', max_single)
                        print('Allow multi cluster validity: ', max_multi)
                        print('precom cluster validity: ', max_precom)

                    if max_single == -1 and max_multi == -1 and max_precom == -1:
                        self.labels = labels_single
                        max_validity = -1
                        min_validity = 1
                    elif max(max_single, max_multi, max_precom) == max_precom:
                        self.labels = labels_precomputed
                        max_validity = max_precom
                        precomputed = True

                    elif max(max_single, max_multi) == max_single:
                        self.labels = labels_single
                        max_validity = max_single

                    else:
                        self.labels = labels_multi
                        max_validity = max_multi

                    # get original size of bin

                    set_labels = set(self.labels)

                    if debug:
                        print("No. of Clusters:", len(set_labels), set_labels)

                except IndexError:
                    # Index error occurs when doing recluster after adding disconnected TNF
                    # contigs. Since the embedding array does not contain the missing contigs
                    # as such, new embeddings have to be calculated
                    self.labels = np.array([-1 for i in range(unbinned_embeddings.shape[0])])
                    max_validity = -1
            else:
                # Keep embeddings size consistent if they have been updated
                if truth_array is not None:
                    unbinned_array = \
                        self.large_contigs[~self.disconnected][~self.disconnected_intersected][truth_array]['tid'].isin(
                            tids)
                else:
                    unbinned_array = \
                        self.large_contigs[~self.disconnected][~self.disconnected_intersected]['tid'].isin(
                            tids)
                try:
                    unbinned_embeddings = self.embeddings[unbinned_array]
                    self.labels = np.array([-1 for i in range(unbinned_embeddings.shape[0])])
                    max_validity = -1
                except TypeError:
                    self.embeddings = np.zeros((self.large_contigs[~self.disconnected][~self.disconnected_intersected].shape[0], 2))
                    unbinned_embeddings = self.embeddings[unbinned_array]
                    self.labels = np.array([-1 for i in range(unbinned_embeddings.shape[0])])
                    max_validity = -1

            if -1 in self.labels and len(set(self.labels)) == 2:
                # fringe case where only single cluster formed with external
                # unbinned contigs on external part of bin.
                if max_validity > min_validity:
                    # lower binning quality and see if reembedding can do better
                    max_validity = min_validity

            if max_validity < 0.95 and reembed and len(tids) >= 5:
                # Generate new emebddings if clustering seems fractured
                # contigs, log_lengths, tnfs = self.extract_contigs(tids)
                try:
                    self.fit_transform(tids, max_n_neighbours)
                    # self.switch_intersector(switch=switch)
                    new_embeddings = self.intersection_mapper.embedding_
                    if update_embeddings:
                        self.embeddings = new_embeddings

                    labels_single = self.iterative_clustering(new_embeddings,
                                                              allow_single_cluster=True,
                                                              prediction_data=False,
                                                              double=skip_clustering)
                    labels_multi = self.iterative_clustering(new_embeddings,
                                                             allow_single_cluster=False,
                                                             prediction_data=False,
                                                             double=skip_clustering)

                    validity_single = self.validity(labels_single, new_embeddings)
                    validity_multi = self.validity(labels_multi, new_embeddings)

                    # Calculate silhouette scores, will fail if only one label
                    # Silhouette scores don't work too well with HDBSCAN though since it
                    # usually requires pretty uniform clusters to generate a value of use
                    try:
                        silho_single = sk_metrics.silhouette_score(unbinned_embeddings, labels_single)
                    except ValueError:
                        silho_single = -1

                    try:
                        silho_multi = sk_metrics.silhouette_score(unbinned_embeddings, labels_multi)
                    except ValueError:
                        silho_multi = -1

                    if debug:
                        print('Allow single cluster validity: ', validity_single)
                        print('Allow multi cluster validity: ', validity_multi)
                        print('Allow single cluster silho: ', silho_single)
                        print('Allow multi cluster silho: ', silho_multi)

                    max_single = max(validity_single, silho_single)
                    max_multi = max(validity_multi, silho_multi)

                    if max_single >= max_multi:
                        if max_validity <= max_single:
                            if all(label == -1 for label in labels_single):
                                if debug:
                                    print('using non re-embedded...')
                            else:
                                unbinned_embeddings = new_embeddings
                                self.labels = labels_single
                                max_validity = max_single
                                if precomputed:
                                    precomputed = False  # No longer the best results


                        else:
                            if debug:
                                print('using non re-embedded... %f' % max_validity)
                    else:
                        if max_validity <= max_multi:
                            if all(label == -1 for label in labels_multi):
                                logging.debug('using non re-embedded...')
                            else:
                                unbinned_embeddings = new_embeddings
                                self.labels = labels_multi
                                max_validity = max_multi
                                if precomputed:
                                    precomputed = False  # No longer the best results


                        else:
                            if debug:
                                print('using non re-embedded... %f' % max_validity)


                except TypeError:
                    self.labels = np.array([-1 for i in range(unbinned_embeddings.shape[0])])


            min_validity = min(min_validity, default_min_validity)
            max_validity = round(max_validity, 2)

            set_labels = set(self.labels)

            if debug:
                print("No. of Clusters:", len(set_labels), set_labels)
                print("Max validity: ", max_validity)

            plots = self.add_plot(plots, unbinned_embeddings, contigs, self.labels,
                                  n, x_min, x_max, y_min, y_max, max_validity, precomputed)

            if delete_unbinned:
                self.unbinned_tids = []

            if noise:
                # Clustering was a bit funky, so put back into unbinned and pull out again
                self.unbinned_tids = self.unbinned_tids + tids
                remove = True
            elif (len(set_labels) == 1) or (max_validity < min_validity) and not force:
                if debug:
                    print("Labels are bad")
                # Reclustering resulted in single cluster or all noise,
                # either case just use original bin
                remove = False

            else:
                new_bins = {}
                unbinned = []

                for (idx, label) in enumerate(self.labels):
                    if label != -1:
                        bin_key = max_bin_id + label.item() + 1
                        if isinstance(bin_key, np.int64):
                            bin_key = bin_key.item()
                        try:
                            new_bins[bin_key].append(tids[idx])  # inputs values as tid
                        except KeyError:
                            new_bins[bin_key] = [tids[idx]]
                    else:
                        unbinned.append(tids[idx])
                if debug:
                    print("No. of new bins:", new_bins.keys())
                    print("No. unbinned: ", len(unbinned))

                split = True
                if not force:  # How much of original bin was binned?
                    not_recovered = 0
                    new_bin_ids = []

                    for bin, new_tids in new_bins.items():
                        contigs, log_lengths, tnfs = self.extract_contigs(new_tids)
                        bin_size = contigs['contigLen'].sum()
                        new_bin_ids.append(bin)
                        logging.debug("Recovered enough: %d of %d" % (bin_size, original_size))

                    contigs, _, _ = self.extract_contigs(unbinned)
                    not_recovered += contigs['contigLen'].sum()

                    if not_recovered > original_size // 2 and not force:
                        logging.debug("Didn't recover enough: %d of %d, %.3f percent" %
                                      (not_recovered, original_size, not_recovered / original_size))
                        split = False
                        remove = False
                    elif ((len(new_bin_ids) < 2 and max_validity < 0.9)
                          or max_validity < min_validity) \
                            and not force:
                        split = False
                        remove = False
                if split:

                    # Half the original input has been binned if reembedding
                    for bin, new_tids in new_bins.items():
                        new_tids = list(np.sort(new_tids))
                        contigs, log_lengths, tnfs = self.extract_contigs(new_tids)
                        bin_size = contigs['contigLen'].sum()
                        if bin_size >= self.min_bin_size:
                            #  Keep this bin
                            if debug:
                                print("Removing original bin, keeping bin: ", bin)
                                print("Length: ", bin_size)
                            remove = True  # remove original bin
                            self.bins[bin] = new_tids
                            self.overclustered = True
                            if bin_size >= 14e6:
                                self.overclustered = True
                        else:
                            # put into unbinned
                            if debug:
                                print("Not adding new bin: ", bin, bin_size)
                            unbinned = unbinned + new_tids

                    if len(unbinned) != len(tids):
                        logging.debug("New bin(s) added... Total bins: ", len(self.bins.keys()))
                        contigs, log_lengths, tnfs = self.extract_contigs(unbinned)
                        bin_size = contigs['contigLen'].sum()
                        if self.n_samples > 0:
                            n_samples = self.n_samples
                            sample_distances = self.short_sample_distance
                        else:
                            n_samples = self.long_samples
                            sample_distances = self.long_sample_distance
                        try:
                            _, \
                            _, \
                            _, \
                            mean_agg, \
                            per_contig_avg = \
                                metrics.get_averages(np.concatenate((contigs.iloc[:, 3:].values,
                                                                     log_lengths.values[:, None],
                                                                     tnfs.iloc[:, 2:].values), axis=1),
                                                     n_samples,
                                                     sample_distances)
                        except ZeroDivisionError:
                            mean_agg = 0

                        bin_id = max(self.bins.keys()) + 1
                        if bin_size >= 2e5 and mean_agg <= 0.5:  # just treat it as a bin
                            if debug:
                                print("Unbinned contigs are bin: %d of size: %d" % (bin_id, bin_size))
                            self.bins[bin_id] = unbinned
                        else:
                            for contig in contigs.itertuples():
                                self.unbinned_tids.append(self.assembly[contig.contigName])

                    else:
                        remove = False
                        if debug:
                            print("No new bin added.")

                else:
                    remove = False
                    if debug:
                        print("No new bin added")
        else:
            remove = False
            if debug:
                print("No new bin added")


        return plots, remove



    def verify_contigs_are_in_best_bin(self, tids, current_bin, n_samples, sample_distances, debug=False):
        """
        Finds the best bin for a contig in a set of contigs
        based on distance metrics and returns a boolean

        :returns: boolean indicating if a contig has been removed or not
        """


        # Calculate the stats of current bin
        current_contig, current_log_lengths, current_tnfs = self.extract_contigs(tids)
        current_depths = np.concatenate((current_contig.iloc[:, 3:].values,
                                         current_log_lengths.values[:, None],
                                         current_tnfs.iloc[:, 2:].values), axis=1)


        for tid in tids:

            result = self.find_best_bin_for_contig(
                tid,
                current_bin,
                current_depths,
                n_samples,
                sample_distances,
                debug
            )

            if debug:
                print("Tid result: ", tid, result)
            if result is not None:
                self.bins[result].append(tid)
                self.bins[current_bin].remove(tid)
                return True # contigs were moved
            else:
                return False # contigs weren't moved


    def check_bad_bins_and_unbinned(self, min_bin_size=2e5, debug=False):
        n_samples, sample_distances = self.get_n_samples_and_distances()

        logging.debug("Checking bin internal distances...")
        a_contig_has_moved = False
        bins = self.bins.keys()
        bins_to_remove = []
        if debug:
            print(min_bin_size, " bin size minimum")
        for bin_id in bins:
            logging.debug("Beginning check on bin: ", bin_id)
            tids = self.bins[bin_id]
            if bin_id != 0 and len(tids) > 1:
                ## Dissolve very small or loq quality bins for re-emebedding
                contigs, log_lengths, tnfs = self.extract_contigs(tids)
                bin_size = contigs['contigLen'].sum()
                if debug:
                    print(bin_size, " current bin size")
                if min_bin_size is not None:

                    if bin_size <= min_bin_size:
                        if debug:
                            print(bin_size, " bin size")
                        if self.verify_contigs_are_in_best_bin(tids, bin_id, n_samples, sample_distances, debug):
                            a_contig_has_moved = True

                else:
                    if self.verify_contigs_are_in_best_bin(tids, bin_id, n_samples, sample_distances, debug):
                        a_contig_has_moved = True
            else:
                for tid in tids:
                    contigs, log_lengths, tnfs = self.extract_contigs([tid])
                    bin_size = contigs['contigLen'].sum()
                    if debug:
                        print("Checking unbinned contig ", tid, bin_size)
                    if min_bin_size is not None:
                        if bin_size <= min_bin_size:
                            if self.verify_contigs_are_in_best_bin(
                                    [tid], bin_id, n_samples, sample_distances, debug
                            ):
                                a_contig_has_moved = True
                    elif self.verify_contigs_are_in_best_bin(
                                [tid], bin_id, n_samples, sample_distances, debug
                        ):
                            a_contig_has_moved = True

            if len(tids) == 0:
                bins_to_remove.append(bin_id)

        for bin_id in bins_to_remove:
            try:
                _ = self.bins.pop(bin_id)
            except KeyError:
                pass

        return a_contig_has_moved

    def handle_new_embedding(self,
                             original_bin_id,
                             labels,
                             unbinned_embeddings,
                             max_validity,
                             min_validity,
                             plots,
                             n=0,
                             x_min=20,
                             x_max=20,
                             y_min=20,
                             y_max=20,
                             force=False,
                             precomputed=False,
                             debug=False):
        remove = False
        set_labels = set(labels)
        tids = self.bins[original_bin_id]
        try:
            max_bin_id = max(self.bins.keys()) + 1
        except ValueError:
            max_bin_id = 1

        if debug:
            print("No. of Clusters:", len(set_labels), set_labels)
            print("Max validity: ", max_validity)

        contigs, log_lengths, tnfs = self.extract_contigs(self.bins[original_bin_id])
        plots = self.add_plot(plots, unbinned_embeddings, contigs, labels,
                              n, x_min, x_max, y_min, y_max, max_validity, precomputed)

        original_size = contigs['contigLen'].sum()

        if (len(set_labels) == 1) or (max_validity < min_validity) and not force:
            if debug:
                print("Labels are bad")
            # Reclustering resulted in single cluster or all noise,
            # either case just use original bin
            remove = False

        else:
            new_bins = {}
            unbinned = []

            for (idx, label) in enumerate(labels):
                if label != -1:
                    bin_key = max_bin_id + label.item() + 1
                    if isinstance(bin_key, np.int64):
                        bin_key = bin_key.item()
                    try:
                        new_bins[bin_key].append(tids[idx])  # inputs values as tid
                    except KeyError:
                        new_bins[bin_key] = [tids[idx]]
                else:
                    unbinned.append(tids[idx])
            if debug:
                print("No. of new bins:", new_bins.keys())
                print("No. unbinned: ", len(unbinned))

            split = True
            if not force:  # How much of original bin was binned?
                not_recovered = 0
                new_bin_ids = []

                for bin, new_tids in new_bins.items():
                    contigs, log_lengths, tnfs = self.extract_contigs(new_tids)
                    bin_size = contigs['contigLen'].sum()
                    new_bin_ids.append(bin)
                    logging.debug("Recovered enough: %d of %d" % (bin_size, original_size))

                contigs, _, _ = self.extract_contigs(unbinned)
                not_recovered += contigs['contigLen'].sum()

                if not_recovered > original_size // 2 and not force:
                    logging.debug("Didn't recover enough: %d of %d, %.3f percent" %
                                  (not_recovered, original_size, not_recovered / original_size))
                    split = False
                    remove = False
                elif ((len(new_bin_ids) < 2 and max_validity < 0.9)
                      or max_validity < min_validity) \
                        and not force:
                    split = False
                    remove = False
            if split:

                # Half the original input has been binned if reembedding
                for bin, new_tids in new_bins.items():
                    new_tids = list(np.sort(new_tids))
                    contigs, log_lengths, tnfs = self.extract_contigs(new_tids)
                    bin_size = contigs['contigLen'].sum()
                    if bin_size >= self.min_bin_size:
                        #  Keep this bin
                        if debug:
                            print("Removing original bin, keeping bin: ", bin)
                            print("Length: ", bin_size)
                        remove = True  # remove original bin
                        self.bins[bin] = new_tids
                        self.overclustered = True
                        if bin_size >= 14e6:
                            self.overclustered = True
                    else:
                        # put into unbinned
                        if debug:
                            print("Not adding new bin: ", bin, bin_size)
                        unbinned = unbinned + new_tids

                if len(unbinned) != len(tids):
                    logging.debug("New bin(s) added... Total bins: ", len(self.bins.keys()))
                    contigs, log_lengths, tnfs = self.extract_contigs(unbinned)
                    bin_size = contigs['contigLen'].sum()
                    if self.n_samples > 0:
                        n_samples = self.n_samples
                        sample_distances = self.short_sample_distance
                    else:
                        n_samples = self.long_samples
                        sample_distances = self.long_sample_distance
                    try:
                        _, \
                        _, \
                        _, \
                        mean_agg, \
                        per_contig_avg = \
                            metrics.get_averages(np.concatenate((contigs.iloc[:, 3:].values,
                                                                 log_lengths.values[:, None],
                                                                 tnfs.iloc[:, 2:].values), axis=1),
                                                 n_samples,
                                                 sample_distances)
                    except ZeroDivisionError:
                        mean_agg = 0

                    bin_id = max(self.bins.keys()) + 1
                    if bin_size >= 2e5 and mean_agg <= 0.5:  # just treat it as a bin
                        if debug:
                            print("Unbinned contigs are bin: %d of size: %d" % (bin_id, bin_size))
                        self.bins[bin_id] = unbinned
                    else:
                        for contig in contigs.itertuples():
                            self.unbinned_tids.append(self.assembly[contig.contigName])

                else:
                    remove = False
                    if debug:
                        print("No new bin added.")

            else:
                remove = False
                if debug:
                    print("No new bin added")



        return plots, remove



def reembed_static(
            original_bin_id, tids, extraction,
            unbinned_embeddings,
            n_samples, sample_distances,
            a, b,
            max_n_neighbours=200,
            default_min_validity=0.85,
            reembed=False,
            force=False,
            skip_clustering=False,
            switch=None,
            debug=False,
            random_seed=42069
):
    """
    Recluster -> Re-embedding -> Reclustering on the specified set of contigs
    Any clusters that look better than current cluster are kept and old cluster is thrown out
    Anything that doesn't get binned is thrown in the unbinned_tids list

    Params:
        :tids:
            List of contig target ids to be reclustered
        :max_bin:
            The current large bin key
        :plots:
            List of plots resulting from all previous embeddings. Gets turned into a gif
        :x_min, x_max, y_max, y_min:
            Parameters for the plot to keep all plots at the same aspect ratio
        :n:
            The current iteration
        :force:
            Whether to force the results even if they look bad. This is used when a cluster looks
            looks especially heinous or is too big. Use this param lightly, it can bust your bins for sure
        :reembed:
            Whether to use the re-embedding feature. Setting to false will skip out on UMAP. Can make things
            faster if original clustering is easy to disentangle, but setting to false can miss things

    ignore the other shit
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    if switch is None:
        switch = [0, 1, 2]
    remove = False
    noise = False
    precomputed = False  # Whether the precomputed clustering was the best result
    tids = list(np.sort(tids))
    contigs, log_lengths, tnfs = extraction

    original_size = contigs['contigLen'].sum()
    min_validity = default_min_validity
    labels = None
    bins = {}
    max_validity = -1
    new_embeddings = None

    if original_size >= 14e6:
        force = True

    if len(set(tids)) > 1:
        if not skip_clustering:

            try:
                # + 1 because we don't want unlabelled
                labels_single = iterative_clustering_static(unbinned_embeddings,
                                                          allow_single_cluster=True,
                                                          double=False)
                labels_multi = iterative_clustering_static(unbinned_embeddings,
                                                         allow_single_cluster=False,
                                                         double=False)


                # Try out precomputed method, validity metric does not work here
                # so we just set it to 1 and hope it ain't shit. Method for this is
                # not accept a clustering result with noise. Not great, but

                distances = metrics.distance_matrix(np.concatenate((contigs.iloc[:, 3:].values,
                                                                    log_lengths.values[:, None],
                                                                    tnfs.iloc[:, 2:].values), axis=1),
                                                    n_samples,
                                                    sample_distances)

                distances = np.nan_to_num(distances)

                labels_precomputed = iterative_clustering_static(distances, metric="precomputed")

                validity_single = Clusterer.validity(labels_single, unbinned_embeddings)
                validity_multi = Clusterer.validity(labels_multi, unbinned_embeddings)
                validity_precom = Clusterer.validity(labels_precomputed, unbinned_embeddings)


                # Calculate silhouette scores, will fail if only one label
                # Silhouette scores don't work too well with HDBSCAN though since it
                # usually requires pretty uniform clusters to generate a value of use
                try:
                    silho_single = sk_metrics.silhouette_score(unbinned_embeddings, labels_single)
                except ValueError:
                    silho_single = -1

                try:
                    silho_multi = sk_metrics.silhouette_score(unbinned_embeddings, labels_multi)
                except ValueError:
                    silho_multi = -1

                try:
                    silho_precom = sk_metrics.silhouette_score(distances, labels_precomputed)
                except ValueError:
                    silho_precom = -1

                max_single = max(validity_single, silho_single)
                max_multi = max(validity_multi, silho_multi)
                max_precom = max(validity_precom, silho_precom)

                if debug:
                    print('Allow single cluster validity: ', max_single)
                    print('Allow multi cluster validity: ', max_multi)
                    print('precom cluster validity: ', max_precom)

                if max_single == -1 and max_multi == -1 and max_precom == -1:
                    labels = labels_single
                    max_validity = -1
                    min_validity = 1
                elif max(max_single, max_multi, max_precom) == max_precom:
                    labels = labels_precomputed
                    max_validity = max_precom
                    precomputed = True

                elif max(max_single, max_multi) == max_single:
                    labels = labels_single
                    max_validity = max_single

                else:
                    labels = labels_multi
                    max_validity = max_multi

                # get original size of bin

                set_labels = set(labels)

                if debug:
                    print("No. of Clusters:", len(set_labels), set_labels)

            except IndexError:
                # Index error occurs when doing recluster after adding disconnected TNF
                # contigs. Since the embedding array does not contain the missing contigs
                # as such, new embeddings have to be calculated
                labels = np.array([-1 for i in range(unbinned_embeddings.shape[0])])
                max_validity = -1
        else:

            labels = np.array([-1 for i in range(unbinned_embeddings.shape[0])])
            max_validity = -1


        if -1 in labels and len(set(labels)) == 2:
            # fringe case where only single cluster formed with external
            # unbinned contigs on external part of bin.
            if max_validity > min_validity:
                # lower binning quality and see if reembedding can do better
                max_validity = min_validity

        if max_validity < 0.95 and reembed and len(tids) >= 5:
            # Generate new emebddings if clustering seems fractured
            # contigs, log_lengths, tnfs = self.extract_contigs(tids)
            # try:
            tnf_reducer = umap.UMAP(
                metric=metrics.rho,
                n_neighbors=max_n_neighbours,
                n_components=2,
                disconnection_distance=2,
                min_dist=0,
                set_op_mix_ratio=1,
                a=a,
                b=b,
                init='spectral',
                random_state=42069
            )

            euc_reducer = umap.UMAP(
                metric=metrics.tnf_euclidean,
                n_neighbors=max_n_neighbours,
                n_components=2,
                disconnection_distance=10,
                min_dist=0,
                set_op_mix_ratio=1,
                a=a,
                b=b,
                init='spectral',
                random_state=42069
            )

            depth_reducer = umap.UMAP(
                metric=metrics.aggregate_tnf,
                metric_kwds={"n_samples": n_samples,
                             "sample_distances": sample_distances},
                n_neighbors=max_n_neighbours,
                disconnection_distance=2,
                n_components=2,
                min_dist=0,
                set_op_mix_ratio=1,
                a=a,
                b=b,
                init='spectral',
                random_state=42069
            )
            for a in [1.5, 1.9]:
                multi_transform_static(
                    contigs, log_lengths, tnfs,
                    depth_reducer, tnf_reducer, euc_reducer,
                    tids, max_n_neighbours, a=a, switch=switch
                )
                intersection_mapper = switch_intersector_static(
                    depth_reducer,
                    tnf_reducer,
                    euc_reducer,
                    switch=switch
                )
                new_embeddings = intersection_mapper.embedding_

                labels_single = iterative_clustering_static(new_embeddings,
                                                          allow_single_cluster=True,
                                                          double=skip_clustering)
                labels_multi = iterative_clustering_static(new_embeddings,
                                                         allow_single_cluster=False,
                                                         double=skip_clustering)

                validity_single = Clusterer.validity(labels_single, new_embeddings)
                validity_multi = Clusterer.validity(labels_multi, new_embeddings)

                # Calculate silhouette scores, will fail if only one label
                # Silhouette scores don't work too well with HDBSCAN though since it
                # usually requires pretty uniform clusters to generate a value of use
                try:
                    silho_single = sk_metrics.silhouette_score(unbinned_embeddings, labels_single)
                except ValueError:
                    silho_single = -1

                try:
                    silho_multi = sk_metrics.silhouette_score(unbinned_embeddings, labels_multi)
                except ValueError:
                    silho_multi = -1

                if debug:
                    print('Allow single cluster validity: ', validity_single)
                    print('Allow multi cluster validity: ', validity_multi)
                    print('Allow single cluster silho: ', silho_single)
                    print('Allow multi cluster silho: ', silho_multi)

                max_single = max(validity_single, silho_single)
                max_multi = max(validity_multi, silho_multi)

                if max_single >= max_multi:
                    if max_validity <= max_single:
                        if all(label == -1 for label in labels_single):
                            if debug:
                                print('using non re-embedded...')
                        else:
                            unbinned_embeddings = new_embeddings
                            labels = labels_single
                            max_validity = max_single
                            if precomputed:
                                precomputed = False  # No longer the best results


                    else:
                        if debug:
                            print('using non re-embedded... %f' % max_validity)
                else:
                    if max_validity <= max_multi:
                        if all(label == -1 for label in labels_multi):
                            logging.debug('using non re-embedded...')
                        else:
                            unbinned_embeddings = new_embeddings
                            labels = labels_multi
                            max_validity = max_multi
                            if precomputed:
                                precomputed = False  # No longer the best results


                    else:
                        if debug:
                            print('using non re-embedded... %f' % max_validity)


            # except TypeError:
            #     # labels = np.array([-1 for i in range(unbinned_embeddings.shape[0])])
            #     pass


        min_validity = min(min_validity, default_min_validity)
        max_validity = round(max_validity, 2)

        set_labels = set(labels)

        if debug:
            print("No. of Clusters:", len(set_labels), set_labels)
            print("Max validity: ", max_validity)

    if new_embeddings is None:
        new_embeddings = unbinned_embeddings

    return original_bin_id, labels, new_embeddings, max_validity, min_validity, force, precomputed

