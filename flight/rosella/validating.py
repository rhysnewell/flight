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
import umap
import seaborn as sns
import matplotlib
import multiprocessing
import pebble
from concurrent.futures import TimeoutError
import scipy.spatial.distance as sp_distance

# self imports
import flight.metrics as metrics
import flight.distance as distance
from flight.rosella.clustering import Clusterer, iterative_clustering_static, kmeans_cluster
from flight.rosella.embedding import Embedder
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

###############################################################################
################################ - Classes - ##################################

class Validator(Clusterer, Embedder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def validate_bins(self, plots, n, x_min, x_max, y_min, y_max,
                      bin_unbinned=False, reembed=False,
                      size_only=False, big_only=False,
                      quick_filter=False, size_filter=False, debug=False,
                      large_bins_only=False):
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
        big_reembed_separately = []  # container for bin ids that look like chimeras
        big_force_new_clustering = []
        big_lower_thresholds = []
        big_reembed_if_no_cluster = []
        big_switches = []
        bins = self.bins.keys()
        for bin in bins:
            logging.debug("Beginning check on bin: %d " % bin)
            tids = self.bins[bin]
            if len(tids) == 1:
                continue
            elif bin == 0:
                continue
            if big_only:
                removed_single = []  # Single contig bin
                contigs, log_lengths, tnfs = self.extract_contigs(tids)

                bin_size = contigs['contigLen'].sum()
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
                                    metrics.get_averages(
                                        np.concatenate((contigs.iloc[:, 3:].values,
                                                        log_lengths.values[:, None],
                                                        tnfs.iloc[2:].values), axis=1),
                                         n_samples,
                                         sample_distances)

                            per_contig_avg = np.array(per_contig_avg)

                            removed = []

                            # check for any obviously misplaced contigs
                            for (tid, avgs) in zip(tids, per_contig_avg):
                                if avgs[0] >= 0.75 or avgs[1] >= 0.3 or avgs[2] >= 7 or avgs[3] >= 0.7:
                                    removed.append(tid)
                            remove = False

                            if len(removed) > 0 and len(removed) != len(tids):
                                [(tids.remove(r), self.unbinned_tids.append(r)) for r in removed]
                                current_contigs, current_lengths, current_tnfs = self.extract_contigs(tids)

                                if current_contigs['contigLen'].sum() <= self.min_bin_size:
                                    [self.unbinned_tids.append(tid) for tid in tids]
                                    remove = True

                            if len(tids) == 0 or remove:
                                bins_to_remove.append(bin)
                        except (ZeroDivisionError, ValueError) as e:
                            # Only one contig left, break out
                            continue
                elif bin_size >= 3e6 and len(tids) >= 2:

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
                    r_level = 0.15
                    e_level = 4

                    if ((round(mean_md, 2) >= md_filt
                         or round(mean_agg, 2) >= agg_filt
                         or round(mean_tnf, 2) >= r_level)
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
                                         or max_values[2] >= euc_filt) or \
                                        (max_values[3] >= 0.1 or max_values[0] >= 0.1) and \
                                        (max_values[1] >= 0.2
                                         or max_values[2] >= e_level) \
                                        or bin_size >= 15e6:
                                    if debug:
                                        print("Removing contig: ", max_idx, tids[max_idx])
                                    removed_single.append(tids[max_idx])

                    if len(removed_single) > 0:
                        [(big_tids.append(r), tids.remove(r)) for r in removed_single]
            elif reembed:

                contigs, log_lengths, tnfs = self.extract_contigs(tids)
                bin_size = contigs['contigLen'].sum()

                if large_bins_only and bin_size <= 2e6:
                    self.survived.append(bin)
                    continue

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

                    except ZeroDivisionError:
                        continue

                    if debug:
                        print('before check for distant contigs: ', len(tids))
                        _, _, _, _ = self.bin_stats(bin)

                    f_level = 0.3
                    m_level = 0.2
                    r_level = 0.15
                    e_level = 5
                    per_contig_avg = np.array(per_contig_avg)

                    # detect if a large portion of contigs seem out of place
                    misplaced_contigs = False

                    if contigs['contigLen'][np.any(per_contig_avg[:, [0, 3]] > 0.35, axis = 1)].sum() >= 1e6:
                        misplaced_contigs = True

                    # Always check bins with bad bin stats or if they are large, just for sanity check
                    if (bin_size >= 10e6 or misplaced_contigs
                            or mean_agg >= f_level or mean_md >= m_level
                            or mean_euc >= e_level or mean_tnf >= r_level):
                        logging.debug(bin, mean_md, mean_tnf, mean_agg, len(tids))

                        if debug:
                            print("Reclustering bin %d" % bin)
                        if bin_size >= 12e6 or contigs['contigLen'][np.any(per_contig_avg[:, [0, 3]] > 0.5, axis = 1)].sum() >= 1e6:
                            factor = min(max(mean_md, mean_agg) * 2, 1.0)
                        elif misplaced_contigs or bin_size >= 10e6:
                            factor = per_contig_avg[:, [0, 3]].max()
                        else:
                            factor = max(mean_md, mean_agg)
                        reembed_separately.append(bin)
                        lower_thresholds.append(1 - factor)
                        force_new_clustering.append(False)  # send it to regular hell
                        reembed_if_no_cluster.append(True)
                        switches.append([0, 1, 2])
                    else:
                        factor = 1 - max((mean_md + mean_tnf + mean_agg) / 3, 0.05)
                        if factor < 0.95:
                            reembed_separately.append(bin)
                            force_new_clustering.append(False)  # send it to regular hell
                            reembed_if_no_cluster.append(True)
                            lower_thresholds.append(factor)
                            switches.append([0, 1, 2])
                        else:
                            self.survived.append(bin)


                else:
                    logging.debug(bin, self.survived)

        try:
            max_bin_id = max(self.bins.keys()) + 1
        except ValueError:
            max_bin_id = 1
        for k, v in new_bins.items():
            self.bins[max_bin_id + k] = list(np.sort(np.array(v)))

        if reembed:
            if self.n_samples > 0:
                n_samples = self.n_samples
                sample_distances = self.short_sample_distance
            else:
                n_samples = self.long_samples
                sample_distances = self.long_sample_distance

            numpy_thread_limit = max(self.threads // 5, 1)
            if numpy_thread_limit == 1:
                worker_limit = self.threads
            else:
                worker_limit = max(self.threads // numpy_thread_limit, 1)

            with pebble.ProcessPool(max_workers=min(self.threads, 5), context=multiprocessing.get_context('forkserver')) as executor:
                futures = [
                    executor.schedule(
                        reembed_static,
                            (
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
                                self.n_components,
                                min_validity,
                                reembed_cluster,
                                force_new,
                                False,
                                switch,
                                False,
                                numpy_thread_limit
                            ),
                            timeout = self.max_time_to_recluster_bin * 5
                    )
                    for (bin, force_new, min_validity, reembed_cluster, switch) in zip(
                        reembed_separately,
                        force_new_clustering,
                        lower_thresholds,
                        reembed_if_no_cluster,
                        switches
                    )]

                for future in futures:
                    try:
                        result = future.result()
                        # result = future
                        plots, remove = self.handle_new_embedding(
                            result[0], result[1], result[2], result[3], result[4],
                            plots, n, x_min, x_max, y_min, y_max, result[5], result[6], debug=False
                        )
                        logging.debug("Problem bin result... removing: %d %d" % (remove, result[0]))

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
                            # self.survived.append(result[0])
                    except TimeoutError:
                        # future.cancel()
                        break
        
        # find any bins that potentially timedout
        [self.survived.append(bin) for bin in reembed_separately if bin not in bins_to_remove]

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


    def handle_new_embedding(
        self,
        original_bin_id,
        labels,
        unbinned_embeddings,
        max_validity,
        min_validity,
        plots=None,
        n=0,
        x_min=20,
        x_max=20,
        y_min=20,
        y_max=20,
        force=False,
        precomputed=False,
        debug=False
    ):
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
        if plots is not None:
            plots = self.add_plot(plots, unbinned_embeddings, contigs, labels,
                                  n, x_min, x_max, y_min, y_max, max_validity, precomputed, min_validity)

        original_size = contigs['contigLen'].sum()

        if (len(set_labels) == 1) or \
                (max_validity < min_validity and not max_validity <= round(min_validity, 2)) and \
                (not force and min_validity != 0):
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
            if not force and min_validity != 0:  # How much of original bin was binned?
                not_recovered = 0
                new_bin_ids = []

                for bin, new_tids in new_bins.items():
                    contigs, log_lengths, tnfs = self.extract_contigs(new_tids)
                    bin_size = contigs['contigLen'].sum()
                    new_bin_ids.append(bin)
                    logging.debug("Recovered enough: %d of %d" % (bin_size, original_size))

                contigs, _, _ = self.extract_contigs(unbinned)
                not_recovered += contigs['contigLen'].sum()

                if not_recovered > (original_size * 0.6) and not force:
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
    n_components=2,
    default_min_validity=0.85,
    reembed=False,
    force=False,
    skip_clustering=False,
    switch=None,
    debug=False,
    random_seed=42069,
    threads=10
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

    import warnings
    from numba import set_num_threads
    import threadpoolctl
    set_num_threads(threads)

    noise = False
    precomputed = False  # Whether the precomputed clustering was the best result
    tids = list(np.sort(tids))
    contigs, log_lengths, tnfs = extraction

    original_size = contigs['contigLen'].sum()
    min_validity = default_min_validity
    labels = []
    max_validity = -1
    new_embeddings = None

    with warnings.catch_warnings() and threadpoolctl.threadpool_limits(limits=threads, user_api="blas"):
        warnings.simplefilter("ignore")
        if original_size >= 14e6:
            force = True

        if len(set(tids)) >= 3:

            labels_multi = Clusterer.get_cluster_labels_array(
                unbinned_embeddings,
                top_n=3,
                metric="euclidean",
                selection_method = "eom",
                solver="hbgf",
                threads=threads,
                use_multiple_processes=False
            )[-1]

            # Try out precomputed method, validity metric does not work here
            # so we just set it to 1 and hope it ain't shit. Method for this is
            # not accept a clustering result with noise. Not great, but
            try:
                de = distance.ProfileDistanceEngine()
                stat = de.makeRankStat(
                    contigs.iloc[:, 3:].values,
                    tnfs.iloc[:, 2:].values,
                    log_lengths.values,
                    silent=True,
                    fun=lambda a: a / max(a),
                    use_multiple_processes=False
                )

                distances = np.nan_to_num(sp_distance.squareform(stat))

                labels_kmeans_embeddings, kmeans_score_embeddings = get_best_kmeans_result(unbinned_embeddings, 5, random_seed)
                labels_kmeans_precom, kmeans_score_precom = get_best_kmeans_result(distances, 5, random_seed)
                validity_multi = Clusterer.validity(labels_multi, unbinned_embeddings)

                max_multi = validity_multi
                max_precom = kmeans_score_precom
                max_kmeans = kmeans_score_embeddings

                if debug:
                    print('Allow multi cluster validity: ', max_multi)
                    print('precom cluster validity: ', max_precom)

                if max_multi == -1 and max_precom == -1 and max_kmeans == -1:
                    labels = max_precom
                    max_validity = -1
                elif max(max_multi, max_precom, max_kmeans) == max_kmeans:
                    labels = labels_kmeans_embeddings
                    max_validity = kmeans_score_embeddings
                elif max(max_multi, max_precom) == max_precom:
                    labels = labels_kmeans_precom
                    max_validity = max_precom
                    precomputed = True
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


            if -1 in labels and len(set(labels)) == 2:
                # fringe case where only single cluster formed with external
                # unbinned contigs on external part of bin.
                if max_validity > min_validity:
                    # lower binning quality and see if reembedding can do better
                    max_validity = min_validity

            if (max_validity < 0.95 or min_validity <= 0.5) and reembed and len(tids) >= 5:
                # Generate new emebddings if clustering seems fractured
                # contigs, log_lengths, tnfs = self.extract_contigs(tids)
                # try:
                precomputed_reducer_low = umap.UMAP(
                    metric="precomputed",
                    n_neighbors=max_n_neighbours,
                    n_components=n_components,
                    a=1.48,
                    b=0.4,
                    n_jobs=threads,
                )

                precomputed_reducer_mid = umap.UMAP(
                    metric="precomputed",
                    n_neighbors=max_n_neighbours,
                    n_components=n_components,
                    a=1.9,
                    b=0.4,
                    n_jobs=threads,
                )

                # precomputed_reducer_high = umap.UMAP(
                #     metric="precomputed",
                #     n_neighbors=max_n_neighbours,
                #     n_components=n_components,
                #     a=1.68,
                #     b=0.5,
                #     n_jobs=threads,
                #     random_state=random_seed
                # )

                try:
                    new_embeddings_1 = precomputed_reducer_low.fit_transform(sp_distance.squareform(stat))
                    new_embeddings_2 = precomputed_reducer_mid.fit_transform(sp_distance.squareform(stat))
                    # new_embeddings_3 = precomputed_reducer_high.fit_transform(sp_distance.squareform(stat))

                    labels_multi, validity_multi = Clusterer.ensemble_cluster_multiple_embeddings(
                            [
                                new_embeddings_1,
                                new_embeddings_2,
                                # new_embeddings_3
                            ],
                            top_n=3,
                            metric="euclidean",
                            cluster_selection_methods="eom",
                            solver="hbgf",
                            threads=threads,
                            use_multiple_processes=False
                        )
                    labels_multi = labels_multi[-1]
                    validity_multi = validity_multi[-1]

                    new_embeddings = new_embeddings_1

                    # validity_multi = Clusterer.validity(labels_multi, new_embeddings)

                    # if validity_multi >= validity_leaf:
                    best_labels = labels_multi
                    best_val = validity_multi


                    if max_validity <= best_val or (min_validity < 0.1 and max_validity <= 0.5):
                        if all(label == -1 for label in best_labels):
                            logging.debug('using non re-embedded...')
                        else:
                            unbinned_embeddings = new_embeddings
                            labels = best_labels
                            max_validity = best_val
                            if precomputed:
                                precomputed = False  # No longer the best results

                    else:
                        if debug:
                            print('using non re-embedded... %f' % max_validity)
                except (FloatingPointError, TypeError):
                    if max_multi == -1 and max_precom == -1 and max_kmeans == -1:
                        labels = max_precom
                        max_validity = -1
                    elif max(max_multi, max_precom, max_kmeans) == max_kmeans:
                        labels = labels_kmeans_embeddings
                        max_validity = kmeans_score_embeddings
                    elif max(max_multi, max_precom) == max_precom:
                        labels = labels_kmeans_precom
                        max_validity = max_precom
                        precomputed = True
                    else:
                        labels = labels_multi
                        max_validity = max_multi

                if unbinned_embeddings.all() == 0 or \
                        (max_validity <= 0 and max_validity != -1 and min_validity != 0) or \
                        (-1 in labels and len(set(labels)) == 2) or \
                        (max_validity < min_validity and min_validity <= 0.6):
                    # embedding failed, which only happens for smaller bins
                    # check to see if kmeans clustering splits bins sensibly
                    try:
                        labels, score = get_best_kmeans_result(distances, 5, random_seed)
                    except ValueError:
                        distances = metrics.distance_matrix(np.concatenate((contigs.iloc[:, 3:].values,
                                                                            log_lengths.values[:, None],
                                                                            tnfs.iloc[:, 2:].values), axis=1),
                                                            n_samples,
                                                            sample_distances)

                        distances = np.nan_to_num(distances)
                        labels, score = get_best_kmeans_result(distances, 5, random_seed)

                    max_validity = score
                    # min_validity = 0.5
                    precomputed = True
                    noise = True


            if noise:
                min_validity = 0.5

            min_validity = min(min_validity, default_min_validity)
            max_validity = round(max_validity, 2)

            set_labels = set(labels)

            if debug:
                print("No. of Clusters:", len(set_labels), set_labels)
                print("Max validity: ", max_validity)

        if new_embeddings is None:
            new_embeddings = unbinned_embeddings

    if len(labels) == 0:
        labels = np.array([-1 for _ in range(len(tids))])
        max_validity = -1
        new_embeddings = unbinned_embeddings

    return original_bin_id, labels, new_embeddings, max_validity, min_validity, force, precomputed


def get_best_kmeans_result(distances, max_n_clusters=10, random_seed=42069, n_jobs=1):
    max_score = 0
    best_labels = None
    for n in range(2, max_n_clusters + 1):
        try:
            labels, score = kmeans_cluster(distances, 2, random_seed)
            if score > max_score:
                best_labels = labels
                max_score = score
        except ValueError:
            continue

    if best_labels is None:
        best_labels = np.array([0 for _ in range(0, distances.shape[0])])

    return best_labels, max_score