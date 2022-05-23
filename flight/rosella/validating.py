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

# Function imports
import numpy as np
import umap
import seaborn as sns
import matplotlib
import multiprocessing
import pebble
from concurrent.futures import TimeoutError
import scipy.spatial.distance as sp_distance
import threadpoolctl
import warnings
import signal
import random

# self imports
import flight.metrics as metrics
import flight.distance as distance
from flight.rosella.clustering import Clusterer, iterative_clustering_static, kmeans_cluster
from flight.rosella.embedding import Embedder
# Set plotting style
sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
matplotlib.use('pdf')

###############################################################################
############################### - Exceptions - ################################
"""
Function to handle signal IOErrors after missing input
"""
def handler(signum, frame):
     raise TimeoutError
signal.signal(signal.SIGALRM, handler)

class BadTreeFileException(Exception):
    pass

###############################################################################
################################ - Classes - ##################################

class Validator(Clusterer, Embedder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def average_bin_stats(self, min_bin_size_to_check=1e6):
        mean_of_means = [0, 0, 0, 0]
        n_usable_bins = 0
        for bin_id in self.bins:
            tids = self.bins[bin_id]
            tids = set(tids)
            if len(tids) == 1:
                continue
            elif bin_id == 0:
                continue

            contigs, log_lengths, tnfs = self.extract_contigs(tids)
            bin_size = contigs['contigLen'].sum()

            if bin_size > min_bin_size_to_check:
                mean_md, \
                mean_tnf, \
                mean_euc, \
                mean_agg, \
                _ = \
                    metrics.get_averages(np.concatenate((contigs.iloc[:, 3:].values,
                                                         log_lengths.values[:, None],
                                                         tnfs.iloc[:, 2:].values), axis=1),
                                         self.n_samples,
                                         self.short_sample_distance)
                mean_of_means[0] += mean_md
                mean_of_means[1] += mean_tnf
                mean_of_means[2] += mean_euc
                mean_of_means[3] += mean_agg
                n_usable_bins += 1

        try:
            mean_of_means = [mean_of_means[i] / n_usable_bins for i in range(4)]
        except ZeroDivisionError:
            return mean_of_means

        return mean_of_means

    def prune_bin(
            self,
            bin_id,
            tids,
            contigs,
            log_lengths,
            tnfs,
            n_samples = None,
            sample_distances = None,
            max_contig_size = 2e5,
            bins_to_remove = None,
            something_changed = None,
            min_completeness = 50,
            max_contamination = 10,
            bin_completeness = None,
            bin_contamination = None,
            debug = False,
    ):
        """
        If a bin isn't reclustering even though it looks like it should be, then it likely has
        some noisy contigs present. These are contigs that are close enough to the actual bins contigs
        that they cluster together, but separate enough that they ruin any attempt to refine the bin.
        It's risky to just remove contigs ad hoc like this, so we limit the size of the contigs that can be
        removed.
        :params:
        :bin_id: - the target bin identifier
        :tids: - the tids present within the bin
        :contigs, log_lengths, tnfs: - the contigs dataframe extracted using extract_contigs
        :n_samples: - number of samples
        :sample_distances: - coverage distances between samples
        :max_contig_size: - The maximum contig size to be considered for pruning. Try to limit this to
                            smaller contigs to avoid completely destroying large contig bins.
        :bins_to_remove: - if the bin gets completely removed by this function add the bin id here
        :something_changed: - if something was removed from the bin add it here
        """
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


        removed = []

        std_md = per_contig_avg[:, 0].std()
        std_tnf = per_contig_avg[:, 1].std()
        std_euc = per_contig_avg[:, 2].std()
        std_agg = per_contig_avg[:, 3].std()

        a_level = min(max(0.35, mean_agg * 2), mean_agg + std_agg * 3.0)
        m_level = min(max(0.3, mean_md * 2), mean_md + std_md * 3.0)
        r_level = min(max(0.15, mean_tnf * 2), mean_tnf + std_tnf * 3.0)
        e_level = min(max(6, mean_euc * 2), mean_euc + std_euc * 3.0)

        if debug:
            print(f"Pruning {bin_id} {a_level} {m_level} {r_level} {e_level}")

        counter = 0
        while counter <= 20:
            counter += 1
            # check for any obviously misplaced contigs
            for (tid, avgs, log_length) in zip(tids, per_contig_avg, log_lengths):
                if log_length <= max_contig_size and (avgs[0] >= m_level or avgs[1] >= r_level or avgs[2] >= e_level):
                    removed.append(tid)
            remove = False

            if len(removed) > 0 and len(removed) != len(tids):
                [(tids.remove(r), self.unbinned_tids.append(r)) for r in removed]
                current_contigs, current_lengths, current_tnfs = self.extract_contigs(tids)

                if current_contigs['contigLen'].sum() <= self.min_bin_size:
                    [self.unbinned_tids.append(tid) for tid in tids]
                    remove = True

            if len(tids) == 0 or remove:
                bins_to_remove.append(bin_id)
                break
            elif len(removed) > 0:
                something_changed.append(bin_id)
                break
            elif bin_contamination >= max_contamination:
                # forcibly lower the thresholds slightly and try again
                m_level *= 0.9
                r_level *= 0.9
                e_level *= 0.9
                if debug:
                    print(f"Reducing {bin_id} {a_level} {m_level} {r_level} {e_level}")
                continue
            else:
                break


    def retrieve_bin_checkm_stats(self, bin_id):
        bin_completeness, bin_contamination = 100, 0
        if self.input_bin_stats is not None:
            bin_in_input_stats = self.input_bin_stats["bin_index"] == bin_id
            if np.any(bin_in_input_stats):
                bin_values = self.input_bin_stats[bin_in_input_stats]
                bin_completeness = bin_values["completeness"].values[0]
                bin_contamination = bin_values["contamination"].values[0]

        return bin_completeness, bin_contamination

    def validate_bins(
            self,
            plots, n, x_min, x_max, y_min, y_max,
            bin_unbinned=False,
            reembed=False,
            big_only=False,
            quick_filter=False,
            debug=False,
            large_bins_only=False,
            min_completeness=50,
            max_contamination=10,
            min_bin_size=1e6,
            contaminated_only=False,
            refining_mode=False,
            max_bin_size=2e7
    ):
        """
        Function for deciding whether a bin needs to be reembedded or split up
        Uses internal bin statistics, mainly mean ADP and Rho values
        """

        n_samples, sample_distances = self.get_n_samples_and_distances()

        bins_to_remove = []
        something_changed = []
        new_bins = {}
        big_tids = []
        reembed_separately = []  # container for bin ids that look like chimeras
        force_new_clustering = []
        lower_thresholds = []
        reembed_if_no_cluster = []
        switches = []

        md_thresh, tnf_thresh, euc_thresh, agg_thresh = self.average_bin_stats(min_bin_size)

        bins = self.bins.keys()
        for bin_id in bins:
            bin_completeness, bin_contamination = self.retrieve_bin_checkm_stats(bin_id)
            if debug:
                print(f"{bin_id} {bin_completeness} {bin_contamination}")
            # first guard check for contaminated only flag
            if n == 0 and contaminated_only:
                if bin_contamination <= max_contamination or bin_completeness <= min_completeness:
                    # remove this bin from further analysis
                    bins_to_remove.append(bin_id)
                    continue

            tids = self.bins[bin_id]
            if len(tids) == 1 or bin_id == 0:
                continue

            contigs, log_lengths, tnfs = self.extract_contigs(tids)

            bin_size = contigs['contigLen'].sum()

            if quick_filter:
                ## Remove stuff that is obviously wrong
                bin_size = contigs['contigLen'].sum()

                if bin_size < self.min_bin_size:
                    self.unbinned_tids = self.unbinned_tids + tids
                    bins_to_remove.append(bin_id)
                else:
                    try:
                        self.prune_bin(
                            bin_id,
                            tids,
                            contigs,
                            log_lengths,
                            tnfs,
                            n_samples,
                            sample_distances,
                            1e4,
                            bins_to_remove,
                            something_changed,
                            min_completeness,
                            max_contamination,
                            bin_completeness,
                            bin_contamination
                        )
                    except (ZeroDivisionError) as e:
                        # Only one contig left, break out
                        continue
            elif big_only:
                removed_single = []  # Single contig bin

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

                        std_md = per_contig_avg[:, 0].std()
                        std_tnf = per_contig_avg[:, 1].std()
                        std_euc = per_contig_avg[:, 2].std()
                        std_agg = per_contig_avg[:, 3].std()
                    except ZeroDivisionError:
                        # Only one contig left, break out
                        break


                    if len(tids) == 2:
                        # Lower thresholds for fewer contigs
                        agg_filt = max(0.35, agg_thresh * 1.25)
                        md_filt = max(0.2, md_thresh * 1.25)
                        rho_filt = max(0.1, tnf_thresh * 1.25)
                        euc_filt = max(4, euc_thresh * 1.25)
                        # Two contigs by themselves that are relatively distant. Remove them separately
                    else:
                        agg_filt = min(max(0.35, agg_thresh * 1.5), mean_agg + std_agg * 2.0)
                        md_filt = min(max(0.3, md_thresh * 1.25), mean_md + std_md * 2.0)
                        rho_filt = min(max(0.15, tnf_thresh * 1.5), mean_tnf + std_tnf * 2.0)
                        euc_filt = min(max(6, euc_thresh * 1.25), mean_euc + std_euc * 2.0)

                    r_level = 0.15
                    e_level = 4

                    # detect if a large portion of contigs seem out of place
                    misplaced_contigs = False

                    misplaced_array = np.array(per_contig_avg[:, 0] >= md_filt) \
                                      + np.array(per_contig_avg[:, 1] >= rho_filt) \
                                      + np.array(per_contig_avg[:, 2] >= euc_filt)

                    if contigs['contigLen'][misplaced_array].sum() >= 1.0e6:
                        misplaced_contigs = True

                    if misplaced_contigs or bin_size >= 15e6:
                        if debug:
                            print("Checking big contigs for bin: ", bin_id)
                        for max_idx in range(per_contig_avg.shape[0]):
                            max_values = per_contig_avg[max_idx, :]
                            contig_length = contigs['contigLen'].iloc[max_idx]
                            if debug:
                                print("Contig size and tid: ", contig_length, tids[max_idx])

                            if contig_length < 2e6: continue

                            if debug:
                                print("Found large contig: ", max_idx, tids[max_idx])
                            if (max_values[3] >= agg_filt or max_values[0] >= md_filt) or \
                                    (max_values[1] >= rho_filt
                                     or max_values[2] >= euc_filt):
                                if debug:
                                    print("Removing contig: ", max_idx, tids[max_idx])
                                removed_single.append(tids[max_idx])

                    if len(removed_single) > 0:
                        [(big_tids.append(r), tids.remove(r)) for r in removed_single]

            elif reembed and bin_id not in self.survived:

                if large_bins_only and bin_size <= 2e6:
                    self.survived.append(bin_id)
                    continue

                if debug:
                    print(bin_id, bin_size)

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

                    std_md = per_contig_avg[:, 0].std()
                    std_tnf = per_contig_avg[:, 1].std()
                    std_euc = per_contig_avg[:, 2].std()
                    std_agg = per_contig_avg[:, 3].std()

                except ZeroDivisionError:
                    continue

                if debug:
                    print(f'before check for distant contigs: {len(tids)}')
                    _, _, _, _ = self.bin_stats(bin_id)

                # a_level = max(0.35, agg_thresh * 1.5)
                # m_level = max(0.3, md_thresh * 1.25)
                # r_level = max(0.15, tnf_thresh * 1.5)
                # e_level = max(6, euc_thresh * 1.25)

                a_level = min(max(0.35, agg_thresh * 1.5), mean_agg + std_agg * 1.5)
                m_level = min(max(0.3, md_thresh * 1.25), mean_md + std_md * 1.5)
                r_level = min(max(0.15, tnf_thresh * 1.5), mean_tnf + std_tnf * 1.5)
                e_level = min(max(6, euc_thresh * 1.25), mean_euc + std_euc * 1.5)

                a_upper = mean_agg + std_agg
                m_upper = mean_md + std_md
                r_upper = mean_tnf + std_tnf
                e_upper = mean_euc + std_euc

                a_lower = mean_agg - std_agg
                m_lower = mean_md - std_md
                r_lower = mean_tnf - std_tnf
                e_lower = mean_euc - std_euc

                # detect if a large portion of contigs seem out of place
                slightly_misplaced_contigs = False
                misplaced_contigs = False
                very_misplaced_contigs = False

                slightly_misplaced_array = np.array(per_contig_avg[:, 0] <= m_lower) \
                                           + np.array(per_contig_avg[:, 1] <= r_lower) \
                                           + np.array(per_contig_avg[:, 2] <= e_lower) \
                                           + np.array(per_contig_avg[:, 0] >= m_upper) \
                                           + np.array(per_contig_avg[:, 1] >= r_upper) \
                                           + np.array(per_contig_avg[:, 2] >= e_upper)

                misplaced_array = np.array(per_contig_avg[:, 0] >= m_level) \
                                  + np.array(per_contig_avg[:, 1] >= r_level) \
                                  + np.array(per_contig_avg[:, 2] >= e_level)
                # very_misplaced_array = np.array(per_contig_avg[:, 0] > m_level) + np.array(per_contig_avg[:, 1] > r_level) + np.array(per_contig_avg[:, 2] > e_level)
                if contigs['contigLen'][misplaced_array].sum() >= 1.0e6:
                    misplaced_contigs = True
                if contigs['contigLen'][misplaced_array].sum() >= 2.0e6:
                    very_misplaced_contigs = True
                if contigs['contigLen'][slightly_misplaced_array].sum() >= 1.0e6:
                    slightly_misplaced_contigs = True

                # Always check bins with bad bin stats or if they are large, just for sanity check
                if (
                    bin_size >= 10e6
                    or misplaced_contigs or very_misplaced_contigs or slightly_misplaced_contigs
                    or mean_agg >= a_level or mean_md >= m_level
                    or mean_euc >= e_level or mean_tnf >= r_level
                    or bin_contamination > max_contamination
                ):

                    if debug:
                        print("Reclustering bin %d" % bin_id)
                    if bin_contamination > max_contamination:
                        factor = min(max(m_level, a_level) * 5.0 + bin_contamination / 100, 1.0)
                    elif bin_size >= max_bin_size:
                        factor = 1
                    elif bin_size >= 16e6:
                        factor = min(max(m_level, a_level) * 2.5, 1.0)
                    elif bin_size >= 14e6 or very_misplaced_contigs:
                        factor = min(max(m_level, a_level) * 2.0, 1.0)
                    elif misplaced_contigs or bin_size >= 12e6:
                        factor = min(max(m_level, a_level) * 1.5, 1.0)
                    # elif bin_size >= 10e6:
                    #     factor = min(max(mean_md, mean_agg) * 1.25, 1.0)
                    else:
                        factor = min(max(a_upper, m_upper) * 1.25, 1.0)

                    if debug:
                        print(f"{bin_id} {factor} {bin_size}")
                    reembed_separately.append(bin_id)
                    lower_thresholds.append(1 - factor)
                    force_new_clustering.append(True)  # send it to turbo hell
                    reembed_if_no_cluster.append(True)
                    switches.append([0, 1, 2])
                else:
                    factor = 1 - ((mean_md + mean_tnf) / 2)
                    if factor < 0.95:
                        reembed_separately.append(bin_id)
                        force_new_clustering.append(False)  # send it to regular hell
                        reembed_if_no_cluster.append(True)
                        lower_thresholds.append(factor)
                        switches.append([0, 1, 2])
                    else:
                        self.survived.append(bin_id)

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

            for (bin_id, force_new, min_validity, reembed_cluster, switch) in zip(
                    reembed_separately,
                    force_new_clustering,
                    lower_thresholds,
                    reembed_if_no_cluster,
                    switches
            ):

                try:
                    signal.alarm(300)
                    if debug:
                        print(f"{bin_id} reembedding")
                    result = reembed_static(
                        bin_id,
                        self.bins[bin_id],
                        self.extract_contigs(self.bins[bin_id]),
                        self.embeddings[
                            self.large_contigs[
                                ~self.disconnected][
                                ~self.disconnected_intersected
                            ]['tid'].isin(self.bins[bin_id])
                        ],
                        n_samples,
                        sample_distances,
                        self.n_neighbors,
                        self.n_components,
                        min_validity,
                        reembed_cluster,
                        False,
                        False,
                        2022,
                        self.threads,
                    )
                    signal.alarm(0)
                    if debug:
                        print(f"{bin_id} handling")
                    plots, remove = self.handle_new_embedding(
                        result[0], result[1], result[2], result[3], result[4],
                        plots, n, x_min, x_max, y_min, y_max, False, result[6], debug=False
                    )

                    if remove:
                        if debug:
                            print("Removing bin %d..." % result[0])
                        bins_to_remove.append(result[0])
                        self.overclustered = True

                    else:
                        continue
                        # try:
                        #     contigs, log_lengths, tnfs = self.extract_contigs(self.bins[result[0]])
                        #     bin_completeness, bin_contamination = self.retrieve_bin_checkm_stats(result[0])
                        #     bin_size = contigs['contigLen'].sum()
                        #     if result[4] == 0.0:
                        #         max_contig_size_to_remove = 1e9 # essentially unlimited
                        #     elif result[5]:
                        #         max_contig_size_to_remove = 1e5
                        #     else:
                        #         max_contig_size_to_remove = 1e4
                        #
                        #     if debug:
                        #         print(f"max contig size to remove {max_contig_size_to_remove} for {result[0]}")
                        #     self.prune_bin(
                        #         result[0],
                        #         self.bins[result[0]],
                        #         contigs,
                        #         log_lengths,
                        #         tnfs,
                        #         n_samples,
                        #         sample_distances,
                        #         max_contig_size_to_remove,
                        #         bins_to_remove,
                        #         something_changed,
                        #         min_completeness,
                        #         max_contamination,
                        #         bin_completeness,
                        #         bin_contamination,
                        #         debug=debug
                        #     )
                        # except ZeroDivisionError:
                        #     # Only one contig left, break out
                        #     continue

                except TimeoutError:

                    continue

        # find any bins that potentially timedout
        [self.survived.append(bin_id) for bin_id in reembed_separately if bin_id not in bins_to_remove or bin not in something_changed]

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

                contigs, _, _ = self.extract_contigs(unbinned)
                not_recovered += contigs['contigLen'].sum()

                if not_recovered > (original_size * 0.6) and not force:
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
                    if len(unbinned) >= 1:
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
    original_bin_id,
    tids,
    extraction,
    unbinned_embeddings,
    n_samples,
    sample_distances,
    max_n_neighbours=200,
    n_components=2,
    default_min_validity=0.85,
    reembed=False,
    force=False,
    debug=False,
    random_seed=22,
    threads=10,
    attempts=0,
    max_attempts=3
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
    """

    # import warnings
    # from numba import set_num_threads
    # import threadpoolctl
    # set_num_threads(threads)

    noise = False
    precomputed = False  # Whether the precomputed clustering was the best result
    tids = list(np.sort(tids))
    contigs, log_lengths, tnfs = extraction

    min_validity = default_min_validity
    labels = []
    max_validity = -1
    new_embeddings = None

    with warnings.catch_warnings() and threadpoolctl.threadpool_limits(limits=threads, user_api='blas'):
        warnings.simplefilter("ignore")

        if len(set(tids)) >= 3:
            if debug:
                print(f"Beginning first cluster of bin {original_bin_id}")

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

                if debug:
                    print(f"Generating distance matrix...")

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

                # labels_multi = np.array([-1 for _ in range(len(tids))])
                # labels_kmeans_precom = np.array([-1 for _ in range(len(tids))])
                # labels_kmeans_embeddings = np.array([-1 for _ in range(len(tids))])

                # validity_multi = -1
                # kmeans_score_precom = -1
                # kmeans_score_embeddings = -1

                max_multi = validity_multi
                max_precom = kmeans_score_precom
                max_kmeans = kmeans_score_embeddings

                if debug:
                    print('Allow multi cluster validity: ', max_multi)
                    print('precom cluster validity: ', max_precom)

                if max_multi == -1 and max_precom == -1 and max_kmeans == -1:
                    labels = max_precom
                    max_validity = -1
                elif max(max_multi, max_precom, max_kmeans) == max_kmeans and max_multi < min_validity:
                    labels = labels_kmeans_embeddings
                    max_validity = kmeans_score_embeddings
                elif max(max_multi, max_precom) == max_precom and max_multi < min_validity:
                    labels = labels_kmeans_precom
                    max_validity = max_precom
                    precomputed = True
                else:
                    labels = labels_multi
                    max_validity = max_multi

                set_labels = set(labels)

                if debug:
                    print("No. of Clusters:", len(set_labels), set_labels)

            except IndexError:
                if debug:
                    print('Caught index error')
                # Index error occurs when doing recluster after adding disconnected TNF
                # contigs. Since the embedding array does not contain the missing contigs
                # as such, new embeddings have to be calculated
                labels = np.array([-1 for _ in range(unbinned_embeddings.shape[0])])
                max_validity = -1


            if -1 in labels and len(set(labels)) == 2:
                # fringe case where only single cluster formed with external
                # unbinned contigs on external part of bin.
                if max_validity > min_validity:
                    # lower binning quality and see if reembedding can do better
                    max_validity = min_validity

            if debug:
                print(f"max v {max_validity} min v thresh {min_validity}, reembed? {reembed} {len(tids)} {(max_validity < 0.95 or min_validity <= 0.5) and reembed and len(tids) >= 5}")

            if (max_validity < 0.95 or min_validity <= 0.5) and reembed and len(tids) >= 5:
                # Generate new emebddings if clustering seems fractured
                # contigs, log_lengths, tnfs = self.extract_contigs(tids)
                # try:
                embeddings = []
                for i in range(0, 2):
                    seed = random_seed << i
                    if seed >= 2**32:
                        random_seed = random.randint(69, 420)
                        seed = random_seed << i

                    precomputed_reducer_low = umap.UMAP(
                        metric="precomputed",
                        n_neighbors=max_n_neighbours,
                        n_components=n_components,
                        a=1.48,
                        b=0.3,
                        n_jobs=threads,
                        random_state = seed
                    )

                    precomputed_reducer_mid = umap.UMAP(
                        metric="precomputed",
                        n_neighbors=max_n_neighbours,
                        n_components=n_components,
                        a=1.48,
                        b=0.4,
                        n_jobs=threads,
                        random_state = seed
                    )

                    try:
                        new_embeddings_1 = precomputed_reducer_low.fit_transform(sp_distance.squareform(stat))
                        new_embeddings_2 = precomputed_reducer_mid.fit_transform(sp_distance.squareform(stat))
                        embeddings.append(new_embeddings_1)
                        embeddings.append(new_embeddings_2)
                    except (FloatingPointError, TypeError) as e:
                        if debug:
                            print(f"Hit error {e}.. continuing")


                if len(embeddings) >= 1:
                    labels_multi, validity_multi, _, _ = Clusterer.ensemble_cluster_multiple_embeddings(
                            embeddings,
                            top_n=3,
                            metric="euclidean",
                            cluster_selection_methods="eom",
                            solver="hbgf",
                            threads=threads,
                            use_multiple_processes=False
                        )
                    labels_multi = labels_multi[0]
                    validity_multi = validity_multi[0]

                    kmeans_labels, kmeans_score, kmeans_embedding_idx = get_best_kmeans_of_multiple(
                        embeddings,
                        random_seed=random_seed
                    )

                    if kmeans_score > validity_multi and validity_multi < min_validity:
                        new_embeddings = embeddings[kmeans_embedding_idx]
                        best_labels = kmeans_labels
                        best_val = kmeans_score
                    else:
                        new_embeddings = embeddings[0]
                        best_labels = labels_multi
                        best_val = validity_multi

                    if debug:
                        print(f"Umap results: {len(set(labels))} {set(labels)} {best_val}")
                    if max_validity <= best_val or (min_validity < 0.1 and max_validity <= 0.5):
                        if all(label == -1 for label in best_labels):
                            pass
                        else:
                            unbinned_embeddings = new_embeddings
                            labels = best_labels
                            max_validity = best_val
                            if precomputed:
                                precomputed = False  # No longer the best results
                            if debug:
                                print("No. of Clusters:", len(set(labels)), set(labels))

                    else:
                        if debug:
                            print('using non re-embedded... %f' % max_validity)
                else:
                    if debug:
                        print('No suitable embeddings...')

                if unbinned_embeddings.all() == 0 or \
                        (max_validity <= 0 and max_validity != -1 and min_validity != 0) or \
                        (-1 in labels and len(set(labels)) == 2) or \
                        (max_validity < min_validity and min_validity <= 0.6):
                    # embedding failed, which only happens for smaller bins
                    # check to see if kmeans clustering splits bins sensibly
                    if debug:
                        print('Using precomputed kmeans result...')
                        print(unbinned_embeddings.all() == 0, max_validity, min_validity, set(labels))
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

                    if score > max_validity:
                        max_validity = score
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

    if max_validity < min_validity \
            and max_validity * 1.5 >= min_validity \
            and attempts <= max_attempts and min_validity <= 0.75:
        return reembed_static(
                    original_bin_id,
                    tids,
                    extraction,
                    unbinned_embeddings,
                    n_samples,
                    sample_distances,
                    max_n_neighbours,
                    n_components,
                    default_min_validity,
                    reembed,
                    force,
                    debug,
                    random_seed << 1 + random.randint(69, 420),
                    threads,
                    attempts + 1,
                    max_attempts
                )

    if len(labels) == 0:
        labels = np.array([-1 for _ in range(len(tids))])
        max_validity = -1
        new_embeddings = unbinned_embeddings

    return original_bin_id, labels, new_embeddings, max_validity, min_validity, force, precomputed


def get_best_kmeans_result(distances, max_n_clusters=10, random_seed=22, n_jobs=1):
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

def get_best_kmeans_of_multiple(list_of_embeddings, max_n_clusters=10, random_seed=22):
    """
    embeddings - a list of multiple different embedding results, not a single embedding. Or at least one
                 embedding wrapped in a list
    """
    max_score = 0
    max_labels = None
    best_idx = None
    for (idx, embeddings) in enumerate(list_of_embeddings):
        best_label, best_score = get_best_kmeans_result(embeddings, max_n_clusters, random_seed)
        if best_score > max_score:
            max_score = best_score
            max_labels = best_label
            best_idx = idx

    return max_labels, max_score, best_idx

