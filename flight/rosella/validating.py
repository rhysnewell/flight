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
from numba import njit
import seaborn as sns
import matplotlib
import sklearn.metrics as sk_metrics

# self imports
import flight.metrics as metrics
import flight.utils as utils
from flight.rosella.binning import Binner
from flight.rosella.clustering import Clusterer
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
                      quick_filter=False, debug=False,
                      force=False, truth_array=None, dissolve=False):
        """
        Function for deciding whether a bin needs to be reembedded or split up
        Uses internal bin statistics, mainly mean ADP and Rho values
        """

        if self.n_samples > 0:
            n_samples = self.n_samples
            sample_distances = self.short_sample_distance
        else:
            n_samples = self.long_samples
            sample_distances = self.long_sample_distance

        bins_to_remove = []
        new_bins = {}
        new_bin_counter = 0
        logging.debug("Checking bin internal distances...")
        big_tids = []
        reembed_separately = []  # container for bin ids that look like chimeras
        force_new_clustering = []
        reembed_if_no_cluster = []
        bins = self.bins.keys()
        for bin in bins:
            logging.debug("Beginning check on bin: ", bin)
            tids = self.bins[bin]
            if len(tids) == 1:
                continue
            elif bin == 0:
                continue

            if dissolve:
                ## Dissolve very small or loq quality bins for re-emebedding
                contigs, log_lengths, tnfs = self.extract_contigs(tids)
                bin_size = contigs['contigLen'].sum()
                if bin_size <= 5e5:
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
                    except ZeroDivisionError:
                        # Only one contig left, break out
                        break

                    # IFF the bin is extra busted just obliterate it
                    if (mean_md >= 0.5 or mean_agg >= 0.5) and (mean_tnf >= 0.2 or mean_euc >= 3.5):
                        self.unbinned_tids = self.unbinned_tids + tids
                        bins_to_remove.append(bin)

            elif quick_filter:
                ## Remove stuff that is obviously wrong
                contigs, log_lengths, tnfs = self.extract_contigs(tids)
                bin_size = contigs['contigLen'].sum()
                if bin_size < 1e6:
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
                    except ZeroDivisionError:
                        # Only one contig left, break out
                        break

                    removed = []

                    if debug:
                        print('before check for distant contigs: ', len(tids))
                        _, _, _, _ = self.compare_bins(bin)

                    if mean_md >= 0.15 or mean_agg >= 0.25:
                        # Simply remove
                        for (tid, avgs) in zip(tids, per_contig_avg):
                            if ((avgs[0] >= 0.7 or avgs[3] >= 0.5) and
                                (avgs[1] > 0.15 or avgs[2] >= 4)) or \
                                    ((avgs[0] >= 0.5 or avgs[3] >= 0.5) and
                                     (avgs[1] >= 0.5 or avgs[2] >= 6)):
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


            elif big_only:

                # filtering = True
                remove = False  # Whether to completely remove original bin
                removed_single = []  # Single contig bin
                removed_together = []  # These contigs form their own bin
                contigs, log_lengths, tnfs = self.extract_contigs(tids)

                bin_size = contigs['contigLen'].sum()

                if bin_size >= 3e6:
                    # while filtering:

                    # Extract current contigs and get statistics
                    # contigs, log_lengths, tnfs = self.extract_contigs(tids)
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

                    removed_inner = []  # inner container that is rewritten every iteration

                    if len(tids) == 2:
                        # Higher thresholds for fewer contigs
                        md_filt = 0.35
                        agg_filt = 0.45
                        euc_filt = 2
                        rho_filt = 0.05
                        # Two contigs by themselves that are relatively distant. Remove them separately
                        together = True
                    elif len(tids) <= 5:
                        # Higher thresholds for fewer contigs
                        md_filt = 0.35
                        agg_filt = 0.45
                        euc_filt = 2
                        rho_filt = 0.05
                        together = True
                    else:
                        # Lower thresholds for fewer contigs
                        md_filt = 0.35
                        agg_filt = 0.45
                        euc_filt = 2
                        rho_filt = 0.05
                        together = True

                    if mean_md >= 0.15 or mean_agg >= 0.25:
                        if debug:
                            print("Checking big contigs for bin: ", bin)
                        md_std = max(np.std(per_contig_avg[:, 0]), 0.1)
                        rho_std = max(np.std(per_contig_avg[:, 1]), 0.05)
                        euc_std = max(np.std(per_contig_avg[:, 2]), 0.5)
                        agg_std = max(np.std(per_contig_avg[:, 3]), 0.1)
                        for max_idx in range(per_contig_avg.shape[0]):
                            # max_idx = np.argmax(per_contig_avg[:, 3]) # Check mean_agg first
                            max_values = per_contig_avg[max_idx, :]
                            contig_length = contigs['contigLen'].iloc[max_idx]
                            if debug:
                                print("Contig size and tid: ", contig_length, tids[max_idx])

                            if contig_length >= 1e6:
                                if debug:
                                    print("Found large contig: ", max_idx, tids[max_idx])
                                if (max_values[3] >= agg_filt or max_values[0] >= md_filt
                                    or max_values[3] >= (mean_agg + agg_std)
                                    or max_values[0] >= (mean_md + md_std)) and \
                                        ((max_values[1] >= rho_filt
                                          or max_values[1] >= (mean_tnf + rho_std))
                                         or (max_values[2] >= euc_filt
                                             or max_values[2] >= (mean_euc + euc_std))):

                                    if debug:
                                        print("Removing contig: ", max_idx, tids[max_idx])
                                    if together:
                                        removed_inner.append(tids[max_idx])
                                        removed_together.append(tids[max_idx])
                                    else:
                                        removed_inner.append(tids[max_idx])
                                        removed_single.append(tids[max_idx])
                                elif (max_values[0] >= 0.6 or max_values[1] >= 0.25 or max_values[2] >= 6.5):
                                    if together:
                                        removed_inner.append(tids[max_idx])
                                        removed_together.append(tids[max_idx])
                                    else:
                                        removed_inner.append(tids[max_idx])
                                        removed_single.append(tids[max_idx])

                        if len(removed_inner) > 0:
                            [tids.remove(r) for r in removed_inner]

                    # logging.debug(filtering, len(removed_single), len(removed_together), bin)
                    if len(removed_single) > 0 or len(removed_together) > 0:
                        [big_tids.append(r) for r in removed_single]

                        new_bins[new_bin_counter] = []
                        [new_bins[new_bin_counter].append(r) for r in removed_together]
                        new_bin_counter += 1

                        current_contigs, current_lengths, current_tnfs = self.extract_contigs(tids)
                        if current_contigs['contigLen'].sum() <= self.min_bin_size:
                            [self.unbinned_tids.append(tid) for tid in tids]
                            remove = True

                        if bin in self.survived:
                            self.survived.remove(bin)

                        if len(tids) == 0 or remove:
                            bins_to_remove.append(bin)


            elif not size_only \
                    and reembed \
                    and bin != 0 \
                    and len(tids) > 1:

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
                    except ZeroDivisionError:
                        continue

                    removed = []

                    if debug:
                        print('before check for distant contigs: ', len(tids))
                        _, _, _, _ = self.compare_bins(bin)

                    f_level = 0.15
                    m_level = 0.15
                    shared_level = 0.1

                    if len(removed) >= 1:
                        # calc new bin size and stats
                        contigs, log_lengths, tnfs = self.extract_contigs(tids)
                        bin_size = contigs['contigLen'].sum()
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
                            print('contigs removed: ', len(tids))
                            _, _, _, _ = self.compare_bins(bin)

                    if ((mean_md >= m_level
                         or mean_agg >= f_level
                         or (mean_md >= shared_level and (mean_tnf >= shared_level or mean_euc >= 2))
                         or ((mean_md >= 0.05 or mean_agg >= 0.15) and (mean_tnf >= 0.1 or mean_euc >= 2)))
                        and bin_size > 1e6) or bin_size >= 12e6:
                        logging.debug(bin, mean_md, mean_tnf, mean_agg, len(tids))
                        reembed_separately.append(bin)
                        if (((mean_md >= 0.4 or mean_agg >= 0.45) and (mean_tnf >= 0.1 or mean_euc >= 3))
                                or bin_size >= 13e6):
                            if debug:
                                logging.debug("Forcing bin %d" % bin)
                                self.compare_bins(bin)
                            if bin_size >= 13e6 or ((mean_tnf >= 0.15 or mean_euc >= 3.5) and (bin_size >= 2e6)):
                                force_new_clustering.append(False)  # send it to turbo hell
                            else:
                                force_new_clustering.append(False)
                            reembed_if_no_cluster.append(True)
                        elif bin_size > 1e6:
                            if debug:
                                print("Reclustering bin %d" % bin)
                            force_new_clustering.append(False)  # send it to regular hell
                            reembed_if_no_cluster.append(True)
                    else:
                        reembed_separately.append(bin)
                        force_new_clustering.append(False)  # send it to regular hell
                        reembed_if_no_cluster.append(False)  # take it easy, okay?
                        # if debug:
                        #     print("bin survived %d" % bin)
                        #     self.compare_bins(bin)
                        # self.survived.append(bin)
                else:
                    logging.debug(bin, self.survived)



            elif size_only:
                logging.debug("Size only check when size only is ", size_only)
                contigs, log_lengths, tnfs = self.extract_contigs(tids)
                bin_size = contigs['contigLen'].sum()

                if bin_size >= 13e6 and bin != 0:
                    # larger than most bacterial genomes, way larger than archaeal
                    # Likely strains getting bunched together. But they won't disentangle, so just dismantle the bin
                    # rescuing any large contigs. Only way I can think  of atm to deal with this.
                    # Besides perhaps looking at variation level?? But this seems to be a problem with
                    # the assembly being TOO good.
                    if reembed:
                        reembed_separately.append(bin)
                        reembed_if_no_cluster.append(True)
                        force_new_clustering.append(False)  # turbo hell
                elif bin_size >= 1e6 and bin != 0:
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

                    if (mean_md >= 0.4 or mean_agg >= 0.45) and (mean_tnf >= 0.15 or mean_euc >= 3.5) \
                            and bin_size > 1e6:
                        if debug:
                            print("In final bit. ", bin)
                            self.compare_bins(bin)
                        reembed_separately.append(bin)
                        reembed_if_no_cluster.append(True)
                        force_new_clustering.append(False)  # send it to turbo hell

                    else:
                        # self.survived.append(bin)
                        pass

            elif force:
                contigs, log_lengths, tnfs = self.extract_contigs(tids)
                bin_size = contigs['contigLen'].sum()

                if bin_size >= 1e6 and bin != 0:
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

                    if (mean_md >= 0.5 or mean_agg >= 0.45) and (mean_tnf >= 0.15 or mean_euc >= 3.5) \
                            and bin_size > 1e6:
                        if debug:
                            print("In final bit. ", bin)
                            self.compare_bins(bin)
                        reembed_separately.append(bin)
                        reembed_if_no_cluster.append(True)
                        force_new_clustering.append(True)  # send it to turbo hell
                    else:
                        pass

        try:
            max_bin_id = max(self.bins.keys()) + 1
        except ValueError:
            max_bin_id = 1
        for k, v in new_bins.items():
            self.bins[max_bin_id + k] = list(np.sort(np.array(v)))

        for bin, force_new, reembed_cluster in zip(reembed_separately, force_new_clustering, reembed_if_no_cluster):
            tids = self.bins[bin]

            logging.debug("Checking bin %d..." % bin)
            try:
                max_bin_id = max(self.bins.keys()) + 1
            except ValueError:
                max_bin_id = 1

            if isinstance(max_bin_id, np.int64):
                max_bin_id = max_bin_id.item()

            plots, remove = self.reembed(tids, max_bin_id, plots,
                                         x_min, x_max, y_min, y_max, n,
                                         relaxed=force_new,
                                         reembed=reembed_cluster,
                                         truth_array=truth_array, debug=debug)
            if debug:
                print("Problem bin result... removing: ", remove)

            if remove:
                if debug:
                    print("Removing bin %d..." % bin)
                bins_to_remove.append(bin)
                self.overclustered = True
            elif force_new:
                logging.debug("Removing bin %d through force..." % bin)
                big_tids = big_tids + self.bins[bin]
                bins_to_remove.append(bin)
            else:
                if debug:
                    print("Keeping bin %d..." % bin)
                self.survived.append(bin)

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
            None

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
                delete_unbinned=False,
                bin_unbinned=False,
                force=False,
                relaxed=False,
                reembed=False,
                skip_clustering=False,
                update_embeddings=False,
                truth_array=None,
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
        remove = False
        noise = False
        precomputed = False  # Whether the precomputed clustering was the best result
        tids = list(np.sort(tids))
        contigs, log_lengths, tnfs = self.extract_contigs(tids)
        original_size = contigs['contigLen'].sum()
        min_validity = 1

        if not reembed and not force:
            strict = True
        else:
            strict = False

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
                    labels_single = self.iterative_clustering(unbinned_embeddings,
                                                              allow_single_cluster=True,
                                                              prediction_data=False,
                                                              double=False)
                    labels_multi = self.iterative_clustering(unbinned_embeddings,
                                                             allow_single_cluster=False,
                                                             prediction_data=False,
                                                             double=False)

                    # if max_validity == -1:
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

                    validity_single, _ = self._validity(labels_single, unbinned_embeddings)
                    validity_multi, _ = self._validity(labels_multi, unbinned_embeddings)
                    validity_precom, _ = self._validity(labels_precomputed, unbinned_embeddings)

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
                        if strict:
                            min_validity = 0.8
                        else:
                            min_validity = 0.7
                    elif max(max_single, max_multi) == max_single:
                        self.labels = labels_single
                        max_validity = max_single
                        if strict:
                            min_validity = 0.9
                        else:
                            min_validity = 0.85
                    else:
                        self.labels = labels_multi
                        max_validity = max_multi
                        if strict:
                            min_validity = 0.9
                        else:
                            min_validity = 0.85

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
                unbinned_embeddings = self.embeddings[unbinned_array]
                self.labels = np.array([-1 for i in range(unbinned_embeddings.shape[0])])
                max_validity = -1

            if max_validity < 0.95 and reembed and len(tids) >= 5:
                # Generate new emebddings if clustering seems fractured
                contigs, log_lengths, tnfs = self.extract_contigs(tids)

                try:

                    self.fit_transform(tids, max_n_neighbours)

                    new_embeddings = self.intersection_mapper.embedding_
                    if update_embeddings:
                        self.embeddings = new_embeddings

                    labels_single = self.iterative_clustering(new_embeddings,
                                                              allow_single_cluster=True,
                                                              prediction_data=False,
                                                              double=False)
                    labels_multi = self.iterative_clustering(new_embeddings,
                                                             allow_single_cluster=False,
                                                             prediction_data=False,
                                                             double=False)

                    validity_single, _ = self._validity(labels_single, new_embeddings)
                    validity_multi, _ = self._validity(labels_multi, new_embeddings)

                    logging.debug('Allow single cluster validity: ', validity_single)
                    logging.debug('Allow multi cluster validity: ', validity_multi)

                    if validity_single >= validity_multi:
                        if max_validity <= validity_single:
                            if all(label == -1 for label in labels_single):
                                if debug:
                                    print('using non re-embedded...')
                            else:
                                unbinned_embeddings = new_embeddings
                                self.labels = labels_single
                                max_validity = validity_single
                                min_validity = 0.85
                                if precomputed:
                                    precomputed = False  # No longer the best results
                        else:
                            if debug:
                                print('using non re-embedded... %f' % max_validity)
                    else:
                        if max_validity <= validity_multi:
                            if all(label == -1 for label in labels_multi):
                                logging.debug('using non re-embedded...')
                            else:
                                unbinned_embeddings = new_embeddings
                                self.labels = labels_multi
                                max_validity = validity_multi
                                min_validity = 0.85
                                if precomputed:
                                    precomputed = False  # No longer the best results
                        else:
                            if debug:
                                print('using non re-embedded... %f' % max_validity)




                except TypeError:
                    self.labels = np.array([-1 for i in range(unbinned_embeddings.shape[0])])

            if relaxed:
                min_validity = 0.5
                min_distance = 0.5

            set_labels = set(self.labels)

            if debug:
                print("No. of Clusters:", len(set_labels), set_labels)
                print("Max validity: ", max_validity)

            plots = self.add_plot(plots, unbinned_embeddings, contigs,
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
                        # if bin_size < self.min_bin_size:
                        #     if debug:
                        #         print("Didn't recover enough: %d of %d" % (bin_size, original_size))
                        #     not_recovered += bin_size
                        # else:
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
                    # ## Get cluster distances.
                    # cluster_separation = utils.cluster_distances(unbinned_embeddings, self.labels)
                    # new_bins = []
                    # Half the original input has been binned if reembedding
                    for bin, new_tids in new_bins.items():
                        new_tids = list(np.sort(new_tids))
                        contigs, log_lengths, tnfs = self.extract_contigs(new_tids)
                        bin_size = contigs['contigLen'].sum()
                        if (bin_size >= 1e6 and reembed) \
                                or (not reembed and bin_size >= self.min_bin_size) \
                                or (force and bin_size >= self.min_bin_size):
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
                                # if contig.contigLen >= 2e6:
                                #     self.bins[bin_id] = [self.assembly[contig.contigName]]
                                #     bin_id += 1
                                # else:
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
            try:
                max_bin_id = max(self.bins.keys())
            except ValueError:
                max_bin_id = 1

            for idx in tids:
                if self.large_contigs[self.large_contigs['tid'] == idx]['contigLen'].iloc[0] >= self.min_bin_size:
                    max_bin_id += 1
                    self.bins[max_bin_id] = [idx]

        try:
            max_bin_id = max(self.bins.keys())
        except ValueError:
            max_bin_id = 1

        if isinstance(max_bin_id, np.int64):
            max_bin_id = max_bin_id.item()

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

        return plots, remove