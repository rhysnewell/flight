#!/usr/bin/env python
###############################################################################
# flight.py - A fast binning algorithm spinning off of the methodology of
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
#                                                                             #
# You should have received a copy of the GNU General Public License           #
# along with this program. If not, see <http://www.gnu.org/licenses/>.        #
#                                                                             #
###############################################################################
from Bio import SeqIO
from flight.__init__ import __version__
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
import datetime

# Function imports
import numpy as np
import pandas as pd
from numba import config, set_num_threads
import threadpoolctl
import warnings
import imageio

# Self imports
from .binning import Binner
from .cluster import Cluster
import flight.utils as utils

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
def main():
    ############################ ~ Main Parser ~ ##############################
    main_parser = argparse.ArgumentParser(prog='flight',
                                          formatter_class=CustomHelpFormatter,
                                          add_help=False)
    main_parser.add_argument('--version',
                             action='version',
                             version=__version__,
                             help='Show version information.')
    main_parser.add_argument(
        '--verbosity',
        help=
        '1 = critical, 2 = error, 3 = warning, 4 = info, 5 = debug. Default = 4 (logging)',
        type=int,
        default=4)
    main_parser.add_argument('--log',
                             help='Output logging information to file',
                             default=False)
    subparsers = main_parser.add_subparsers(help="--", dest='subparser_name')

    ########################## ~ sub-parser ~ ###########################
    fit_options = subparsers.add_parser(
        'fit',
        description='Perform UMAP and then HDBSCAN on array of variant depths',
        formatter_class=CustomHelpFormatter,
        epilog='''
                                    ~ fit ~
        How to use fit:

        flight fit --depths depths.npy

        ''')
    ## Main input array. Depths or Distances
    fit_options.add_argument(
        '--input',
        help='.npy file contain depths of variants for each sample',
        dest="input",
        required=True)

    ## UMAP parameters
    fit_options.add_argument('--n_neighbors',
                             help='Number of neighbors considered in UMAP',
                             dest="n_neighbors",
                             default=100)

    fit_options.add_argument(
        '--min_dist',
        help=
        'Minimum distance used by UMAP during construction of high dimensional graph',
        dest="min_dist",
        default=0)

    fit_options.add_argument('--n_components',
                             help='Dimensions to use in UMAP projection',
                             dest="n_components",
                             default=2)

    fit_options.add_argument('--metric',
                             help='Metric to use in UMAP projection',
                             dest="metric",
                             default="euclidean")

    ## HDBSCAN parameters
    fit_options.add_argument('--min_cluster_size',
                             help='Minimum cluster size for HDBSCAN',
                             dest="min_cluster_size",
                             default=5)

    fit_options.add_argument('--min_samples',
                             help='Minimum samples for HDBSCAN',
                             dest="min_samples",
                             default=1)

    ## Genral parameters
    fit_options.add_argument(
        '--precomputed',
        help='Minimum cluster size for HDBSCAN',
        dest="precomputed",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
    )

    fit_options.add_argument(
        '--cores',
        help='Number of cores to run UMAP with',
        dest='threads',
        default=8
    )

    fit_options.set_defaults(func=fit)


    bin_options = subparsers.add_parser(
        'bin',
        description='Perform UMAP and then HDBSCAN on array of variant depths',
        formatter_class=CustomHelpFormatter,
        epilog='''
                                ~ bin ~
    How to use bin:

    flight bin --input coverm_output.tsv --assembly scaffolds.fasta

    ''')
    ## Main input array. Coverages from CoverM contig
    bin_options.add_argument(
        '--input',
        help='CoverM coverage results',
        dest="input")

    bin_options.add_argument(
        '--long_input',
        help='CoverM coverage results',
        dest="long_input")

    bin_options.add_argument(
        '--assembly',
        help='FASTA file containing scaffolded contigs of the metagenome assembly',
        dest="assembly",
        required=True,
    )

    bin_options.add_argument(
        '--variant_rates',
        help='Per contig SNV and SV rates over a given sliding window size',
        dest='variant_rates',
        required=False
    )

    bin_options.add_argument(
        '--kmer_frequencies',
        help='Per contig kmer frequencies. Can be calculated using rosella kmer mode',
        dest='kmer_frequencies',
        required=True,
    )

    bin_options.add_argument(
        '--min_bin_size',
        help='The minimum size of a returned MAG in base pairs',
        dest="min_bin_size",
        default=200000,
        required=False,
    )

    bin_options.add_argument(
        '--scaler',
        help='The method used to scale the input data',
        dest="scaler",
        default='clr',
        choices=['clr', 'minmax', 'none'],
        required=False,
    )

    bin_options.add_argument(
        '--min_contig_size',
        help='The minimum contig size to be considered for binning',
        dest="min_contig_size",
        default=1000,
        required=False,
    )

    bin_options.add_argument(
        '--output_directory',
        help='Output directory',
        dest="output",
        default="rosella_bins",
        required=False,
    )

    ## UMAP parameters
    bin_options.add_argument('--n_neighbors',
                             help='Number of neighbors considered in UMAP',
                             dest="n_neighbors",
                             default=100)


    bin_options.add_argument(
        '--a_spread',
        help=
        'The spread of UMAP embeddings. Directly manipulates the "a" parameter',
        dest="a",
        default=1.58)

    bin_options.add_argument(
        '--b_tail',
        help=
        'Similar to the heavy-tail parameter sometimes used in t-SNE. Directly manipulates the "b" parameter',
        dest="b",
        default=0.5)

    bin_options.add_argument(
        '--min_dist',
        help=
        'Minimum distance used by UMAP during construction of high dimensional graph',
        dest="min_dist",
        default=0.1)

    bin_options.add_argument('--n_components',
                             help='Dimensions to use in UMAP projection',
                             dest="n_components",
                             default=3)

    bin_options.add_argument('--metric',
                             help='Metric to use in UMAP projection',
                             dest="metric",
                             default="aggregate_variant_tnf")
    ## HDBSCAN parameters
    bin_options.add_argument('--min_cluster_size',
                             help='Minimum cluster size for HDBSCAN',
                             dest="min_cluster_size",
                             default=5)

    bin_options.add_argument('--min_samples',
                             help='Minimum samples for HDBSCAN',
                             dest="min_samples",
                             default=5)

    bin_options.add_argument('--cluster_selection_method',
                             help='Cluster selection method used by HDBSCAN. Either "eom" or "leaf"',
                             dest='cluster_selection_method',
                             default='eom')

    ## Genral parameters
    bin_options.add_argument(
        '--precomputed',
        help='Flag indicating whether the input matrix is a set of precomputed distances',
        dest="precomputed",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
    )

    bin_options.add_argument('--cores',
                             help='Number of cores to run in parallel',
                             dest='threads',
                             default=8)
    bin_options.set_defaults(func=bin)

    vamb_options = subparsers.add_parser(
        'vamb',
        description='Bin out the results of vamb',
        formatter_class=CustomHelpFormatter,
        epilog='''
                                        ~ vamb ~
            How to use vamb:

            flight vamb --reference assembly.fasta --clusters vamb_clusters.tsv

            ''')

    vamb_options.add_argument('--reference',
                              help='The assembly file to be binned',
                              dest='assembly')

    vamb_options.add_argument('--clusters',
                              help='The vamb clusters',
                              dest='clusters')

    vamb_options.add_argument('--min_size',
                              help='Minimum bin size',
                              dest='min_size',
                              default=200000)

    vamb_options.add_argument('--output',
                              help='The output directory',
                              dest='output',
                              default='vamb_bins/')

    vamb_options.set_defaults(func=vamb)
    ###########################################################################
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Parsing input ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    if (len(sys.argv) == 2 or len(sys.argv) == 1 or sys.argv[1] == '-h'
            or sys.argv[1] == '--help'):
        phelp()
    else:
        args = main_parser.parse_args()
        time = datetime.datetime.now().strftime('%H:%M:%S %d-%m-%Y')

        if args.log:
            if os.path.isfile(args.log):
                raise Exception("File %s exists" % args.log)
            logging.basicConfig(
                filename=args.log,
                level=debug[args.verbosity],
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S %p')
        else:
            logging.basicConfig(
                level=debug[args.verbosity],
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S %p')

        logging.info("Time - %s" % (time))
        logging.info("Command - %s" % ' '.join(sys.argv))

        args.func(args)

def fit(args):
    prefix = args.input.replace(".npy", "")
    os.environ["NUMEXPR_MAX_THREADS"] = args.threads
    if not args.precomputed:
        clusterer = Cluster(args.input,
                           prefix,
                           n_neighbors=int(args.n_neighbors),
                           min_cluster_size=int(args.min_cluster_size),
                           min_samples=int(args.min_samples),
                           min_dist=float(args.min_dist),
                           n_components=int(args.n_components),
                           threads=int(args.threads),
                           )
        clusterer.fit_transform()
        clusterer.cluster()
        # clusterer.break_clusters()
        clusterer.plot()

        np.save(prefix + '_labels.npy', clusterer.labels())
        np.save(prefix + '_separation.npy', clusterer.cluster_separation())
    else:
        clusterer = Cluster(args.input,
                           prefix,
                           n_neighbors=int(args.n_neighbors),
                           min_cluster_size=int(args.min_cluster_size),
                           min_samples=int(args.min_samples),
                           scaler="none",
                           precomputed=args.precomputed,
                           threads=int(args.threads),
                           )
        clusterer.cluster_distances()
        clusterer.plot_distances()
        np.save(prefix + '_labels.npy', clusterer.labels())


def bin(args):
    prefix = args.output
    os.environ["NUMEXPR_MAX_THREADS"] = "1"
    set_num_threads(int(args.threads))
    if args.long_input is None and args.input is None:
        logging.warning("bin requires either short or longread coverage values.")
        sys.exit()

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    if not args.precomputed:
        clusterer = Binner(args.input,
                           args.long_input,
                           args.kmer_frequencies,
                           # args.variant_rates,
                           prefix,
                           args.assembly,
                           n_neighbors=int(args.n_neighbors),
                           metric=args.metric,
                           min_cluster_size=int(args.min_cluster_size),
                           min_contig_size=int(args.min_contig_size),
                           min_samples=int(args.min_samples),
                           min_dist=float(args.min_dist),
                           scaler=args.scaler,
                           n_components=int(args.n_components),
                           cluster_selection_method=args.cluster_selection_method,
                           threads=int(args.threads),
                           a=float(args.a),
                           b=float(args.b),
                           )

        kwargs_write = {'fps':1.0, 'quantizer':'nq'}
        plots = []
        with threadpoolctl.threadpool_limits(limits=int(args.threads), user_api='blas'):
            
            if clusterer.tnfs.values.shape[0] > int(args.n_neighbors):
                # First pass quick TNF filter to speed up next steps
                clusterer.update_parameters()
                clusterer.filter()
                if clusterer.tnfs[~clusterer.disconnected].values.shape[0] > int(args.n_neighbors)*5:
                    # Second pass intersection filtering
                    clusterer.update_parameters()
                    found_disconnections = clusterer.fit_disconnect()
                    # found_disconnections = True
                    if clusterer.tnfs[~clusterer.disconnected][~clusterer.disconnected_intersected].values.shape[0] > int(args.n_neighbors) * 5 and found_disconnections:
                        # Final fully filtered embedding to cluster on
                        # clusterer.update_parameters()
                        clusterer.fit_transform()
                        clusterer.cluster()

                        ## Plot limits
                        x_min = min(clusterer.embeddings[:, 0]) - 5
                        x_max = max(clusterer.embeddings[:, 0]) + 5
                        y_min = min(clusterer.embeddings[:, 1]) - 5
                        y_max = max(clusterer.embeddings[:, 1]) + 5

                        plots.append(utils.plot_for_offset(clusterer.embeddings, clusterer.clusterer.labels_, x_min, x_max, y_min, y_max, 0))
                        clusterer.bin_contigs(args.assembly, int(args.min_bin_size))

                        n = 0
                        old_tids = []
                        logging.info("Performing iterative clustering with disconnections...")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            while n <= 5:
                                
                                clusterer.pairwise_distances()
                                try:
                                    max_bin_id = max(clusterer.bins.keys()) + 1
                                except ValueError:
                                    max_bin_id = 1
                                    
                                if n == 0 or old_tids != set(clusterer.unbinned_tids):
                                    old_tids = set(clusterer.unbinned_tids)
                                else:
                                    
                                    break  # nothing changed
                                plots = clusterer.reembed_unbinned(clusterer.unbinned_tids, max_bin_id,
                                                                plots, x_min, x_max, y_min, y_max, n+1, delete_unbinned=True)
                                n += 1


                            plots = clusterer.reembed_unbinned(clusterer.unbinned_tids, max_bin_id,
                                                                   plots, x_min, x_max, y_min, y_max, n+1,
                                                                   delete_unbinned=True,
                                                                   bin_unbinned=False) # First pass get bins
                            clusterer.pairwise_distances(bin_unbinned=True) # Bin out large unbinned contigs
                            try:
                                max_bin_id = max(clusterer.bins.keys()) + 1
                            except ValueError:
                                max_bin_id = 1
                            plots = clusterer.reembed_unbinned(clusterer.unbinned_tids, max_bin_id,
                                                                   plots, x_min, x_max, y_min, y_max, n+1,
                                                                   delete_unbinned=True,
                                                                   bin_unbinned=True) # second pass get bins

                        clusterer.bin_filtered(int(args.min_bin_size))
                        clusterer.plot()
                    elif not found_disconnections: # run clustering based off first round of embeddings
                        clusterer.cluster()
                        clusterer.bin_contigs(args.assembly, int(args.min_bin_size))
                        ## Plot limits
                        x_min = min(clusterer.embeddings[:, 0]) - 5
                        x_max = max(clusterer.embeddings[:, 0]) + 5
                        y_min = min(clusterer.embeddings[:, 1]) - 5
                        y_max = max(clusterer.embeddings[:, 1]) + 5
                        plots.append(utils.plot_for_offset(clusterer.embeddings, clusterer.clusterer.labels_, x_min, x_max, y_min, y_max, 0))


                        n = 0
                        old_tids = []
                        logging.info("Performing iterative clustering...")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            while n <= 5:

                                clusterer.pairwise_distances()
                                try:
                                    max_bin_id = max(clusterer.bins.keys()) + 1
                                except ValueError:
                                    max_bin_id = 1
                                    
                                if n == 0 or old_tids != set(clusterer.unbinned_tids):
                                    old_tids = set(clusterer.unbinned_tids)
                                else:

                                    break  # nothing changed
                                plots = clusterer.reembed_unbinned(clusterer.unbinned_tids,
                                                                   max_bin_id,
                                                                   plots, x_min, x_max, y_min, y_max, n+1, delete_unbinned=True)
                                n += 1

                            plots = clusterer.reembed_unbinned(clusterer.unbinned_tids, max_bin_id,
                                                                                           plots, x_min, x_max, y_min, y_max, n+1,
                                                                                           delete_unbinned=True,
                                                                                           bin_unbinned=False) # First pass get bins
                            clusterer.pairwise_distances(bin_unbinned=True) # Bin out large unbinned contigs
                            try:
                                max_bin_id = max(clusterer.bins.keys()) + 1
                            except ValueError:
                                max_bin_id = 1
                            plots = clusterer.reembed_unbinned(clusterer.unbinned_tids, max_bin_id,
                                                                   plots, x_min, x_max, y_min, y_max, n+1,
                                                                   delete_unbinned=True,
                                                                   bin_unbinned=True) # second pass get bins

                        clusterer.bin_filtered(int(args.min_bin_size))
                        clusterer.plot()
                    else:
                        clusterer.rescue_contigs(int(args.min_bin_size))
                else:
                    clusterer.rescue_contigs(int(args.min_bin_size))
            else:
                clusterer.rescue_contigs(int(args.min_bin_size))

        clusterer.write_bins(int(args.min_bin_size))
        imageio.mimsave(clusterer.path + '/UMAP_projections.gif', plots, fps=1)

def vamb(args):
    min_bin_size = int(args.min_size)
    prefix = args.output
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    bins = {}
    with open(args.clusters, 'r') as vamb_file:
        for line in vamb_file:
            line = line.split()
            try:
                bins[line[0]].append(line[1])
            except KeyError:
                bins[line[0]] = [line[1]]

    assembly = SeqIO.to_dict(SeqIO.parse(args.assembly, "fasta"))

    logging.info("Writing bins...")
    max_cluster_id = max(bins.keys())
    for (bin, contigs) in bins.items():
        if bin != -1:
            # Calculate total bin size and check if it is larger than min_bin_size
            bin_length = sum([len(assembly[contig].seq) for contig in contigs])
            if bin_length >= min_bin_size:
                with open(prefix + '/vamb_bin.' + str(bin) + '.fna', 'w') as f:
                    for contig in contigs:
                        write_contig(contig, assembly, f)

        else:
            # Get final bin value
            max_cluster_id += 1
            # Rescue any large unbinned contigs and put them in their own cluster
            for contig in contigs:
                if len(assembly[contig].seq) >= min_bin_size:
                    with open(prefix + '/vamb_bin.' + str(max_cluster_id) + '.fna', 'w') as f:
                        write_contig(contig, assembly, f)


def write_contig(contig, assembly, f):
    seq = assembly[contig]
    fasta = ">" + seq.id + '\n'
    fasta += str(seq.seq) + '\n'
    f.write(fasta)

def phelp():
    print("""
Usage:
flight [SUBCOMMAND] ..

Subcommands:
bin - Bin sets of metagenomic contigs into MAGs
fit - Genotype variants into metagenomic strains *For use with Lorikeet*

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



if __name__ == '__main__':

   sys.exit(main())
