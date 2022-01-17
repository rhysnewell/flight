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
# from joblib import parallel_backend
#
# parallel_backend("multiprocessing")
# Function imports
import random
import numpy
import faulthandler
faulthandler.enable()

# Self imports


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
    random.seed(42069)
    numpy.random.seed(42069)
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
        default=0.4)

    bin_options.add_argument(
        '--min_dist',
        help=
        'Minimum distance used by UMAP during construction of high dimensional graph',
        dest="min_dist",
        default=0.0)

    bin_options.add_argument('--n_components',
                             help='Dimensions to use in UMAP projection',
                             dest="n_components",
                             default=2)

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

    filter_options = subparsers.add_parser(
        'filter',
        description='Filter out contigs larger than the given base pair threshold',
        formatter_class=CustomHelpFormatter,
        epilog='''
                                        ~ filter ~
            How to use filter:

            flight filter --reference assembly.fasta --min_size 200000

            ''')

    filter_options.add_argument('--reference',
                              help='The assembly file to be binned',
                              dest='assembly')


    filter_options.add_argument('--min_size',
                              help='Minimum bin size',
                              dest='min_size',
                              default=200000)

    filter_options.add_argument('--output',
                              help='The output directory',
                              dest='output',
                              default='filtered_contigs/')

    filter_options.set_defaults(func=filter)
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
    os.environ["NUMBA_NUM_THREADS"] = args.threads
    os.environ["MKL_NUM_THREADS"] = args.threads
    os.environ["OPENBLAS_NUM_THREADS"] = args.threads
    from flight.lorikeet.cluster import Cluster
    from flight.rosella.clustering import Clusterer
    import flight.distance as distance
    import threadpoolctl
    import warnings
    import numpy as np

    with threadpoolctl.threadpool_limits(limits=int(args.threads), user_api='blas'):
        with warnings.catch_warnings():
            if not args.precomputed:
                clusterer = Cluster(
                    args.input,
                    prefix,
                    n_neighbors=int(args.n_neighbors),
                    min_cluster_size=int(args.min_cluster_size),
                    min_samples=int(args.min_samples),
                    min_dist=float(args.min_dist),
                    n_components=int(args.n_components),
                    threads=int(args.threads),
                )
                try:
                    de = distance.ProfileDistanceEngine()
                    stat = de.makeRanksStatVariants(clusterer.clr_depths)
                    clusterer.fit_transform(stat)
                    labels, validities, _, _ = Clusterer.ensemble_cluster_multiple_embeddings(
                        [clusterer.precomputed_reducer_low.embedding_,
                         clusterer.precomputed_reducer_mid.embedding_,
                         clusterer.precomputed_reducer_high.embedding_],
                        top_n=3,
                        metric="euclidean",
                        cluster_selection_methods="eom",
                        solver="hbgf",
                        use_multiple_processes=False
                    )

                    clusterer.labels = labels[0]
                    clusterer.recover_unbinned()
                    clusterer.recover_unbinned()
                    clusterer.recluster()
                    # clusterer.cluster_means = clusterer.get_cluster_means()
                    clusterer.combine_bins()
                    clusterer.plot()
                except ZeroDivisionError:
                    clusterer.labels = np.array([-1 for _ in range(clusterer.clr_depths.shape[0])])

                logging.info("Writing variant labels...")
                numpy.save(prefix + '_labels.npy', clusterer.labels_for_printing())
                logging.info("Calculating cluster separation values...")
                numpy.save(prefix + '_separation.npy', clusterer.separation)
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
                numpy.save(prefix + '_labels.npy', clusterer.labels())



def bin(args):
    prefix = args.output
    # os.environ["NUMEXPR_MAX_THREADS"] = str(max((int(args.threads) // 2 + 1), 1))
    # # os.environ["NUMBA_NUM_THREADS"] = str(min(1, max((int(args.threads) // 2 + 1), 1))) # try and reduce the number of race conditions occurring in numba functions?
    # os.environ["NUMBA_NUM_THREADS"] = str(max((int(args.threads) // 2 + 1), 1)) # try and reduce the number of race conditions occurring in numba functions?
    # os.environ["MKL_NUM_THREADS"] = str(max((int(args.threads) // 2 + 1), 1))
    # os.environ["OPENBLAS_NUM_THREADS"] = str(max((int(args.threads) // 2 + 1), 1))
    os.environ["NUMEXPR_MAX_THREADS"] = str(int(args.threads))
    os.environ["NUMBA_NUM_THREADS"] = str(int(args.threads))  # try and reduce the number of race conditions occurring in numba functions?
    os.environ["MKL_NUM_THREADS"] = str(int(args.threads))
    os.environ["OPENBLAS_NUM_THREADS"] = str(int(args.threads))
    os.environ["OMP_NUM_THREADS"] = str(int(args.threads))
    # os.environ["THREADING_LAYER"] = 'tbb'
    from flight.rosella.rosella import Rosella

    if args.long_input is None and args.input is None:
        logging.warning("bin requires either short or longread coverage values.")
        sys.exit()

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    if not args.precomputed:
        rosella = Rosella(
            count_path=args.input,
            long_count_path=args.long_input,
            kmer_frequencies=args.kmer_frequencies,
            output_prefix=prefix,
            assembly=args.assembly,
            n_neighbors=int(args.n_neighbors),
            min_contig_size=int(args.min_contig_size),
            min_dist=float(args.min_dist),
            threads=int(args.threads),
            a=float(args.a),
            b=float(args.b),
            initialization='spectral'
            )

        rosella.perform_binning(args)



def filter(args):
    """
    Filters out big contigs in an assembly and writes them to individual fasta files
    """
    min_bin_size = int(args.min_size)
    prefix = args.output
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    assembly = SeqIO.to_dict(SeqIO.parse(args.assembly, "fasta"))

    logging.info("Writing bins...")
    bin_id = 0
    for (contig_name, record) in assembly.items():
        if len(record.seq) >= min_bin_size:
            bin_id += 1
            # Rescue any large unbinned contigs and put them in their own cluster
            with open(prefix + '/filtered_contig.' + str(bin_id) + '.fna', 'w') as f:
                fasta = ">" + record.id + '\n'
                fasta += str(record.seq) + '\n'
                f.write(fasta)


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
