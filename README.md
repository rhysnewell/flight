# Flock
The python component used by [Lorikeet](https:/github.com/rhysnewell/Lorikeet) and [Rosella](https:/github.com/rhysnewell/rosella)
It can be used as a standalone binning algorithm for MAGs, but will not include contig SNP information
which is collected by Rosella.
# Installation
This module is only used in conjunction with other programs, but if you wish to install and play around with the code 
follow these instructions:

```
git clone https://github.com/rhysnewell/flock.git
cd flock
conda env create -n flock -f flock.yml
conda activate flock
pip install .
flock bin --help
```

# Requirements

Initial requirements for binsnek can be downloaded using the `binsnek.yml`:
```
conda env create -n rosella -f rosella.yml
```

# Usage

To perform mag recovery:
```
flock bin --assembly scaffolds.fasta --input coverm.cov --output output_dir/ --threads 24
```
