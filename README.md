# Flight
The python component used by [Lorikeet](https://github.com/rhysnewell/Lorikeet) and [Rosella](https://github.com/rhysnewell/rosella)

# Installation
This module is only used in conjunction with other programs, but if you wish to install and play around with the code 
follow these instructions:

```
git clone https://github.com/rhysnewell/flight.git
cd flight
conda env create -n flight -f flight.yml
conda activate flight
pip install .
flight bin --help
```

# Requirements

Initial requirements for flight can be downloaded using the `flight.yml`:
```
conda env create -n flight -f flight.yml
```

# Usage

To perform mag recovery:
```
flight bin --assembly scaffolds.fasta --input coverm.cov --output output_dir/ --threads 24
```
