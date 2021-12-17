import io
from os.path import dirname, join
from setuptools import setup
import setuptools


# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def get_version(relpath):
  """Read version info from a file without importing it"""
  for line in io.open(join(dirname(__file__), relpath), encoding="cp437"):
    if "__version__" in line:
      if '"' in line:
        return line.split('"')[1]
      elif "'" in line:
        return line.split("'")[1]


setup(
    name='flight-genome',
    version=get_version("flight/__init__.py"),
    url='https://github.com/rhysnewell/flight',
    license='BSD-3',
    author='Rhys Newell',
    author_email='rhys.newell@hdr.qut.edu.au',
    description='flight - metagenomic binner and variant clusterer.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    package_data={'': [
            "flight/*",
                       ]},
    data_files=[(".", ["README.md", "LICENSE"])],
    include_package_data=True,
    install_requires= [
        "umap-learn >= 0.5",
        "numpy >= 1.17",
        "scikit-learn >= 0.22",
        "scipy >= 1.0",
        "numba >= 0.49",
        "pandas >= 1.3",
        "pynndescent >= 0.5",
        "hdbscan >= 0.8",
        "pebble",
        "threadpoolctl",
        "imageio",
        "matplotlib",
        "seaborn",
        "tqdm",
        "tbb",
        "joblib",
    ],
    entry_points={
          'console_scripts': [
              'flight = flight.flight:main'
          ]
    },
    classifiers=["Topic :: Scientific/Engineering :: Bio-Informatics"],
)
