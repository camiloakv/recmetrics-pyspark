# recmetrics-pyspark: recommender systems metrics for big data

**recmetrics-pyspark** obtains the most relevant internal metrics for items recommendations from pySpark DataFrames. It efficiently handles huge amounts of data. Most routines are adapted from the **recmetrics** library which works with pandas DataFrames. 

DISCLAIMER: recmetrics-pyspark is not affiliated nor endorsed by recmetrics or its authors.
Some routines have been adapted from recmetrics to work with pySpark DataFrames
and/or to handle bigger datasets. Therefore, some chunks of code have been copied verbatim,
and functions and parameters names have been kept the same (as much as possible) for better usability.

Furthermore, if you are dealing with small datasets, we recommend to use the recmetrics library (https://github.com/statisticianinstilettos/recmetrics) instead, as it most efficiently handles smaller datasets.

## Where to get it
The source code is currently hosted on GitHub at:
https://github.com/camiloakv/recmetrics-pyspark

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/recmetrics-pyspark).

```sh
pip install recmetrics-pyspark
```

Available metrics as of version 0.0.1:

 - `long_tail_plot`
 - `coverage`
 - Novelty:
   - `novelty_refac` A small refactoring of recmetrics' implementation.
   - `novelty_pandas` Similar implementation to novelty_refac but using pandas DataFrames as inputs
   - `novelty` pySpark implementation
 - `personalization`
 - `intra_list_similarities`
