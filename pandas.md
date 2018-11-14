# Pandas
For kernel-only competitions you need to think about optimizing memory usage - the
kernels are only permitted to use 4GB of memory in total.

Pandas allows us to do this in many different ways.

For instance: a subset of your data:
 - `nrows`: How many rows to read
 - `skiprows`: Offset to read from
 - `chunksize`: Lazy-load data when looking through the table.
 - `dtype`: Data type

Another thing you can do is get a subset (`set_subset`). This gets you a subset of the columns.

Scientific notation: You can suppress this as well.

One "brute-force" way of dealing with this is that we can loop over all the values
in a pandas table and then re-encode them as some smaller datatype (eg, long64 -> int8).

This works surprisingly well - compresses by about 70%

*pandas_profiling*: Very useful module. Can show you all the highly correlated data. Use

```py
pandas_profiling.ProfileReport(df)
```

You can also find missing values using the `missingno` package:

```py
import missingno
missingno.matrix(df)
```

There's lots of different ways to fill in missing values, eg, interpolation, filling them in with the
previous/next non-missing values in the dataframe.

You can select certain features by their column names `df['foo', 'bar', 'baz']`

You can also look at correlation: `df.corr()`. In practice, if we see that two features
are correlated, we can reduce the dimensionality of our data because the correlation means that
some data is not giving us any more useful information.

How do we replace as the `nan` values? Use `df.fillna(0, inplace=True)`

To aggregate features, use `.groupby`. Now, just like SQL we can take certain features for our
grouped data and then calculate summary statistics: `df.groupby(['fooId'])['field1'].mean()

Seeing unique values of a feature: `df['foo'].unique()`. Useful for finding categories.

Transformations: `df['column'].apply(func)`

Plotting: Use `df.plot()` (or on any subset of columns). Allows us to plot really
quickly without stuffing about with matplotlib.

Cluter Map: Plots a diagram showing a tree of correlations.
