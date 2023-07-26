"""    Library for recommender system metrics with huge datasets (Big Data) using Pyspark

Hosted in https://github.com/camiloakv/recmetrics-pyspark
Based on the `recmetrics` library available at https://github.com/statisticianinstilettos/recmetrics

DISCLAIMER: recmetrics-pyspark is not affiliated nor endorsed by recmetrics or its authors.
Some routines have been adapted from recmetrics to work with pySpark DataFrames
and/or to handle bigger datasets. Therefore, some chunks of code have been copied verbatim,
and functions and parameters names have been kept the same (as much as possible) for better usability.

This is a work in progress focused on those metrics which might be troublesome with bigger amounts of data.
Some routines might not be implemented in the foreseen future.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

import pyspark.sql.functions as sf
import pyspark.sql.types as st

def long_tail_plot(df, item_id_column, interaction_type, percentage=None, x_labels=True, xticks_rotation=45):
    """    A minimal adaptation from recmetrics' long_tail_plot to work with pySpark DataFrames.

    Huge dfs can make the first lines of the original implementation unbearable.
    After reducing the data size, we repeat the original implementation from
    recmetrics using pandas on the smaller DataFrame.

    Seaborn library no longer needed, matplotlib used instead.
    FutureWarning removed replacing append for concat.

    Parameters
    ----------
    df: pySpark DataFrame
        must have column `item_id_column`
        
    Check recmetrics' long_tail_plot documentation for other parameters and usage.
    """

    volume_df = df\
      .select([item_id_column])\
      .groupBy(item_id_column).count()\
      .withColumnRenamed("count", "volume")\
      .orderBy(sf.desc("volume"))\
      .toPandas()
    
    # from this point on, code was copied from recmetrics
    volume_df[item_id_column] = volume_df[item_id_column].astype(str)
    volume_df['cumulative_volume'] = volume_df['volume'].cumsum()
    volume_df['percent_of_total_volume'] = volume_df['cumulative_volume']/volume_df['volume'].sum()

    #line plot of cumulative volume
    x = range(0, len(volume_df))
    ax = plt.plot(x, volume_df["volume"], color="black")
    plt.xticks(x)

    #set labels
    plt.title('Long Tail Plot')
    plt.ylabel('# of ' + interaction_type)
    plt.xlabel(item_id_column)

    if percentage != None:
        #plot vertical line at the tail location
        head = volume_df[volume_df["percent_of_total_volume"] <= percentage]
        tail = volume_df[volume_df["percent_of_total_volume"] > percentage]
        items_in_head = len(head)
        items_in_tail = len(tail)
        plt.axvline(x=items_in_head, color="red", linestyle='--')

        # fill area under plot
        head = pd.concat([head, tail.head(1)])
        x1 = head.index.values
        y1 = head['volume']
        x2 = tail.index.values
        y2 = tail['volume']
        plt.fill_between(x1, y1, color="blue", alpha=0.2)
        plt.fill_between(x2, y2, color="orange", alpha=0.2)

        #create legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=str(items_in_head)+': items in the head', markerfacecolor='blue', markersize=5),
            Line2D([0], [0], marker='o', color='w', label=str(items_in_tail)+': items in the tail', markerfacecolor='orange', markersize=5)
        ]
        plt.legend(handles=legend_elements, loc=1)

    else:
        x1 = volume_df[item_id_column]
        y1 = volume_df['volume']
        plt.fill_between(x1, y1, color="blue", alpha=0.3)

    ax = plt.gca()
    if x_labels == False:
        plt.xticks([], [])
        ax.set(xticklabels=[])
    else:
        ax.set_xticklabels(labels=volume_df[item_id_column], rotation=xticks_rotation, ha="right")

    plt.show()


def coverage(df_recommendations, df_ratings, col_item="id_product"):
    """    Return ratio between unique items in df_recommendations and df_ratings (sales)"""

    return len(set(df_recommendations.select("id_product").rdd.flatMap(lambda x: x).collect())) / \
        len(set(df_ratings.select("id_product").rdd.flatMap(lambda x: x).collect()))


def novelty_refac(recommendations, freqs, u, n):
    """    A small refactoring of recmetrics' implementation.
    Parameters renamed: `predicted` to `recommendations`, `pop` to `freqs`
    Parameter n no longer needed, it is extracted from recommendations
    """

    mean_self_information = []
    for sublist in recommendations:
        self_information = sum([-np.log2(freqs[i] / u) for i in sublist])
        mean_self_information.append(self_information / n)
    novelty = sum(mean_self_information) / len(recommendations)

    return novelty, mean_self_information

def novelty_pandas(
    dfp_sales, dfp_recommendations, u=None,
    col_user="person", col_item="product"
):
    """    Similar implementation to novelty_refac but using pandas DataFrames as inputs
    """
    if u is None:
        u = dfp_sales[col_user].nunique()
    dfp_sg = dfp_sales.groupby(col_item).count()
    dfp_sg["log2"] = -np.log2(dfp_sg[col_user] / u)
    dfp_sg = dfp_sg.reset_index().drop(columns=col_user)
    dfp_msi = dfp_recommendations\
        .merge(dfp_sg, on=col_item, how="left")\
        .groupby(col_user)\
        .agg({"log2": ["sum", "count"]})\
        .droplevel(axis=1, level=0)
    dfp_msi["mean_self_information"] = dfp_msi["sum"] / dfp_msi["count"]
    
    return dfp_msi["mean_self_information"].mean(), list(dfp_msi["mean_self_information"])

def novelty(
    df_sales, df_recommendations, u=None,
    col_user="person", col_item="product"
):
    """    Measure the novelty of items present in the recommendations,
    i.e. as compared to the items present on the sales dataset weighted by volume (number of users).
    """
    if u is None:
        u = len(set(df_sales.select(col_user).rdd.flatMap(lambda x: x).collect()))
    df_sg = df_sales.groupBy(col_item).count()
    df_sg = df_sg.withColumn("log2", -sf.log(2., (sf.col("count") / sf.lit(u))))
    df_msi = df_recommendations.join(df_sg, on=col_item, how="left").orderBy([col_user, col_item])
    df_msi = df_msi.select([col_user, "log2"])
    df_msi_sum = df_msi.groupBy(col_user).sum()
    df_msi_count = df_msi.groupBy(col_user).count()
    df_msi2 = df_msi_sum.join(df_msi_count, on=col_user, how="inner")
    df_msi2 = df_msi2\
        .withColumn(
            "mean_self_information",
            sf.col("sum(log2)") / sf.col("count")
        )\
        .orderBy(col_user)
    msis = df_msi2\
        .select("mean_self_information")\
        .rdd.flatMap(lambda x: x).collect()
    
    return sum(msis) / len(msis), msis


# Helper routine with pandas
def get_similarities_from_stacked(df, col_index, col_columns, col_values):
    df_pivot = df\
        .pivot_table(index=col_index, columns=col_columns, values=col_values)\
        .fillna(0)
    return cosine_similarity(sparse.csr_matrix(df_pivot.values))

# Helper routine with pandas
def get_upper_triangle_mean(similarities):
    assert similarities.shape[0] == similarities.shape[1]
    n_rows = similarities.shape[0]
    soma = similarities.sum()
    return 1 - ((soma - n_rows) / (n_rows * (n_rows - 1)))

def personalization_pandas(df, col_index, col_columns, col_values):
    """Used by intra_list_similarities"""
    similarities = get_similarities_from_stacked(df, col_index, col_columns, col_values)
    return get_upper_triangle_mean(similarities)


# Helper routine with pySpark
def cosine_similarity_matrix(df_pivot, columns, index_col):
    """    Return square similarity matrix among elements in index_col
    based on columns of df_pivot
    
    Uses exclusively pyspark objects and methods
    
    Parameters
    ----------
    df_pivot : pyspark.sql.dataframe.DataFrame
        DataFrame with rows indexed by index_col for which similarities
        based on values in columns are to be found
    columns : list
        List of columns of df_pivot to use for calculating similarities
    index_col : string
        Column of df_pivot to use as index of square output matrix
        
    Returns
    -------
    dot : pyspark.mllib.linalg.distributed.BlockMatrix
        Square matrix of cosine similarities
    """
    
    # create features column
    from pyspark.ml.feature import VectorAssembler
    assembler = VectorAssembler(
        inputCols=columns,
        outputCol="features"
    )
    df_pivot_feats = assembler.transform(df_pivot)

    # normalize
    from pyspark.ml.feature import Normalizer
    normalizer = Normalizer(inputCol="features", outputCol="features_norm")
    df_pivot_feats = normalizer.transform(df_pivot_feats)

    # to Dense
    from pyspark.ml.functions import vector_to_array
    df_pivot_feats = df_pivot_feats.withColumn(
        'features_dense_norm',
        vector_to_array('features_norm')
    )

    # calculate similarities
    from pyspark.mllib.linalg.distributed import IndexedRowMatrix
    mat = IndexedRowMatrix(
        df_pivot_feats.select(index_col, "features_dense_norm").rdd
    ).toBlockMatrix()
    dot = mat.multiply(mat.transpose())

    return dot

# Helper routine with pySpark
def vertical_vector(n_rows, default_value=1):
    """    Return a vertical vector (n_rows x 1) filled with default_value
    
    Parameters
    ----------
    n_rows : int
        number of rows of output vector
    
    Returns
    -------
    vvec : pyspark.mllib.linalg.distributed.BlockMatrix
        (n_rows x 1) vector
    
    """

    df_vvec = spark.createDataFrame(
        range(n_rows),
        st.IntegerType()
    ).selectExpr("value as index")
    df_vvec = df_vvec.withColumn("val", sf.lit(default_value))

    from pyspark.ml.feature import VectorAssembler
    assembler = VectorAssembler(
        inputCols=[x for x in df_vvec.columns if x not in ["index"]],
        outputCol="features"
    )
    df_vvec = assembler.transform(df_vvec)

    from pyspark.ml.functions import vector_to_array
    df_vvec = df_vvec.withColumn("features_dense", vector_to_array("features"))

    from pyspark.mllib.linalg.distributed import IndexedRowMatrix
    vvec = IndexedRowMatrix(
        df_vvec.select("index", "features_dense").rdd
    ).toBlockMatrix()
    
    return vvec

# Helper routine with pySpark
def get_symmetrical_matrix_mean(symmetrical_matrix):
    """Return mean of upper diagonal of a symmetrical matrix"""
    
    n_rows = symmetrical_matrix.numCols()
    vv = vertical_vector(n_rows)
    soma = symmetrical_matrix.multiply(vv)
    soma = vv.transpose().multiply(soma)
    soma = soma.toLocalMatrix().toArray()
    soma = soma[0][0]

    return ((soma - n_rows) / (n_rows * (n_rows - 1)))

def personalization(df, col_index, col_columns, version=2):
    """    Return mean personalization ($1-cosine_similarity$) among all different pairs of
    entries in col_index calculating cosine_similarity based on col_columns

    Parameters
    ----------
    df : pyspark.sql.dataframe.DataFrame
        Dataframe containing columns col_index and col_columns with stacked elements
    col_index, col_columns : str
        Column names to be used as index (pairs to be compared) and columns (features)
    version : {0, 1, 2}, optional, default 2
        Which version of the code to use. All three version return the same value, 
        version 2 was tested with bigger dataframes
    
    Returns
    -------
    personalization : float
        Mean personalization between all distinct pairs in col_index
    
    Usage
    -----
    >>> dfx = [(1, 1), 
    ... (1, 2), 
    ... (2, 2), 
    ... (2, 1), 
    ... (3, 3), 
    ... (3, 4) ]
    >>> dfx = spark.createDataFrame(data=dfx, schema=["cod_pessoa", "cod_ean"])
    >>> personalization(
    ...     dfx,
    ...     col_index="cod_pessoa",
    ...     col_columns="cod_ean"
    ... )
    0.666666666666667
    """

    # pivot
    df_pivot = df.select(col_index, col_columns).withColumn("binary", sf.lit(1))
    df_pivot = df_pivot \
        .groupBy(col_index) \
        .pivot(col_columns) \
        .sum("binary") \
        .fillna(0)

    from pyspark.sql.window import Window
    df_pivot = df_pivot.withColumn(
        "index",
        sf.row_number().over(Window.orderBy(sf.monotonically_increasing_id())) - 1
    )
    df_pivot.cache()

    columns=[x for x in df_pivot.columns if x not in [col_index, "index"]]

    if version == 0:
        dot = cosine_similarity_matrix(
            df_pivot,
            columns=columns,
            index_col="index"
        )

        ## expensive even for small datasets!
        #similarity_matrix = dot.toLocalMatrix().toArray()

        personalization = 1 - get_symmetrical_matrix_mean(dot)
        return personalization
    elif version in [1, 2]:
        from pyspark.mllib.linalg.distributed import RowMatrix
        
        from pyspark.ml.feature import VectorAssembler
        assembler = VectorAssembler(
            inputCols=columns,
            outputCol="features"
        )
        df_pivot_feats = assembler.transform(df_pivot)

        from pyspark.ml.feature import Normalizer
        normalizer = Normalizer(inputCol="features", outputCol="features_norm")
        df_pivot_feats = normalizer.transform(df_pivot_feats)

        from pyspark.sql.functions import udf
        from pyspark.mllib.linalg import VectorUDT, DenseVector

        to_dense = udf(lambda x: DenseVector(x), VectorUDT())
        df_pivot_feats = df_pivot_feats.withColumn(
            'features_dense_norm',
            to_dense(sf.col('features_norm'))
        )

        mat = RowMatrix(df_pivot_feats.select("features_dense_norm"))

        from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry

        cm = CoordinateMatrix(
            mat.rows.zipWithIndex().flatMap(
                lambda x: [MatrixEntry(x[1], j, v) for j, v in enumerate(x[0])]
            )
        )
        
        if version == 1:
            bm = cm.toBlockMatrix()
            dot = bm.multiply(bm.transpose())
            personalization = 1 - get_symmetrical_matrix_mean(dot)
            return personalization
        elif version == 2:
            matt = cm.transpose().toRowMatrix()
            exact = matt.columnSimilarities()
            rdd = exact.entries
            soma = rdd.map(lambda x: x.value).sum()
            n_rows = exact.numRows()
            personalization = 1 - (2 * soma / (n_rows * (n_rows-1)))
            return personalization
    else:
        raise ValueError("version {} not implemented.".format(version))


def intra_list_similarities(df, col_group, col_index, col_columns, col_values):
    """    Return intra-list similarities for pandas DataFrame

    The global intra-list similarity can be obtained as the mean of the returned series
    """
    
    s = df\
        .groupby(col_group)\
        .apply(lambda x: personalization_pandas(
            df=x, 
            col_index=col_index,
            col_columns=col_columns,
            col_values=col_values
        ))
    s.name = "intra_list_similarity"
    s = s.reset_index()
    s["intra_list_similarity"] = 1. - s["intra_list_similarity"]
    return s


