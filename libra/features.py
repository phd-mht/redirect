import json
import os
import pickle
import sys
import uuid

import numpy as np
import pandas as pd
import igraph as ig

from pyspark.sql import functions as sf
from pyspark.sql import types as st

from common import reset_multi_proc_staging, MULTI_PROC_STAGING_LOCATION


SCHEMA_FEAT_UDF = st.StructType([st.StructField("features", st.StringType())])


def get_segments(source_column, target_column, data_in):
    sources = set(data_in[source_column].unique())
    targets = set(data_in[target_column].unique())
    source_or_target = sources.union(targets)
    source_and_target = sources.intersection(targets)
    source_only = sources.difference(targets)
    target_only = targets.difference(sources)
    return sources, targets, source_or_target, source_and_target, source_only, target_only


def generate_features(df, row, graph_features=False):
    # TODO: This can be made much faster!
    sources, targets, source_or_target, source_and_target, source_only, target_only = get_segments(
        "source", "target", df
    )
    node_name = row["key"]
    features_row = {
        "key": node_name,
        "num_sources": len(sources),
        "num_targets": len(targets),
        "num_source_or_target": len(source_or_target),
        "num_source_and_target": len(source_and_target),
        "num_source_only": len(source_only),
        "num_target_only": len(target_only),
        "num_transactions": df["num_transactions"].sum(),
    }

    agg = {"amount": "sum", "amount_weighted": "sum"}
    columns = ["amount", "amount_weighted"]
    left = df.loc[:, ["target"] + columns].rename(columns={"target": "source"}).groupby("source").agg(agg)
    features_row["max_credit_edges"] = np.max(left["amount"])
    features_row["mean_credit_edges"] = np.mean(left["amount"])
    features_row["median_credit_edges"] = np.median(left["amount"])
    features_row["std_credit_edges"] = np.std(left["amount"])
    features_row["max_credit_edges_weighted"] = np.max(left["amount_weighted"])
    features_row["mean_credit_edges_weighted"] = np.mean(left["amount_weighted"])
    features_row["median_credit_edges_weighted"] = np.median(left["amount_weighted"])
    features_row["std_credit_edges_weighted"] = np.std(left["amount_weighted"])
    
    right = df.loc[:, ["source"] + columns].groupby("source").agg(agg)
    features_row["max_debit_edges"] = np.max(right["amount"])
    features_row["mean_debit_edges"] = np.mean(right["amount"])
    features_row["median_debit_edges"] = np.median(right["amount"])
    features_row["std_debit_edges"] = np.std(right["amount"])
    features_row["max_debit_edges_weighted"] = np.max(right["amount_weighted"])
    features_row["mean_debit_edges_weighted"] = np.mean(right["amount_weighted"])
    features_row["median_debit_edges_weighted"] = np.median(right["amount_weighted"])
    features_row["std_debit_edges_weighted"] = np.std(right["amount_weighted"])
    
    result = left.join(right, how="outer", lsuffix="_left").fillna(0).reset_index()
    result.loc[:, "delta"] = result["amount_left"] - result["amount"]
    turnover = float(result[result["delta"] > 0]["delta"].sum())
    features_row["turnover"] = turnover

    if graph_features:
        graph = ig.Graph.DataFrame(df[["source", "target"]], use_vids=False, directed=True)
        features_row["assortativity_degree"]= graph.assortativity_degree(directed=True)
        features_row["assortativity_degree_ud"] = graph.assortativity_degree(directed=False)
        features_row["max_degree"] = max(graph.degree(mode="all"))
        features_row["max_degree_in"] = max(graph.degree(mode="in"))
        features_row["max_degree_out"] = max(graph.degree(mode="out"))
        features_row["diameter"] = graph.diameter(directed=True, unconn=True)
        features_row["diameter_ud"] = graph.diameter(directed=False, unconn=True)
        features_row["density"] = graph.density(loops=False)
        biconn_components, articulation_points = graph.biconnected_components(return_articulation_points=True)
        features_row["num_biconn_components"] = len(biconn_components) 
        features_row["num_articulation_points"] = len(articulation_points)

    return features_row


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def generate_features_udf_wrapper(graph_features):
    def generate_features_udf(df):
        row = df.iloc[0]
        features = json.dumps(
            generate_features(df, row, graph_features=graph_features),
            allow_nan=True, cls=NpEncoder,
        )
        return pd.DataFrame([{"features": features}])
    return generate_features_udf


def generate_features_spark(communities, graph, spark):
    reset_multi_proc_staging()
    chunk_size = 100_000
    
    df_comms = []
    partitions = 0
    for index, (node, comm) in enumerate(communities):
        sub_g = graph.induced_subgraph(comm)
        df_comm = sub_g.get_edge_dataframe()
        if not df_comm.empty:
            df_comm.loc[:, "key"] = node
            df_comms.append(df_comm)
        if not ((index + 1) % chunk_size):
            partitions += 6
            pd.concat(df_comms, ignore_index=True).to_parquet(f"{MULTI_PROC_STAGING_LOCATION}{os.sep}{index + 1}.parquet")
            df_comms = []
    
    if len(df_comms) > 0:
        partitions += 6
        pd.concat(df_comms, ignore_index=True).to_parquet(f"{MULTI_PROC_STAGING_LOCATION}{os.sep}{index + 1}.parquet")
    
    del df_comms

    response = spark.read.parquet(
        str(MULTI_PROC_STAGING_LOCATION)
    ).repartition(partitions, "key").groupby("key").applyInPandas(
        generate_features_udf_wrapper(True), schema=SCHEMA_FEAT_UDF
    ).toPandas()
    
    return pd.DataFrame(response["features"].apply(json.loads).tolist())  # .astype(FEATURE_TYPES)
