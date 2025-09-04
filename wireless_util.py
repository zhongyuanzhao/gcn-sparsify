import os
import sys
import numpy as np
import pandas as pd


def load_ecdf_wts(wts_sample_file, quantiles=np.arange(0., 1, 0.1)):
    df = pd.read_csv(wts_sample_file, index_col=False)
    df.rename(columns={df.columns[0]: "value"}, inplace=True)
    df_stats = df.quantile(quantiles)
    df_stats.reset_index(inplace=True)
    df_stats['index'] = df_stats['index'].round(3)
    df_stats.set_index('index', drop=True, inplace=True)
    dict_stats = df_stats['value'].round(1).to_dict()
    return dict_stats, df


def load_local_ecdf(local_dist, adj_0, quantiles=np.arange(0., 1, 0.1)):
    adj = adj_0.copy()
    graph_size = adj.shape[0]
    # df = pd.DataFrame(columns=quantiles)
    nbhd_ecdf = np.zeros((graph_size, quantiles.size))
    for v in range(graph_size):
        _, neighbors = np.nonzero(adj[v])
        neighborhood = np.append(neighbors, v)
        tmp = local_dist[:, neighborhood]
        qt_v = np.nanquantile(tmp, quantiles)
        nbhd_ecdf[v, :] = qt_v
    local_ecdf = np.transpose(np.nanquantile(local_dist, quantiles, axis=0))
    return nbhd_ecdf, local_ecdf, quantiles

