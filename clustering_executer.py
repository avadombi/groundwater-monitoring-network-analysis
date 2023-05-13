import pandas as pd
import numpy as np
from TrendAnalysis.trends import data_loader
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from Clustering.time_series_clustering import cluster_time_series


def get_data(toResample=False, freq='2D'):
    # paths
    base_path = 'Data/'
    well_path = base_path + 'wells.xlsx'
    head_path = base_path + 'head.xlsx'

    # heads
    t, well_ids, head, info_on_wells = data_loader(well_path, head_path=head_path, climate_path=None,
                                                   isClimateVars=False, resample_op='mean', toResample=toResample,
                                                   freq=freq, toRandomize=False, returnHeadOnly=False)

    # numpy to a list of time series
    head = list(head.T)

    # Time series format (for tslearn)
    head = to_time_series_dataset(head)

    # scale (mean - variance)
    head = TimeSeriesScalerMeanVariance().fit_transform(head)
    return well_ids, info_on_wells, head


def execute_clustering_for_a_given_number_of_clusters(toResample=False, freq='2D', n_clusters=2, metric='dtw'):
    # save path
    save_path = 'Results/clusters_for_%d_clusters.xlsx' % (n_clusters,)
    well_ids, info_on_wells, head = get_data(toResample=toResample, freq=freq)

    # cluster the time series and save the result
    cluster_time_series(well_ids, head, info_on_wells, save_path, n_clusters=n_clusters, metric=metric)


def execute_clustering_for_a_given_max_number_of_clusters(toResample=False, freq='2D', n_max_clusters=10, metric='dtw'):
    # save path
    save_path = 'Results/clusters_for_various_no_clusters.xlsx'
    well_ids, info_on_wells, head = get_data(toResample=toResample, freq=freq)

    # browse over n_max_clusters
    outputs = pd.DataFrame(columns=['n_clusters', 'sil'])
    n_clusters_list = list(range(2, n_max_clusters + 1))
    sil_list = list()

    for n_clusters in n_clusters_list:
        sil_score = cluster_time_series(well_ids, head, info_on_wells, save_path, n_clusters=n_clusters, metric=metric,
                                        toSave=False)

        sil_list.append(np.round(sil_score, 3))

    # results as Excel file
    outputs['n_clusters'] = n_clusters_list
    outputs['sil'] = sil_list

    # save
    outputs.to_excel(save_path)


execute_clustering_for_a_given_number_of_clusters(toResample=True, freq='5D', n_clusters=2, metric='dtw')
execute_clustering_for_a_given_max_number_of_clusters(toResample=True, freq='5D', n_max_clusters=5, metric='dtw')
