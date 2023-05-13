import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans, silhouette_score


def cluster_time_series(well_ids, head, info_on_wells, save_path, n_clusters=2, metric='dtw', toSave=True):
    # set a seed
    seed = 0

    # cluster the time series
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, verbose=True, random_state=seed)
    labels = km.fit_predict(head)
    sil_score = silhouette_score(head, labels, metric=metric)

    # results as Excel file
    outputs = pd.DataFrame(columns=['id', 'x', 'y', 'aq', 'cluster (sil: ' + str(np.round(sil_score, 2)) + ')'])
    outputs['id'] = well_ids
    outputs['cluster (sil: ' + str(np.round(sil_score, 2)) + ')'] = labels + 1

    # get x and y
    x, y, aq_ls = [], [], []
    for name in well_ids:
        coord = info_on_wells[info_on_wells['id'] == name][['x', 'y', 'confinement state']]
        x.append(coord['x'].values[0])
        y.append(coord['y'].values[0])
        aq_ls.append(coord['confinement state'].values[0])

    outputs['x'] = x
    outputs['y'] = y
    outputs['aq'] = aq_ls

    if toSave:
        # save
        outputs.to_excel(save_path)
    else:
        return sil_score



