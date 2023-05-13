import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from numba import jit, cuda
from timeit import default_timer as timer
from TrendAnalysis.trends import data_loader
from HydrographAnalysis.hydrograph import spatial_stability, temporal_stability, colors, colors_for_boxplot, \
    reference_hydrograph


@jit(target_backend='cuda')
def correlation_computation(store_pca_vec, store_scores, num_candidate_pca):
    # store_pca_vec or store_scores
    num_iter = len(store_pca_vec)

    # correlation matrix (num_em, number of candidate pca)
    # num_em = num_iter * (num_iter - 1) / 2  # arithmetic sum
    num_em = int(num_iter * (num_iter - 1) / 2)
    corr_matrix_spatial = np.zeros((num_em, num_candidate_pca))
    corr_matrix_temporal = np.zeros((num_em, num_candidate_pca))

    # loop
    # e.g.: store_vectors = [a, b, c,d] -> C(a,b), C(a,c), C(a,d), C(b,c), C(b,d), C(c,d)
    p = 0  # index of num_em
    for _iter in range(num_iter):
        x_tot_spat = store_pca_vec[_iter]
        x_tot_temp = store_scores[_iter]

        for k_iter in range(_iter + 1, num_iter):
            y_tot_spat = store_pca_vec[k_iter]
            y_tot_temp = store_scores[k_iter]

            # loop on pca
            for j in range(num_candidate_pca):
                x_spat = x_tot_spat[:, j]
                y_spat = y_tot_spat[:, j]

                x_temp = x_tot_temp[:, j]
                y_temp = y_tot_temp[:, j]

                corr_spat = np.corrcoef(x_spat, y_spat)[0, 1]
                corr_spat = corr_spat ** 2

                corr_temp = np.corrcoef(x_temp, y_temp)[0, 1]
                corr_temp = corr_temp ** 2

                # store
                corr_matrix_spatial[p, j] = corr_spat
                corr_matrix_temporal[p, j] = corr_temp

            p += 1

    return corr_matrix_spatial, corr_matrix_temporal


def stability(head, perc=0.7, num_iter=200, threshold=1.0):
    # spatial stability of pca
    print('Spatial stability analysis: started...')
    store_pca_vec, spatial_min_num_pca = spatial_stability(head, perc, num_iter, threshold)
    print('Spatial stability analysis: ended... min pca: %d' % (spatial_min_num_pca,))

    # temporal stability of pca
    print('Temporal stability analysis: started...')
    store_scores, temporal_min_num_pca = temporal_stability(head, perc, num_iter, threshold)
    print('Temporal stability analysis: ended... min pca: %d' % (temporal_min_num_pca,))

    start = timer()
    # get min(spatial_min_num_pca, temporal_min_num_pca)
    num_candidate_pca = np.minimum(spatial_min_num_pca, temporal_min_num_pca)

    # pearson correlation between pca_vec (spatial stability) for all combinations
    # between scores (temporal stability) for all combinations
    corr_matrix_spatial, corr_matrix_temporal = correlation_computation(store_pca_vec, store_scores, num_candidate_pca)

    print("With/without GPU:", timer() - start)
    return corr_matrix_spatial, corr_matrix_temporal


def plot_on_one_axis(ax, data: list, text: str):
    num_pca = len(data)
    labels = ['PC' + str(i) for i in range(1, num_pca + 1)]
    boxes = ax.boxplot(x=data,
                       labels=labels,
                       patch_artist=True,
                       flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': colors_for_boxplot['black']},
                       whiskerprops=dict(color=colors_for_boxplot['black']),
                       medianprops=dict(color=colors_for_boxplot['black']))

    for patch, k in zip(boxes['boxes'], colors):
        patch.set(facecolor=k)

    # format
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        if spine == 'top':
            ax.spines[spine].set_visible(False)

        if spine == 'right':
            ax.spines[spine].set_edgecolor(colors_for_boxplot['gray'])

    for spine in ['bottom', 'left']:
        ax.spines[spine].set_edgecolor(colors_for_boxplot['gray'])
        ax.spines[spine].set_edgecolor(colors_for_boxplot['gray'])

    ax.set_ylabel('$ R^2 $')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.text(0.1, 0.1, text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            fontweight='bold')

    ax.set_ylim(-0.01, 1.0)


def execute_stable_pca(perc=0.7, num_iter=200, threshold=1.0, fs=16):
    # save path
    save_path = 'Results/'

    # load data
    base_path = 'Data/'
    well_path = base_path + 'wells.xlsx'
    head_path = base_path + 'head.xlsx'
    t, well_ids, head, info_on_wells = data_loader(well_path, head_path=head_path, climate_path=None,
                                                   isClimateVars=False, resample_op='mean', toResample=False,
                                                   freq='2D', toRandomize=False, returnHeadOnly=False)

    # spatial and temporal stability
    corr_matrix_spatial, corr_matrix_temporal = stability(head=head, perc=perc, num_iter=num_iter, threshold=threshold)

    # compute median and save
    median_spatial = np.median(corr_matrix_spatial, axis=0).reshape((1, -1))
    median_temporal = np.median(corr_matrix_temporal, axis=0).reshape((1, -1))

    num_pca = median_spatial.shape[1]
    columns = ['PC' + str(i) for i in range(1, num_pca + 1)]

    median_spatial = pd.DataFrame(median_spatial, columns=columns)
    median_temporal = pd.DataFrame(median_temporal, columns=columns)

    median_spatial.to_excel(save_path + 'median_spatial.xlsx')
    median_temporal.to_excel(save_path + 'median_temporal.xlsx')

    # prepare data for boxplot
    spatial_lst = [corr_matrix_spatial[:, i] for i in range(corr_matrix_spatial.shape[1])]
    temporal_lst = [corr_matrix_temporal[:, i] for i in range(corr_matrix_temporal.shape[1])]

    plt.rcParams.update({'font.size': fs, 'font.family': 'Arial'})
    fig, [ax, ay] = plt.subplots(nrows=2, ncols=1, constrained_layout=True, figsize=(15, 15))
    plot_on_one_axis(ax, spatial_lst, '(a)')
    plot_on_one_axis(ay, temporal_lst, '(b)')
    plt.tight_layout()
    plt.savefig(save_path + 'stability.png', dpi=600)


def execute_multi_linear_regression_from_stable_pca(num_pca=2, fs=16):
    # save path
    save_path = 'Results/'

    # load data
    base_path = 'Data/'
    well_path = base_path + 'wells.xlsx'
    head_path = base_path + 'head.xlsx'
    t, well_ids, head, info_on_wells = data_loader(well_path, head_path=head_path, climate_path=None,
                                                   isClimateVars=False, resample_op='mean', toResample=False,
                                                   freq='2D', toRandomize=False, returnHeadOnly=False)

    reference_hydrograph(save_path, t, well_ids, head, info_on_wells, num_pca=num_pca, fs=fs)


# first, get stable pca - num_iter=10000
execute_stable_pca(perc=0.7, num_iter=1000, threshold=1.0, fs=16)

# second, get and plot reference hydrograph and residual
execute_multi_linear_regression_from_stable_pca(num_pca=2, fs=18)
