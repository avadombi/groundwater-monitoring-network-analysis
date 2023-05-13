import matplotlib.pyplot as plt  # for plotting
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

colors_for_boxplot = {
    'black': '#000000',
    'blue': '#3275a1',
    'orange': '#e1802c',
    'green': '#3a923a',
    'red': '#c03d3d',
    'purple': '#9372b2',
    'gray': '#c4c4c4',
    'c1': '#42c2f5',
    'c2': '#f59c42',
    'c3': '#ff1717'
}
colors = [
    colors_for_boxplot['blue'],
    colors_for_boxplot['orange'],
    colors_for_boxplot['green'],
    colors_for_boxplot['red'],
    colors_for_boxplot['purple'],
    colors_for_boxplot['gray'],
    colors_for_boxplot['c1'],
    colors_for_boxplot['c2']
]


def scaler(data):
    scale = StandardScaler()
    return scale.fit_transform(data)


def pca_analyzer(sc_data, threshold=1.0):
    # initialize pca model
    pca_model = PCA()

    # execute pca
    pca_outs = pca_model.fit(sc_data)

    # select pca satisfying the unit threshold of ev
    num_pca = pca_with_ev_greater_than_one(pca_outs, threshold=threshold)

    # redo the pca with the number of pca satisfying the threshold condition
    pca_model = PCA(n_components=num_pca)
    pca_outs = pca_model.fit(sc_data)

    return pca_outs


def pca_with_ev_greater_than_one(pca_outs, threshold=1.0):
    ev = get_eigen_values(pca_outs)
    idx = np.argwhere(ev > threshold)
    num_pca = len(idx)
    return num_pca


def plot_cumulative_variance(pca_outs, toSavePlot, save_path='../../GWLSimilarity/PCA/Plots/'):
    perc_cvar = np.cumsum(pca_outs.explained_variance_ratio_)
    x = np.arange(start=1, stop=perc_cvar.shape[0] + 1, step=1)

    plt.plot(x, perc_cvar, marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')

    if toSavePlot:
        plt.savefig(save_path + 'c_variance.png', dpi=300)
    else:
        plt.show()


def get_number_main_pcs(pca_outs, perc_criterion=0.9):
    perc_cvar = np.cumsum(pca_outs.explained_variance_ratio_)
    idx = (np.abs(perc_cvar - perc_criterion)).argmin()
    return idx + 1  # number of pcs that explain 'perc_criterion' % of variance


def get_singular_values(pca_outs):
    # sqrt of eigen values (a.k.a. standard deviation sii)
    return pca_outs.singular_values_


def get_eigen_values(pca_outs):
    return pca_outs.explained_variance_


def get_pca_components(pca_outs):
    # components (number of time series or pz, number of retained pca)
    # note: for each component, the elements are called 'loadings'
    components = pca_outs.components_
    components = np.transpose(components)
    return components


def get_scores(sc_head, components):
    # sc_head: (number of timesteps, number of pz)
    # components: (number of pz, number of retained pca)
    # score: (number of timesteps, number of retained pca)
    return np.matmul(sc_head, components)


def get_correlation_matrix(sc_head, scores):
    # sc_head: (number of timesteps, number of pz) -> sc_head(n, p)
    # scores : (number of timesteps, number of retained pca) -> cp(n, k)

    num_ts, num_pz = sc_head.shape
    _, num_cp = scores.shape

    # corr_matrix
    corr_matrix = np.zeros((num_pz, num_cp))

    for p in range(num_pz):
        for k in range(num_cp):
            x, y = sc_head[:, p], scores[:, k]
            corr = np.corrcoef(x, y)[0, 1]
            corr_matrix[p, k] = corr ** 2
    return corr_matrix


def run_one_pca(head, threshold=1.0):
    # scale data
    sc_head = scaler(head)

    # apply pca such that the only pca that have ev greater than 'threshold' are selected
    pca_outs = pca_analyzer(sc_head, threshold=threshold)

    # pca components (eigen vectors) -> (p, k) where p: number of columns, k: number of pca eigen vectors
    pca_vec = get_pca_components(pca_outs)

    # scores -> (n, k) where p: number of timesteps, k: number of pca eigen vectors
    scores = get_scores(sc_head, pca_vec)

    return pca_vec, scores


def random_sampling_time_axis(orig_head, perc=0.7):
    # random sampling of the orig_head
    size = orig_head.shape[0]
    num_elements = int(size * perc)

    # random indexes
    idx = np.random.randint(size, size=num_elements)

    # random head
    rd_head = orig_head[idx, :]
    return rd_head


def random_sampling_column_axis(orig_head, perc=0.7):
    # random sampling of the orig_head
    size = orig_head.shape[1]
    num_elements = int(size * perc)

    # random indexes
    idx = np.random.randint(size, size=num_elements)

    # random head
    rd_head = orig_head[:, idx]
    return rd_head


def spatial_stability(orig_head, perc=0.7, num_iter=10000, threshold=1.0):
    # variable to store the data of each iteration (pca_vec)
    store_pca_vec = []

    # identify the minimum number of pca for the ensemble of 'num_iter' iterations
    min_num_pca = 0

    for _iter in range(num_iter):
        # random sampling
        rd_head = random_sampling_time_axis(orig_head, perc)

        # execute the pca
        pca_vec, scores = run_one_pca(rd_head, threshold=threshold)

        # store pca_vec
        store_pca_vec.append(pca_vec)

        if _iter == 0:
            min_num_pca = pca_vec.shape[1]
        else:
            current_num_pca = pca_vec.shape[1]
            if min_num_pca > current_num_pca:
                min_num_pca = current_num_pca

    return store_pca_vec, min_num_pca


def temporal_stability(orig_head, perc=0.7, num_iter=10000, threshold=1.0):
    # variable to store the data of each iteration (pca_vec)
    store_scores = []

    # identify the minimum number of pca for the ensemble of 'num_iter' iterations
    min_num_pca = 0

    for _iter in range(num_iter):
        # random sampling
        rd_head = random_sampling_column_axis(orig_head, perc)

        # execute the pca
        pca_vec, scores = run_one_pca(rd_head, threshold=threshold)

        # store pca_vec
        store_scores.append(scores)

        if _iter == 0:
            min_num_pca = pca_vec.shape[1]
        else:
            current_num_pca = pca_vec.shape[1]
            if min_num_pca > current_num_pca:
                min_num_pca = current_num_pca

    return store_scores, min_num_pca


def pca_analyzer_for_known_number_of_stale_pca(sc_data, num_pca=2):
    # initialize pca model
    pca_model = PCA(n_components=num_pca)

    # execute pca
    pca_outs = pca_model.fit(sc_data)
    return pca_outs


def get_pca_scores(sc_head, pca_outs):
    # scores -> (n, k) where p: number of timesteps, k: number of pca eigen vectors
    pca_vec = np.transpose(pca_outs.components_)
    scores = np.matmul(sc_head, pca_vec)
    return scores, pca_vec


def plot_scores(t, scores, save_path, fs=16, lw=2):
    num_pca = scores.shape[1]

    plt.rcParams.update({'font.size': fs, 'font.family': 'Arial'})
    fig, axes = plt.subplots(nrows=num_pca, ncols=1, constrained_layout=True, figsize=(20, 10))

    for p in range(num_pca):
        if num_pca == 1:
            ax = axes
        else:
            ax = axes[p]

        ax.axhline(y=0.0, color=colors_for_boxplot['gray'], linestyle='-', label='_Hidden', linewidth=lw)
        ax.plot(t, scores[:, p], color=colors_for_boxplot['black'], linewidth=lw)
        ax.set_ylabel('PC%d' % (p + 1,))

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

    plt.savefig(save_path + 'scores.png', dpi=350)
    plt.clf()


def mlr_one_ts(scores, head):
    # initialize and fit the model with the input data
    reg_model = linear_model.LinearRegression()
    reg_model.fit(X=scores, y=head)

    # get coefficients: head_sim = w0 + w1 * sc1 + ... + wp * scp
    # w = (w1, ..., wp): coefficients and b = w0: intercept

    w = reg_model.coef_
    b = reg_model.intercept_

    # grouped w and b as a list
    params = list(w) + [b]

    # ySim
    head_sim = reg_model.predict(scores)
    return params, head_sim


def add_coordinates_to_df(df, info_on_wells):
    # loop
    num_pz = df.shape[0]
    for pz in range(num_pz):
        name_target = df['id'].iloc[pz]
        info = info_on_wells[info_on_wells['id'] == name_target]

        x, y = info['x'].values[0], info['y'].values[0]

        df['x'].iloc[pz] = x
        df['y'].iloc[pz] = y

    return df


def plot_heads(t, yObs, yRef, yRes, rRef, rRes, save_path, name, fs=16, lw=2):
    plt.rcParams.update({'font.size': fs, 'font.family': 'Arial'})
    fig, [ax, ay] = plt.subplots(nrows=2, ncols=1, constrained_layout=True, figsize=(20, 10))

    # yObs abd yRef (#ff1717)
    ax.plot(t, yObs, color=colors_for_boxplot['black'], linewidth=lw)
    ax.plot(t, yRef, color=colors_for_boxplot['c3'], linestyle='--', linewidth=lw)
    ax.set_ylabel('Groundwater level (m)')
    ax.legend(['Observed GWL', 'Reference GWL $ R^2 $ = %.2f' % (rRef,)], ncol=2, frameon=False)
    ax.text(0.02, 0.5, '(a)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            fontweight='bold')

    # yRes
    ay.axhline(y=0.0, color=colors_for_boxplot['gray'], linestyle='-', label='_Hidden', linewidth=lw)
    ay.plot(t, yRes, color=colors_for_boxplot['c3'], linewidth=lw, label='Residuals of (a) $ R^2 $ = %.2f' % (rRes,))
    ay.set_ylabel('Residuals (m)')
    ay.legend(frameon=False)
    ay.text(0.02, 0.5, '(b)', horizontalalignment='center', verticalalignment='center', transform=ay.transAxes,
            fontweight='bold')

    # format
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        ay.spines[spine].set_visible(False)
        if spine == 'top':
            ax.spines[spine].set_visible(False)
            ay.spines[spine].set_visible(False)

        if spine == 'right':
            ax.spines[spine].set_edgecolor(colors_for_boxplot['gray'])
            ay.spines[spine].set_edgecolor(colors_for_boxplot['gray'])

    for spine in ['bottom', 'left']:
        ax.spines[spine].set_edgecolor(colors_for_boxplot['gray'])
        ax.spines[spine].set_edgecolor(colors_for_boxplot['gray'])

        ay.spines[spine].set_edgecolor(colors_for_boxplot['gray'])
        ay.spines[spine].set_edgecolor(colors_for_boxplot['gray'])

    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(save_path + name + '.png', dpi=350)
    plt.clf()


def reference_hydrograph(save_path, t, well_ids, head, info_on_wells, num_pca=2, fs=16):
    # scale data
    sc_head = scaler(head)

    # pca analysis
    pca_outs = pca_analyzer_for_known_number_of_stale_pca(sc_head, num_pca=num_pca)

    # get scores
    scores, pca_vec = get_pca_scores(sc_head, pca_outs)

    # plot
    plot_scores(t, scores, save_path, fs=fs)

    # get reference hydrograph
    num_pz = head.shape[1]
    columns = ['id', 'x', 'y', 'rRef', 'rRes']
    rCoef = pd.DataFrame(np.zeros((num_pz, 5)), columns=columns)
    rCoef['id'] = rCoef['id'].astype('str')
    rCoef['id'] = well_ids

    ## fill id, x, y of rCoef
    rCoef = add_coordinates_to_df(rCoef, info_on_wells)

    index = ['w%d' % (p,) for p in range(1, num_pca + 1)] + ['b']
    outs_params = pd.DataFrame(np.zeros((num_pca + 1, num_pz)), columns=well_ids, index=index)

    num_timesteps = head.shape[0]
    out_yRef = pd.DataFrame(np.zeros((num_timesteps, num_pz + 1)), columns=['t'] + well_ids.tolist())
    out_yRes = pd.DataFrame(np.zeros((num_timesteps, num_pz + 1)), columns=['t'] + well_ids.tolist())

    out_yRef['t'] = out_yRef['t'].astype('str')
    out_yRes['t'] = out_yRes['t'].astype('str')

    # df_pca_vec -> (id, x, y, pc1, pc2))
    columns = ['id', 'x', 'y'] + ['pc%d' % (k,) for k in range(1, num_pca + 1)]
    df_pca_vec = pd.DataFrame(np.zeros((num_pz, num_pca + 3)), columns=columns)
    df_pca_vec['id'] = df_pca_vec['id'].astype('str')
    df_pca_vec['id'] = well_ids

    for k in range(num_pca):
        df_pca_vec['pc%d' % (k + 1,)] = pca_vec[:, k]

    df_pca_vec = add_coordinates_to_df(df_pca_vec, info_on_wells)

    for pz in range(num_pz):
        yObs = head[:, pz]

        # regression
        params, yRef = mlr_one_ts(scores, yObs)

        # residual
        yRes = yObs - yRef

        # pearson correlation coef. (yObs, yRef) and (yObs, yRes)
        rRef = np.corrcoef(yObs, yRef)[0, 1] ** 2
        rRes = np.corrcoef(yObs, yRes)[0, 1] ** 2

        rRef, rRes = np.round(rRef, 2), np.round(rRes, 2)

        # store
        rCoef.iloc[pz, -2:] = [rRef, rRes]
        outs_params.iloc[:, pz] = params

        out_yRef.iloc[:, pz + 1] = yRef
        out_yRes.iloc[:, pz + 1] = yRes

        # plot
        plot_heads(t, yObs, yRef, yRes, rRef, rRes, save_path, well_ids[pz], fs=fs)

    # add time data
    out_yRef['t'] = t
    out_yRes['t'] = t

    # save to excel
    rCoef.to_excel(save_path + 'rCoef.xlsx')
    outs_params.to_excel(save_path + 'params.xlsx')
    out_yRef.to_excel(save_path + 'reference_head.xlsx')
    out_yRes.to_excel(save_path + 'residual_head.xlsx')
    df_pca_vec.to_excel(save_path + 'pca_vec.xlsx')
