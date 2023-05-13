import pandas as pd
import numpy as np
import pymannkendall as mk


def resample_randomize_and_convert_to_numpy(data, well_ids, resample_op='mean', toResample=False, freq='2D',
                                            toRandomize=False):
    assert resample_op == 'mean' or resample_op == 'sum'
    # resample
    if toResample:
        if resample_op == 'mean':
            data = data.resample(freq).mean()
        else:
            data = data.resample(freq).sum()
        data['t'] = data.index

    t = data['t'].values

    if toRandomize:
        # resample randomly with replacement to get a new "network" of piezometers
        data = data.sample(frac=1, replace=True).reset_index(drop=True)

    # choose piezometers of interest
    data = data[well_ids].values
    return t, data


def data_loader(well_path, head_path=None, climate_path=None, isClimateVars=False,
                resample_op='mean', toResample=False, freq='2D', toRandomize=False, returnHeadOnly=False):
    # assertion on well_path
    assert well_path is not None

    # get ids of the observation wells
    info_on_wells = pd.read_excel(well_path)
    info_on_wells['id'] = info_on_wells['id'].astype('str')
    info_on_wells['id'] = '0' + info_on_wells['id']
    well_ids = info_on_wells['id'].values

    if isClimateVars:
        assert climate_path is not None
        # get time series
        data = pd.ExcelFile(climate_path)
        vi = pd.read_excel(data, sheet_name=0, index_col=0)
        ep = pd.read_excel(data, sheet_name=1, index_col=0)

        vi.fillna(method='ffill', inplace=True)
        ep.fillna(method='ffill', inplace=True)

        # reindex
        vi.set_index(vi['t'], inplace=True)
        ep.set_index(vi['t'], inplace=True)

        # resample and randomize if necessary
        t, vi = resample_randomize_and_convert_to_numpy(vi, well_ids, resample_op=resample_op, toResample=toResample,
                                                        freq=freq, toRandomize=toRandomize)
        _, ep = resample_randomize_and_convert_to_numpy(ep, well_ids, resample_op=resample_op, toResample=toResample,
                                                        freq=freq, toRandomize=toRandomize)

        return t, well_ids, vi, ep, info_on_wells
    else:
        assert head_path is not None
        # get time series
        head = pd.read_excel(head_path, sheet_name=0, index_col=0)
        head.fillna(method='ffill', inplace=True)

        # reindex
        head.set_index(head['t'], inplace=True)

        # resample and randomize if necessary
        t, head = resample_randomize_and_convert_to_numpy(head, well_ids, resample_op=resample_op,
                                                          toResample=toResample, freq=freq, toRandomize=toRandomize)

        if returnHeadOnly:
            return head
        else:
            return t, well_ids, head, info_on_wells


def get_coordinates_of_a_given_well(info_on_wells, name):
    coord = info_on_wells[info_on_wells['id'] == name][['x', 'y']]
    x = coord['x'].values[0]
    y = coord['y'].values[0]
    return x, y


def trend_local_scale(well_ids, data, info_on_wells, save_path, alpha=0.05, factor=365.25 * 100):
    # initiate the mk method
    mk_tfp = mk.trend_free_pre_whitening_modification_test

    # trend for each pz time series
    number_piezo = data.shape[1]
    columns = ['id', 'x', 'y', 'trend', 'slope']
    outputs = pd.DataFrame(np.empty((number_piezo, len(columns))), columns=columns)

    for pz in range(number_piezo):
        name = well_ids[pz]
        result = mk_tfp(data[:, pz], alpha)

        # display step
        print('%d/%d - %s' % (pz + 1, number_piezo, name))

        # unpack results
        trend, h, p, z, Tau, s, var_s, slope, intercept = result

        if h:
            slope = np.round(slope * factor, 2)
            trend = 'D' if slope < 0 else 'I'
        else:
            slope = 0

        # get additional information
        x, y = get_coordinates_of_a_given_well(info_on_wells, name)

        # fill the outputs dataframe
        outputs.iloc[pz, :] = [name, x, y, trend, slope]

    outputs.to_excel(save_path)


def trend_regional_scale(well_path, head_path, save_path, alpha=0.05, number_iter=1000,
                         resample_op='mean', toResample=False, freq='2D'):
    # only for groundwater level
    # initiate the mk method
    mk_tfp = mk.trend_free_pre_whitening_modification_test

    # define the outputs dataframe
    outputs = pd.DataFrame(np.zeros((number_iter, 3)), columns=['no_iter', 'upward', 'downward'])

    for _iter in range(number_iter):
        # display step
        print('Iteration %d/%d ...' % (_iter + 1, number_iter))

        # load data
        head = data_loader(well_path=well_path, head_path=head_path, climate_path=None, isClimateVars=False,
                           resample_op=resample_op, toResample=toResample, freq=freq, toRandomize=True,
                           returnHeadOnly=True)

        # compute MK statistic and p value at each piezometer and count number of upward and downward trends
        n_up, n_down = 0, 0
        number_pz = head.shape[1]

        for pz in range(number_pz):
            result = mk_tfp(head[:, pz], alpha)
            trend = result.trend

            if trend == 'increasing':
                n_up += 1

            if trend == 'decreasing':
                n_down += 1

        # save results of iteration '_iter'
        outputs.iloc[_iter, :] = [_iter + 1, n_up, n_down]

    # compute the boostrap empirical cumulative distributions (BECDs) of the numbers of upward and downward trends
    upward = np.sort(np.array(outputs['upward'].values))
    downward = np.sort(np.array(outputs['downward'].values))

    becd_upward = pd.DataFrame(np.zeros((number_iter, 2)), columns=['n_up', 'BECD'])
    becd_downward = pd.DataFrame(np.zeros((number_iter, 2)), columns=['n_down', 'BECD'])

    for k in range(number_iter):
        p_up = (k + 1) / (number_iter + 1)
        p_down = (k + 1) / (number_iter + 1)

        becd_upward.iloc[k, :] = [upward[k], p_up]
        becd_downward.iloc[k, :] = [downward[k], p_down]

    # compute n_upward and n_downward for the real network
    n_upward, n_downward = 0, 0
    head = data_loader(well_path=well_path, head_path=head_path, climate_path=None, isClimateVars=False,
                       resample_op=resample_op, toResample=toResample, freq=freq, toRandomize=False,
                       returnHeadOnly=True)

    number_pz = head.shape[1]
    for pz in range(number_pz):
        result = mk_tfp(head[:, pz], alpha)
        trend = result.trend

        if trend == 'increasing':
            n_upward += 1

        if trend == 'decreasing':
            n_downward += 1

    # compute the p-values of up and down of the real network
    # save
    writer = pd.ExcelWriter(save_path, engine="openpyxl")
    becd_upward.to_excel(writer, sheet_name='upward')
    becd_downward.to_excel(writer, sheet_name='downward')

    # save Nup and Ndown for the real network
    data = np.reshape(np.array([n_upward, n_downward]), (1, -1))
    data = pd.DataFrame(data, columns=['n_upward', 'n_downward'])
    data.to_excel(writer, sheet_name='real_net')

    writer.save()
