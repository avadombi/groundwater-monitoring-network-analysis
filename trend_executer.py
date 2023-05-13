from TrendAnalysis.trends import data_loader, trend_local_scale, trend_regional_scale


def get_data():
    # paths
    base_path = 'Data/'
    well_path = base_path + 'wells.xlsx'
    head_path = base_path + 'head.xlsx'
    climate_path = base_path + 'climate_drivers.xlsx'

    # climate variables (vi: vertical inflow, ep: potential evapotranspiration)
    _, _, vi, ep, _ = data_loader(well_path, head_path=None, climate_path=climate_path,
                                  isClimateVars=True, resample_op='sum', toResample=True,
                                  freq='Y', toRandomize=False, returnHeadOnly=False)

    # heads
    t, well_ids, head, info_on_wells = data_loader(well_path, head_path=head_path, climate_path=None,
                                                   isClimateVars=False, resample_op='mean', toResample=False,
                                                   freq='2D', toRandomize=False, returnHeadOnly=False)

    return t, well_ids, info_on_wells, head, vi, ep


def local_trends_for_groundwater_levels_and_climate_vars(alpha=0.05):
    # load data
    t, well_ids, info_on_wells, head, vi, ep = get_data()

    # local trend in GWL
    save_path = 'Results/local_trends_heads.xlsx'
    trend_local_scale(well_ids=well_ids, data=head, info_on_wells=info_on_wells, save_path=save_path, alpha=alpha,
                      factor=365.25 * 100)  # daily data -> slope (S) in m/day so S = factor * S to get cm/year

    # local trend in vi
    save_path = 'Results/local_trends_vi.xlsx'
    trend_local_scale(well_ids=well_ids, data=vi, info_on_wells=info_on_wells, save_path=save_path, alpha=alpha,
                      factor=1.0)  # daily data -> yearly sample -> slope (S) in mm/year, so factor = 1

    # local trend in ep
    save_path = 'Results/local_trends_ep.xlsx'
    trend_local_scale(well_ids=well_ids, data=ep, info_on_wells=info_on_wells, save_path=save_path, alpha=alpha,
                      factor=1.0)  # daily data -> yearly sample -> slope (S) in mm/year, so factor = 1


def regional_trends_for_groundwater_levels(alpha=0.05, number_iter=1000, resample_op='mean', toResample=False,
                                           freq='2D'):
    # paths
    base_path = 'Data/'
    well_path = base_path + 'wells.xlsx'
    head_path = base_path + 'head.xlsx'
    save_path = 'Results/regional_trends_heads.xlsx'

    trend_regional_scale(well_path, head_path, save_path, alpha=alpha, number_iter=number_iter,
                         resample_op=resample_op, toResample=toResample, freq=freq)


# assess local trends with the TFPW-MK method
local_trends_for_groundwater_levels_and_climate_vars(alpha=0.05)

# assess the regional trends with the bootstrap approach
regional_trends_for_groundwater_levels(alpha=0.05, number_iter=1000, resample_op='mean', toResample=False,
                                       freq='2D')
