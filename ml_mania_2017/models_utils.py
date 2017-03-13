import numpy as np
import pandas as pd
from scipy.stats import linregress

columns_to_drop = ['l_fgm', 'l_fga', 'l_fgm3', 'l_fga3', 'l_ftm', 'l_fta', 'l_or', 'l_dr', 'l_ast', 'l_to', 'l_stl', 'l_blk', 'l_pf', 'h_fgm', 'h_fga',
                   'h_fgm3', 'h_fga3', 'h_ftm', 'h_fta', 'h_or', 'h_dr', 'h_ast', 'h_to', 'h_stl', 'h_blk', 'h_pf', 'l_fgm2', 'h_fgm2', 'l_fga2', 'h_fga2',
                   'l_tr', 'h_tr', 'l_fgm3r', 'h_fgm3r', 'l_fgm2r', 'h_fgm2r', 'l_fga3r', 'h_fga3r', 'l_fga2r', 'h_fga2r', 'l_fgmar', 'h_fgmar', 'l_fgma2r',
                   'h_fgma2r', 'l_fgma3r', 'h_fgma3r', 'l_odrr', 'h_odrr', 'l_orr', 'h_orr', 'l_drr', 'h_drr']


def change_data_l_h(detailed):
    detailed[['l_team', 'h_team', 'l_score', 'h_score', 'l_loc',
              'l_fgm', 'l_fga', 'l_fgm3', 'l_fga3', 'l_ftm', 'l_fta', 'l_or', 'l_dr', 'l_ast', 'l_to', 'l_stl', 'l_blk',
              'l_pf', 'l_elo_all', 'l_elo_season',
              'h_fgm', 'h_fga', 'h_fgm3', 'h_fga3', 'h_ftm', 'h_fta', 'h_or', 'h_dr', 'h_ast', 'h_to', 'h_stl', 'h_blk',
              'h_pf', 'h_elo_all', 'h_elo_season'
              ]] = detailed.apply(lambda x: normalize_data_to_l_h(x), axis=1)

    detailed = detailed.drop(
        ['wteam', 'wscore', 'lteam', 'lscore', 'wloc',
         'wfgm', 'wfga', 'wfgm3', 'wfga3', 'wftm', 'wfta', 'wor', 'wdr', 'wast', 'wto', 'wstl', 'wblk', 'wpf',
         'lfgm', 'lfga', 'lfgm3', 'lfga3', 'lftm', 'lfta', 'lor', 'ldr', 'last', 'lto', 'lstl', 'lblk', 'lpf', 'welo',
         'lelo', 'weloseason', 'leloseason'
         ], axis=1)
    detailed = detailed.assign(result=np.where(detailed['l_score'] > detailed['h_score'], 1, 0))
    return detailed


def normalize_data_to_l_h(x):
    if x['wteam'] < x['lteam']:
        return pd.Series(
            [x['wteam'], x['lteam'], x['wscore'], x['lscore'], x['wloc'],
             x['wfgm'], x['wfga'], x['wfgm3'], x['wfga3'], x['wftm'], x['wfta'], x['wor'], x['wdr'], x['wast'],
             x['wto'], x['wstl'], x['wblk'], x['wpf'], x['welo'], x['weloseason'],
             x['lfgm'], x['lfga'], x['lfgm3'], x['lfga3'], x['lftm'], x['lfta'], x['lor'], x['ldr'], x['last'],
             x['lto'], x['lstl'], x['lblk'], x['lpf'], x['lelo'], x['leloseason']])
    else:
        d = {'H': 'A', 'A': 'H'}
        l_loc = d[x['wloc']] if d.has_key(x['wloc']) else x['wloc']
        return pd.Series(
            [x['lteam'], x['wteam'], x['lscore'], x['wscore'], l_loc,
             x['lfgm'], x['lfga'], x['lfgm3'], x['lfga3'], x['lftm'], x['lfta'], x['lor'], x['ldr'], x['last'],
             x['lto'], x['lstl'], x['lblk'], x['lpf'], x['lelo'], x['leloseason'],
             x['wfgm'], x['wfga'], x['wfgm3'], x['wfga3'], x['wftm'], x['wfta'], x['wor'], x['wdr'], x['wast'],
             x['wto'], x['wstl'], x['wblk'], x['wpf'], x['welo'], x['weloseason']])


def add_statistics_to_l_h_model(detailed):
    # field foals made 2 point calculation
    detailed = detailed.assign(l_fgm2=detailed.l_fgm - detailed.l_fgm3)
    detailed = detailed.assign(h_fgm2=detailed.h_fgm - detailed.h_fgm3)
    detailed = detailed.assign(l_fga2=detailed.l_fga - detailed.l_fga3)
    detailed = detailed.assign(h_fga2=detailed.h_fga - detailed.h_fga3)

    # total rebounds
    detailed = detailed.assign(l_tr=detailed.l_or + detailed.l_dr)
    detailed = detailed.assign(h_tr=detailed.h_or + detailed.h_dr)

    # ratio for 2/total and 3/total goals made
    detailed = detailed.assign(l_fgm3r=detailed.l_fgm3 / detailed.l_fgm)
    detailed = detailed.assign(h_fgm3r=detailed.h_fgm3 / detailed.h_fgm)

    detailed = detailed.assign(l_fgm2r=detailed.l_fgm2 / detailed.l_fgm)
    detailed = detailed.assign(h_fgm2r=detailed.h_fgm2 / detailed.h_fgm)

    # ratio for 2/total and 3/total goals attempt
    detailed = detailed.assign(l_fga3r=detailed.l_fga3 / detailed.l_fga)
    detailed = detailed.assign(h_fga3r=detailed.h_fga3 / detailed.h_fga)

    detailed = detailed.assign(l_fga2r=detailed.l_fga2 / detailed.l_fga)
    detailed = detailed.assign(h_fga2r=detailed.h_fga2 / detailed.h_fga)

    # ratio for goal/attempts (total, 2 or 3)
    detailed = detailed.assign(l_fgmar=detailed.l_fgm / detailed.l_fga)
    detailed = detailed.assign(h_fgmar=detailed.h_fgm / detailed.h_fga)

    detailed = detailed.assign(l_fgma2r=detailed.l_fgm2 / detailed.l_fga2)
    detailed = detailed.assign(h_fgma2r=detailed.h_fgm2 / detailed.h_fga2)

    detailed = detailed.assign(l_fgma3r=detailed.l_fgm3 / detailed.l_fga3)
    detailed = detailed.assign(h_fgma3r=detailed.h_fgm3 / detailed.h_fga3)

    # ratio for offensive/defensive rebounds
    detailed = detailed.assign(l_odrr=detailed.l_or / detailed.l_dr)
    detailed = detailed.assign(h_odrr=detailed.h_or / detailed.h_dr)

    # ratio for offensive/total and defensive/total rebouds
    detailed = detailed.assign(l_orr=detailed.l_or / detailed.l_tr)
    detailed = detailed.assign(h_orr=detailed.h_or / detailed.h_tr)

    detailed = detailed.assign(l_drr=detailed.l_dr / detailed.l_tr)
    detailed = detailed.assign(h_drr=detailed.h_dr / detailed.h_tr)

    return detailed


# optimize
def add_info_about_rank_before_match(rs_detailed, ordinals, year=2016, sys_name='PGH'):
    tmp_o = ordinals[((ordinals.season == year) & (ordinals.sys_name == sys_name))].copy()
    tmp_r = rs_detailed[rs_detailed.season == year].copy()
    tmp_r['l_rank_' + sys_name] = tmp_r.apply(lambda x: tmp_o.loc[tmp_o.loc[((tmp_o.team == x['l_team']) & (
        tmp_o.rating_day_num <= x['daynum'])), 'rating_day_num'].idxmax(), 'orank'] if x['daynum'] >= tmp_o.loc[
        tmp_o.team == x['l_team'], 'rating_day_num'].min() else np.nan, axis=1)
    tmp_r['h_rank_' + sys_name] = tmp_r.apply(lambda x: tmp_o.loc[tmp_o.loc[((tmp_o.team == x['h_team']) & (
        tmp_o.rating_day_num <= x['daynum'])), 'rating_day_num'].idxmax(), 'orank'] if x['daynum'] >= tmp_o.loc[
        tmp_o.team == x['h_team'], 'rating_day_num'].min() else np.nan, axis=1)

    return tmp_r


def create_predict_data_for_model(sample_submission, ordinals, sys_name='PGH'):
    sample_submission['season'] = pd.to_numeric(sample_submission.id.apply(lambda x: x.split('_')[0]))
    sample_submission['lid'] = pd.to_numeric(sample_submission.id.apply(lambda x: x.split('_')[1]))
    sample_submission['hid'] = pd.to_numeric(sample_submission.id.apply(lambda x: x.split('_')[2]))
    sample_submission['low_rank'] = np.nan
    sample_submission['high_rank'] = np.nan
    tmp_o = ordinals[ordinals.sys_name == sys_name].copy()

    sample_submission['low_rank'] = sample_submission.apply(lambda x: tmp_o.loc[
        tmp_o.loc[(tmp_o.team == x['lid']) & (tmp_o.season == x['season']), 'rating_day_num'].idxmax(), 'orank'],
                                                            axis=1)
    sample_submission['high_rank'] = sample_submission.apply(lambda x: tmp_o.loc[
        tmp_o.loc[(tmp_o.team == x['hid']) & (tmp_o.season == x['season']), 'rating_day_num'].idxmax(), 'orank'],
                                                             axis=1)

    return sample_submission


def get_mean_from_last_n_matches(team_id, detailed, day, feature_list, is_l=True, n=10):
    tmp_detailed = detailed[((detailed.l_team == team_id) | (detailed.h_team == team_id)) & (detailed.daynum < day)].sort_values(
        'daynum').iloc[-n:]

    y = range(tmp_detailed.shape[0])
    result = dict()

    label_prefix = 'l_' if is_l else 'h_'
    for feature in feature_list:
        try:
            values = tmp_detailed.apply(lambda x: x['l_' + feature] if x['l_team'] == team_id else x['h_team'], axis=1).tolist()
            values_op = tmp_detailed.apply(lambda x: x['l_' + feature] if x['h_team'] == team_id else x['l_team'], axis=1).tolist()
        except:
            values = []
            values_op = []

        result[label_prefix + feature + '_m' + str(n)] = np.mean(values)
        result[label_prefix + feature + '_om' + str(n)] = np.mean(values_op)
        result[label_prefix + feature + '_std' + str(n)] = np.std(values)
        result[label_prefix + feature + '_ostd' + str(n)] = np.std(values_op)
        if n > 1:
            try:
                slope, intercept, rval, pval, stderr = linregress(values, y)
            except ValueError:
                slope = intercept = rval = pval = stderr = np.NaN
            result[label_prefix + feature + '_slope' + str(n)] = slope
            result[label_prefix + feature + '_int' + str(n)] = intercept
            result[label_prefix + feature + '_rval' + str(n)] = rval
            result[label_prefix + feature + '_pval' + str(n)] = pval
            result[label_prefix + feature + '_stderr' + str(n)] = stderr

            try:
                slope, intercept, rval, pval, stderr = linregress(values_op, y)
            except ValueError:
                slope = intercept = rval = pval = stderr = np.NaN
            result[label_prefix + feature + '_slope' + str(n)] = slope
            result[label_prefix + feature + '_oint' + str(n)] = intercept
            result[label_prefix + feature + '_orval' + str(n)] = rval
            result[label_prefix + feature + '_opval' + str(n)] = pval
            result[label_prefix + feature + '_ostderr' + str(n)] = stderr

    result[label_prefix + '_numot_m' + str(n)] = tmp_detailed.numot.mean()
    result[label_prefix + '_numot_std' + str(n)] = tmp_detailed.numot.std()

    return pd.Series(result)


def get_opposite_mean_from_last_n_matches(team_id, detailed, day, feature, n=10):
    values = \
        detailed[((detailed.l_team == team_id) | (detailed.h_team == team_id)) & (detailed.daynum < day)].sort_values(
            'daynum').iloc[-n:].loc[detailed.l_team == team_id, 'h_' + feature].tolist()
    values.extend(
        detailed[((detailed.l_team == team_id) | (detailed.h_team == team_id)) & (detailed.daynum < day)].sort_values(
            'daynum').iloc[-n:].loc[detailed.h_team == team_id, 'l_' + feature].tolist())

    return np.mean(values)


def add_geo_data(rs_detailed, t_detailed, team_geog, match_geog):
    rs_detailed = add_team_lat_lon(rs_detailed, team_geog)
    t_detailed = add_team_lat_lon(t_detailed, team_geog)

    rs_detailed['home_id'] = rs_detailed.loc[rs_detailed.l_loc == 'H', 'l_team'].append(rs_detailed.loc[rs_detailed.l_loc == 'A', 'h_team'])
    rs_detailed = rs_detailed.merge(team_geog.rename(columns={'lat': 'stadium_lat', 'lng': 'stadium_lon'}), left_on='home_id', right_on='team_id'). \
        drop(['home_id', 'team_id'], axis=1)

    match_geog = match_geog.assign(l_team=match_geog[['wteam', 'lteam']].min(axis=1)).assign(h_team=match_geog[['wteam', 'lteam']].max(axis=1))
    t_detailed = t_detailed.merge(match_geog[['season', 'daynum', 'l_team', 'h_team', 'lat', 'lng']]).rename(
        columns={'lat': 'stadium_lat', 'lng': 'stadium_lon'})
    return rs_detailed, t_detailed


def add_team_lat_lon(detailed, team_geog):
    detailed = detailed.merge(team_geog.rename(columns={'lat': 'l_lat', 'lng': 'l_lon'}), left_on='l_team', right_on='team_id').drop('team_id', axis=1). \
        merge(team_geog.rename(columns={'lat': 'h_lat', 'lng': 'h_lon'}), left_on='h_team', right_on='team_id').drop('team_id', axis=1)
    return detailed


def get_days_before_match(rs_detailed, t_detailed, year):
    data = rs_detailed.append(t_detailed).sort_values(['season', 'daynum'])

    data.assign
