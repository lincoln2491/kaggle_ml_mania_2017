import numpy as np
import pandas as pd


def change_data_l_h(detailed):
    detailed[['l_team', 'h_team', 'l_score', 'h_score', 'l_loc',
              'l_fgm', 'l_fga', 'l_fgm3', 'l_fga3', 'l_ftm', 'l_fta', 'l_or', 'l_dr', 'l_ast', 'l_to', 'l_stl', 'l_blk',
              'l_pf',
              'h_fgm', 'h_fga', 'h_fgm3', 'h_fga3', 'h_ftm', 'h_fta', 'h_or', 'h_dr', 'h_ast', 'h_to', 'h_stl', 'h_blk',
              'h_pf'
              ]] = detailed.apply(lambda x: normalize_data_to_l_h(x), axis=1)

    detailed = detailed.drop(
        ['wteam', 'wscore', 'lteam', 'lscore', 'wloc',
         'wfgm', 'wfga', 'wfgm3', 'wfga3', 'wftm', 'wfta', 'wor', 'wdr', 'wast', 'wto', 'wstl', 'wblk', 'wpf',
         'lfgm', 'lfga', 'lfgm3', 'lfga3', 'lftm', 'lfta', 'lor', 'ldr', 'last', 'lto', 'lstl', 'lblk', 'lpf'
         ], axis=1)
    detailed = detailed.assign(result=np.where(detailed['l_score'] > detailed['h_score'], 1, 0))
    return detailed


def normalize_data_to_l_h(x):
    if x['wteam'] < x['lteam']:
        return pd.Series(
            [x['wteam'], x['lteam'], x['wscore'], x['lscore'], x['wloc'],
             x['wfgm'], x['wfga'], x['wfgm3'], x['wfga3'], x['wftm'], x['wfta'], x['wor'], x['wdr'], x['wast'],
             x['wto'], x['wstl'], x['wblk'], x['wpf'],
             x['lfgm'], x['lfga'], x['lfgm3'], x['lfga3'], x['lftm'], x['lfta'], x['lor'], x['ldr'], x['last'],
             x['lto'], x['lstl'], x['lblk'], x['lpf']])
    else:
        d = {'H': 'A', 'A': 'H'}
        l_loc = d[x['wloc']] if d.has_key(x['wloc']) else x['wloc']
        return pd.Series(
            [x['lteam'], x['wteam'], x['lscore'], x['wscore'], l_loc,
             x['lfgm'], x['lfga'], x['lfgm3'], x['lfga3'], x['lftm'], x['lfta'], x['lor'], x['ldr'], x['last'],
             x['lto'], x['lstl'], x['lblk'], x['lpf'],
             x['wfgm'], x['wfga'], x['wfgm3'], x['wfga3'], x['wftm'], x['wfta'], x['wor'], x['wdr'], x['wast'],
             x['wto'], x['wstl'], x['wblk'], x['wpf']])


#optimize
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
