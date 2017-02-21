import numpy as np
import pandas as pd

columns_for_model = ['result', 'l_rank_PGH', 'h_rank_PGH', 'l_rank_SAG', 'h_rank_SAG', 'l_rank_DOK', 'h_rank_DOK',
                     'l_rank_MAS', 'h_rank_MAS', 'l_rank_KPK', 'h_rank_KPK', 'l_rank_MOR', 'h_rank_MOR', 'l_rank_PIG',
                     'h_rank_PIG', 'l_rank_BBT', 'h_rank_BBT', 'l_rank_POM', 'h_rank_POM', 'l_score_m10', 'h_score_m10',
                     'l_score_om10', 'h_score_om10', 'l_fgm_m10', 'h_fgm_m10', 'l_fgm_om10', 'h_fgm_om10', 'l_fga_m10',
                     'h_fga_m10', 'l_fga_om10', 'h_fga_om10', 'l_fgm3_m10', 'h_fgm3_m10', 'l_fgm3_om10', 'h_fgm3_om10',
                     'l_fga3_m10', 'h_fga3_m10', 'l_fga3_om10', 'h_fga3_om10', 'l_ftm_m10', 'h_ftm_m10', 'l_ftm_om10',
                     'h_ftm_om10', 'l_fta_m10', 'h_fta_m10', 'l_fta_om10', 'h_fta_om10', 'l_or_m10', 'h_or_m10',
                     'l_or_om10', 'h_or_om10', 'l_dr_m10', 'h_dr_m10', 'l_dr_om10', 'h_dr_om10', 'l_ast_m10',
                     'h_ast_m10', 'l_ast_om10', 'h_ast_om10', 'l_to_m10', 'h_to_m10', 'l_to_om10', 'h_to_om10',
                     'l_stl_m10', 'h_stl_m10', 'l_stl_om10', 'h_stl_om10', 'l_blk_m10', 'h_blk_m10', 'l_blk_om10',
                     'h_blk_om10', 'l_pf_m10', 'h_pf_m10', 'l_pf_om10', 'h_pf_om10']


def change_data_l_h(detailed):
    detailed[['l_team', 'h_team', 'l_score', 'h_score', 'l_loc',
              'l_fgm', 'l_fga', 'l_fgm3', 'l_fga3', 'l_ftm', 'l_fta', 'l_or', 'l_dr', 'l_ast', 'l_to', 'l_stl', 'l_blk',
              'l_pf', 'l_elo_all',
              'h_fgm', 'h_fga', 'h_fgm3', 'h_fga3', 'h_ftm', 'h_fta', 'h_or', 'h_dr', 'h_ast', 'h_to', 'h_stl', 'h_blk',
              'h_pf', 'h_elo_all'
              ]] = detailed.apply(lambda x: normalize_data_to_l_h(x), axis=1)

    detailed = detailed.drop(
        ['wteam', 'wscore', 'lteam', 'lscore', 'wloc',
         'wfgm', 'wfga', 'wfgm3', 'wfga3', 'wftm', 'wfta', 'wor', 'wdr', 'wast', 'wto', 'wstl', 'wblk', 'wpf',
         'lfgm', 'lfga', 'lfgm3', 'lfga3', 'lftm', 'lfta', 'lor', 'ldr', 'last', 'lto', 'lstl', 'lblk', 'lpf', 'welo',
         'lelo',
         ], axis=1)
    detailed = detailed.assign(result=np.where(detailed['l_score'] > detailed['h_score'], 1, 0))
    return detailed


def normalize_data_to_l_h(x):
    if x['wteam'] < x['lteam']:
        return pd.Series(
            [x['wteam'], x['lteam'], x['wscore'], x['lscore'], x['wloc'],
             x['wfgm'], x['wfga'], x['wfgm3'], x['wfga3'], x['wftm'], x['wfta'], x['wor'], x['wdr'], x['wast'],
             x['wto'], x['wstl'], x['wblk'], x['wpf'], x['welo'],
             x['lfgm'], x['lfga'], x['lfgm3'], x['lfga3'], x['lftm'], x['lfta'], x['lor'], x['ldr'], x['last'],
             x['lto'], x['lstl'], x['lblk'], x['lpf'], x['lelo']])
    else:
        d = {'H': 'A', 'A': 'H'}
        l_loc = d[x['wloc']] if d.has_key(x['wloc']) else x['wloc']
        return pd.Series(
            [x['lteam'], x['wteam'], x['lscore'], x['wscore'], l_loc,
             x['lfgm'], x['lfga'], x['lfgm3'], x['lfga3'], x['lftm'], x['lfta'], x['lor'], x['ldr'], x['last'],
             x['lto'], x['lstl'], x['lblk'], x['lpf'], x['lelo'],
             x['wfgm'], x['wfga'], x['wfgm3'], x['wfga3'], x['wftm'], x['wfta'], x['wor'], x['wdr'], x['wast'],
             x['wto'], x['wstl'], x['wblk'], x['wpf'], x['welo']])


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
    l_detailed = \
        detailed[((detailed.l_team == team_id) | (detailed.h_team == team_id)) & (detailed.daynum < day)].sort_values(
            'daynum').iloc[-n:].loc[detailed.l_team == team_id]
    h_detailed = \
        detailed[((detailed.l_team == team_id) | (detailed.h_team == team_id)) & (detailed.daynum < day)].sort_values(
            'daynum').iloc[-n:].loc[detailed.h_team == team_id]

    result = dict()

    label_prefix = 'l_' if is_l else 'h_'
    for feature in feature_list:
        result[label_prefix + feature + '_m' + str(n)] = \
            (l_detailed['l_' + feature].sum() + h_detailed['h_' + feature].sum()) / float(n)
        result[label_prefix + feature + '_om' + str(n)] = \
            (l_detailed['h_' + feature].sum() + h_detailed['l_' + feature].sum()) / float(n)
    return pd.Series(result)


def get_opposite_mean_from_last_n_matches(team_id, detailed, day, feature, n=10):
    values = \
        detailed[((detailed.l_team == team_id) | (detailed.h_team == team_id)) & (detailed.daynum < day)].sort_values(
            'daynum').iloc[-n:].loc[detailed.l_team == team_id, 'h_' + feature].tolist()
    values.extend(
        detailed[((detailed.l_team == team_id) | (detailed.h_team == team_id)) & (detailed.daynum < day)].sort_values(
            'daynum').iloc[-n:].loc[detailed.h_team == team_id, 'l_' + feature].tolist())

    return np.mean(values)
