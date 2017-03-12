from datetime import datetime
import sys
import pickle

from ml_mania_2017.model_year_regular_tournament import *
from ml_mania_2017.functions import *

pd.set_option('display.width', 320)

teams, seasons, rs_detailed, t_detailed, t_seeds, t_slots, ordinals, match_geog, team_geog = load_data()
print str(datetime.now()) + ' data loaded'
# rs_detailed, t_detailed = elo_ranking.add_elo_ranking(rs_detailed, t_detailed)
# print str(datetime.now()) + ' elo ranking added'
#
# data_per_season = dict()
# for i in range(2005, 2016 + 1):
#     print str(i) + ' ' + str(datetime.now())
#     tmp = create_train_set_for_year(rs_detailed, ordinals, None, is_train=True, year=i)
#     data_per_season[i] = tmp
#     pickle.dump(tmp, open('tmp_data/' + str(i) + '.p', 'wb'))
# data = pd.concat(data_per_season).reset_index(drop=True)

# tournament_per_season = dict()
# for i in range(2003, 2016 + 1):
#     print str(i) + ' ' + str(datetime.now())
#     tournament_per_season[i] = create_train_set_for_year(rs_detailed, ordinals, t_detailed, is_train=False, year=i)
# tournament_data = pd.concat(tournament_per_season).reset_index(drop=True)
# print 'data transformed'

# data.to_csv('data/rs_detailed-all.csv', sep=';', index=None)
# tournament_data.to_csv('data/t_detailed-all.csv', sep=';', index=None)


rs_detailed = rs_detailed.drop(columns_to_drop, axis=1)
t_detailed = t_detailed.drop(columns_to_drop, axis=1)

rs_detailed = rs_detailed.drop(['season', 'daynum', 'type', 'l_team', 'h_team', 'l_score', 'h_score', 'l_loc', 'numot'], axis=1)
t_detailed = t_detailed.drop(['season', 'daynum', 'type', 'l_team', 'h_team', 'l_score', 'h_score', 'l_loc', 'numot'], axis=1)

# rs_detailed, t_detailed = add_geo_data(rs_detailed, t_detailed, team_geog, match_geog)
#
# rs_detailed = add_rankings_diff(rs_detailed)
# t_detailed = add_rankings_diff(t_detailed)
#
# columns_m10 = [i.replace('l_', '', 1) for i in rs_detailed.columns if i.endswith('_m10') and i.startswith('l_')]
# columns_m5 = [i.replace('l_', '', 1) for i in rs_detailed.columns if i.endswith('_m5') and i.startswith('l_')]
# columns_m3 = [i.replace('l_', '', 1) for i in rs_detailed.columns if i.endswith('_m3') and i.startswith('l_')]
# columns_m1 = [i.replace('l_', '', 1) for i in rs_detailed.columns if i.endswith('_m1') and i.startswith('l_')]
#
# rs_detailed = add_diffs_for_columns(rs_detailed, columns_m10)
# rs_detailed = add_diffs_for_columns(rs_detailed, columns_m5)
# rs_detailed = add_diffs_for_columns(rs_detailed, columns_m3)
# rs_detailed = add_diffs_for_columns(rs_detailed, columns_m1)
#
# t_detailed = add_diffs_for_columns(t_detailed, columns_m10)
# t_detailed = add_diffs_for_columns(t_detailed, columns_m5)
# t_detailed = add_diffs_for_columns(t_detailed, columns_m3)
# t_detailed = add_diffs_for_columns(t_detailed, columns_m1)
#
# rs_detailed = rs_detailed.assign(diff_elo_season=rs_detailed.l_elo_season - rs_detailed.h_elo_season)
# rs_detailed = rs_detailed.assign(diff_elo_all=rs_detailed.l_elo_all - rs_detailed.h_elo_all)
#
# t_detailed = t_detailed.assign(diff_elo_season=t_detailed.l_elo_season - t_detailed.h_elo_season)
# t_detailed = t_detailed.assign(diff_elo_all=t_detailed.l_elo_all - t_detailed.h_elo_all)
#
# col = [i for i in rs_detailed.columns if 'diff_' in i]
# col.append('result')
tmp = rs_detailed.count(axis=0)
cols = tmp[tmp > 60000].index

rs_detailed = rs_detailed[cols]
t_detailed = t_detailed[cols]
# data = rs_detailed
# tournament_data = t_detailed

rs_detailed = rs_detailed[rs_detailed.notnull().all(axis=1)]
# data = data[col]
# tournament_data = tournament_data[col]

# data = data.drop(columns_to_drop, axis=1)
# data = data.drop('diff_score', axis=1)
# tournament_data = tournament_data.drop(columns_to_drop, axis=1)



train_x = rs_detailed.drop('result', 1)
train_y = rs_detailed['result']

test_x = t_detailed.drop('result', 1).copy()
test_y = t_detailed['result'].copy()

dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x, label=test_y)

bst = create_and_train_model(dtrain, dtest)
print 'model created'
