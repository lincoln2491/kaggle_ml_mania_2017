import gc

from ml_mania_2017 import elo_ranking
from datetime import datetime
import sys
import pickle

from ml_mania_2017.model_year_regular_tournament import *
from ml_mania_2017.functions import *

pd.set_option('display.width', 320)

teams, seasons, rs_detailed, t_detailed, t_seeds, t_slots, ordinals, match_geog, team_geog, sample_submission = load_data()
print str(datetime.now()) + ' data loaded'
gc.collect()
types = rs_detailed.dtypes
types = types[types == 'float64'].index
rs_detailed[types] = rs_detailed[types].astype(np.float32)
t_detailed[types] = t_detailed[types].astype(np.float32)
types = sample_submission.dtypes
types = types[types == 'float64'].index
print str(datetime.now()) + ' data changed'

# sample_submission = sample_submission.assign(daynum=134)
# sample_submission = sample_submission.assign(season=2017)
# sample_submission = sample_submission.assign(l_team=sample_submission.Id.apply(lambda x: int(x.split('_')[1])))
# sample_submission = sample_submission.assign(h_team=sample_submission.Id.apply(lambda x: int(x.split('_')[2])))
#
# rs_detailed, t_detailed, sample_submission = elo_ranking.add_elo_ranking(rs_detailed, t_detailed, sample_submission)
#
# print str(datetime.now()) + ' elo ranking added'
#
# data_per_season = dict()
# for i in range(2003, 2017 + 1):
#     print str(i) + ' ' + str(datetime.now())
#     tmp = create_train_set_for_year(rs_detailed, ordinals, None, is_train=True, year=i)
#     data_per_season[i] = tmp
#     pickle.dump(tmp, open('tmp_data/' + str(i) + '.p', 'wb'))
# data = pd.concat(data_per_season).reset_index(drop=True)
#
# tournament_per_season = dict()
# for i in range(2003, 2016 + 1):
#     print str(i) + ' ' + str(datetime.now())
#     tournament_per_season[i] = create_train_set_for_year(rs_detailed, ordinals, t_detailed, is_train=False, year=i)
# tournament_data = pd.concat(tournament_per_season).reset_index(drop=True)
# print 'data transformed'

# data.to_csv('data/rs_detailed-all.csv', sep=';', index=None)
# tournament_data.to_csv('data/t_detailed-all.csv', sep=';', index=None)


# sample_submission = create_train_set_for_year(rs_detailed, ordinals, sample_submission, is_train=False, year=2017, is_sample=True)
# sample_submission.to_csv('data/sample_submission-all.csv', sep = ';', index = None)

rs_detailed = rs_detailed.drop(columns_to_drop, axis=1)
t_detailed = t_detailed.drop(columns_to_drop, axis=1)
gc.collect()

rs_detailed = pd.get_dummies(rs_detailed, columns=['l_team', 'h_team'])
t_detailed = pd.get_dummies(t_detailed, columns=['l_team', 'h_team'])
sample_submission = pd.get_dummies(sample_submission, columns=['l_team', 'h_team'])

col_rs = [i for i in rs_detailed.columns if i.startswith('l_team') or i.startswith('h_team')]
col_t = [i for i in t_detailed.columns if i.startswith('l_team') or i.startswith('h_team')]
col_s = [i for i in sample_submission.columns if i.startswith('l_team') or i.startswith('h_team')]
cols = set(col_rs.__add__(col_t).__add__(col_s))

for i in cols:
    if i not in rs_detailed.columns:
        rs_detailed.loc[:, i] = 0
    if i not in t_detailed.columns:
        t_detailed.loc[:, i] = 0
    if i not in sample_submission.columns:
        sample_submission.loc[:, i] = 0

rs_detailed = rs_detailed.drop(['season', 'daynum', 'type', 'l_score', 'h_score', 'numot'], axis=1)
t_detailed = t_detailed.drop(['season', 'daynum', 'type', 'l_score', 'h_score', 'numot', 'l_loc'], axis=1)
sample_submission = sample_submission.drop(['season', 'daynum', 'pred'], axis=1)

rs_detailed = pd.get_dummies(rs_detailed, columns=['l_loc'])
gc.collect()
t_detailed = t_detailed.assign(l_loc_A=0.0)
t_detailed = t_detailed.assign(l_loc_H=0.0)
t_detailed = t_detailed.assign(l_loc_N=1.0)

sample_submission = sample_submission.assign(l_loc_A=0.0)
sample_submission = sample_submission.assign(l_loc_H=0.0)
sample_submission = sample_submission.assign(l_loc_N=1.0)

# rs_detailed, t_detailed = add_geo_data(rs_detailed, t_detailed, team_geog, match_geog)
#
rs_detailed = add_rankings_diff(rs_detailed)
t_detailed = add_rankings_diff(t_detailed)
sample_submission = add_rankings_diff(sample_submission)
#
columns_m10 = [i.replace('l_', '', 1) for i in rs_detailed.columns if i.endswith('_m10') and i.startswith('l_')]
columns_m5 = [i.replace('l_', '', 1) for i in rs_detailed.columns if i.endswith('_m5') and i.startswith('l_')]
columns_m3 = [i.replace('l_', '', 1) for i in rs_detailed.columns if i.endswith('_m3') and i.startswith('l_')]
columns_m1 = [i.replace('l_', '', 1) for i in rs_detailed.columns if i.endswith('_m1') and i.startswith('l_')]

rs_detailed = add_diffs_for_columns(rs_detailed, columns_m10)
rs_detailed = add_diffs_for_columns(rs_detailed, columns_m5)
rs_detailed = add_diffs_for_columns(rs_detailed, columns_m3)
rs_detailed = add_diffs_for_columns(rs_detailed, columns_m1)

t_detailed = add_diffs_for_columns(t_detailed, columns_m10)
t_detailed = add_diffs_for_columns(t_detailed, columns_m5)
t_detailed = add_diffs_for_columns(t_detailed, columns_m3)
t_detailed = add_diffs_for_columns(t_detailed, columns_m1)

sample_submission = add_diffs_for_columns(sample_submission, columns_m10)
sample_submission = add_diffs_for_columns(sample_submission, columns_m5)
sample_submission = add_diffs_for_columns(sample_submission, columns_m3)
sample_submission = add_diffs_for_columns(sample_submission, columns_m1)

rs_detailed = rs_detailed.assign(diff_elo_season=rs_detailed.l_elo_season - rs_detailed.h_elo_season)
rs_detailed = rs_detailed.assign(diff_elo_all=rs_detailed.l_elo_all - rs_detailed.h_elo_all)

t_detailed = t_detailed.assign(diff_elo_season=t_detailed.l_elo_season - t_detailed.h_elo_season)
t_detailed = t_detailed.assign(diff_elo_all=t_detailed.l_elo_all - t_detailed.h_elo_all)

sample_submission = sample_submission.assign(diff_elo_season=sample_submission.l_elo_season - sample_submission.h_elo_season)
sample_submission = sample_submission.assign(diff_elo_all=sample_submission.l_elo_all - sample_submission.h_elo_all)
#
# col = [i for i in rs_detailed.columns if 'diff_' in i]
# col.append('result')
data = rs_detailed
tournament_data = t_detailed
tournament_data = tournament_data[data.columns]
gc.collect()
# tmp = data.count(axis=0)
# cols = tmp[tmp > 60000].index
#
# data = data[cols]
# tournament_data = tournament_data[cols]
#
# data = data[data.notnull().all(axis=1)]
# tournament_data = tournament_data[tournament_data.notnull().all(axis=1)]

train_x = data.drop('result', 1)
train_y = data['result']

test_x = tournament_data.drop('result', 1)
test_y = tournament_data['result']

dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x, label=test_y)

gc.collect()

bst = create_and_train_model(dtrain, dtest)

prediction = predict(bst, sample_submission)
save_submission_and_model(prediction, bst)

print 'model created'
