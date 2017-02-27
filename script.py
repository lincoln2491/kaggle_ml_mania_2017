from datetime import datetime

from ml_mania_2017.model_year_regular_tournament import *
from ml_mania_2017.functions import *
from ml_mania_2017 import elo_ranking

pd.set_option('display.width', 320)

teams, seasons, rs_detailed, t_detailed, t_seeds, t_slots, ordinals, match_geog, team_geog = load_data()
print str(datetime.now()) + ' data loaded'
# rs_detailed, t_detailed = elo_ranking.add_elo_ranking(rs_detailed, t_detailed)
print str(datetime.now()) + ' elo ranking added'

# data_per_season = dict()
# for i in range(2013, 2016 + 1):
#     print str(i) + ' ' + str(datetime.now())
#     data_per_season[i] = create_train_set_for_year(rs_detailed, ordinals, None, is_train=True, year=i)
# data = pd.concat(data_per_season).reset_index(drop=True)
#
# tournament_per_season = dict()
# for i in range(2013, 2016 + 1):
#     print str(i) + ' ' + str(datetime.now())
#     tournament_per_season[i] = create_train_set_for_year(rs_detailed, ordinals, t_detailed, is_train=False, year=i)
# tournament_data = pd.concat(tournament_per_season).reset_index(drop=True)
# print 'data transformed'

rs_detailed = rs_detailed.merge(team_geog.rename(columns={'lat': 'l_lat', 'lng': 'l_lon'}), left_on='l_team', right_on='team_id').drop('team_id', axis=1). \
    merge(team_geog.rename(columns={'lat': 'h_lat', 'lng': 'h_lon'}), left_on='h_team', right_on='team_id').drop('team_id', axis=1)

rs_detailed['home_id'] = rs_detailed.loc[rs_detailed.l_loc == 'H', 'l_team'].append(rs_detailed.loc[rs_detailed.l_loc == 'A', 'h_team'])

rs_detailed = rs_detailed.merge(team_geog.rename(columns={'lat': 'home_lat', 'lng': 'home_lon'}), left_on='home_id', right_on='team_id'). \
    drop(['home_id', 'team_id'], axis=1)




data = rs_detailed

data = data[data.notnull().all(axis=1)]
data = data.drop(columns_to_drop, axis=1)
data = data.drop('diff_score', axis = 1)
# tournament_data = tournament_data.drop(columns_to_drop, axis=1)

tournament_data = t_detailed

columns_to_drop.remove('result')

train_x = data.drop('result', 1).copy()
train_y = data['result'].copy()

test_x = tournament_data.drop('result', 1).copy()
test_y = tournament_data['result'].copy()

dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x, label=test_y)

bst = create_and_train_model(dtrain, dtest)
print 'model created'
