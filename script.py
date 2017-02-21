from datetime import datetime

from ml_mania_2017.model_year_regular_tournament import *
from ml_mania_2017.functions import *
from ml_mania_2017 import elo_ranking

pd.set_option('display.width', 320)

teams, seasons, rs_detailed, t_detailed, t_seeds, t_slots, ordinals = load_data()
print str(datetime.now()) + ' data loaded'
# rs_detailed, t_detailed = elo_ranking.add_elo_ranking(rs_detailed, t_detailed)
print str(datetime.now()) + ' elo ranking added'

data_per_season = dict()
for i in range(2013, 2016 + 1):
    print str(i) + ' ' + str(datetime.now())
    data_per_season[i] = create_train_set_for_year(rs_detailed, ordinals, None, is_train=True, year=i)
data = pd.concat(data_per_season).reset_index(drop=True)

tournament_per_season = dict()
for i in range(2013, 2016 + 1):
    print str(i) + ' ' + str(datetime.now())
    tournament_per_season[i] = create_train_set_for_year(rs_detailed, ordinals, t_detailed, is_train=False, year=i)
tournament_data = pd.concat(tournament_per_season).reset_index(drop=True)
print 'data transformed'

data = data[data.notnull().all(axis=1)]
data = data.loc[:, columns_for_model]
tournament_data = tournament_data.loc[:, columns_for_model]

columns_for_model.remove('result')

train_x = data.drop('result', 1).copy()
train_y = data['result'].copy()

test_x = tournament_data.drop('result', 1).copy()
test_y = tournament_data['result'].copy()

dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x, label=test_y)

bst = create_and_train_model(dtrain, dtest)
print 'model created'
