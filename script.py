import numpy as np
import pandas as pd

import plotly
import plotly.offline as offline
import plotly.graph_objs as go

from ml_mania_2017.model_year_regular_tournament import *
import xgboost as xgb
from sklearn import model_selection

# from ml_mania_2017.functions import *

teams = pd.read_csv('data/Teams.csv')
teams.columns = [i.lower() for i in teams.columns]

seasons = pd.read_csv('data/Seasons.csv')
seasons.columns = [i.lower() for i in seasons.columns]

rs_detailed = pd.read_csv('data/RegularSeasonDetailedResults.csv')
rs_detailed.columns = [i.lower() for i in rs_detailed.columns]

t_detailed = pd.read_csv('data/TourneyDetailedResults.csv')
t_detailed.columns = [i.lower() for i in t_detailed.columns]

t_seeds = pd.read_csv('data/TourneySeeds.csv')
t_seeds.columns = [i.lower() for i in t_seeds.columns]

t_slots = pd.read_csv('data/TourneySlots.csv')
t_slots.columns = [i.lower() for i in t_slots.columns]

ordinals = pd.read_csv('data/massey_ordinals_2003-2016.csv')

sample_submission = pd.read_csv('data/sample_submission.csv')

data_per_season = dict()

for i in range(2013, 2016 + 1):
    print str(i) + ' ' + str(datetime.now())
    data_per_season[i] = create_train_set_for_year(rs_detailed, ordinals, i)

data = pd.concat(data_per_season)

submission_per_season = dict()
for i in range(2013, 2016 + 1):
    print str(i) + ' ' + str(datetime.now())
    submission_per_season[i] = create_predict_set_for_year(sample_submission, ordinals, i)

predict_data = pd.concat(submission_per_season)

data = data[data.notnull().all(axis=1)]

model_columns = ['season'] + [i for i in data.columns if 'rank' in i] + ['result']

data = data.loc[:, model_columns]

model_columns.remove('result')

predict_data = predict_data.loc[:, model_columns]

train, test = model_selection.train_test_split(data)

train_x = train.drop('result', 1).copy()
train_y = train['result'].copy()

test_x = test.drop('result', 1).copy()
test_y = test['result'].copy()

dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x, label=test_y)

param = {'bst:max_depth': 6, 'bst:eta': 1, 'silent': 1, 'objective': 'binary:logistic', 'nthread': 7,
         'eval_metric': 'logloss'}

num_round = 500
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, evallist)

dpredict = xgb.DMatrix(predict_data)
output = bst.predict(dpredict)

sub = pd.DataFrame({'id': sample_submission.id, 'pred': output})
sub.to_csv('submissions/sub2.csv', sep=',', index=None)
