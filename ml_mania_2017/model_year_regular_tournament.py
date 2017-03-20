from models_utils import *
from datetime import datetime
import pandas as pd
import xgboost as xgb
import os

ranking_names = ['PGH', 'SAG', 'DOK',
                 # 'ACU',
                 # 'BWE',
                 'MAS', 'KPK', 'MOR', 'PIG',
                 # 'TRP',
                 # 'TRK',
                 'BBT',
                 'POM']


def create_train_set_for_year(rs_detailed, ordinals, t_detailed, is_train=True, year=2016, is_sample=False):
    print str(datetime.now()) + ' start'
    if is_train:
        data = rs_detailed
        help_data = rs_detailed
    else:
        data = t_detailed
        help_data = rs_detailed

    print str(datetime.now()) + ' data set setted'
    data = data[data.season == year].copy()
    ordinals = ordinals[ordinals.season == year].copy()
    help_data = help_data[help_data.season == year].copy()
    print str(datetime.now()) + ' data set added'
    if not is_sample:
        data = change_data_l_h(data)
    help_data = change_data_l_h(help_data)
    print str(datetime.now()) + ' data changed'
    for ranking in ranking_names:
        data = add_info_about_rank_before_match(data, ordinals, year, ranking)
        print str(datetime.now()) + ' ' + ranking + ' added'

    if not is_sample:
        data = add_statistics_to_l_h_model(data)
    help_data = add_statistics_to_l_h_model(help_data)
    print str(datetime.now()) + ' statistics added'

    features_m = ['score', 'fgm', 'fga', 'fgm3', 'fga3', 'ftm', 'fta', 'or', 'dr', 'ast', 'to', 'stl', 'blk', 'pf',
                  'fgm2', 'fga2', 'tr', 'fgm3r', 'fgm2r', 'fga3r', 'fga2r', 'fgmar', 'fgma2r', 'fgma3r', 'odrr',
                  'orr', 'drr', 'elo_all', 'elo_season']
    data = data.merge(data.apply(
        lambda x: get_mean_from_last_n_matches(x['l_team'], help_data, x['daynum'], features_m), axis=1),
        left_index=True, right_index=True)
    print str(datetime.now()) + ' mean 1 added'
    data = data.merge(data.apply(
        lambda x: get_mean_from_last_n_matches(x['l_team'], help_data, x['daynum'], features_m, n=5), axis=1),
        left_index=True, right_index=True)
    print str(datetime.now()) + ' mean 2 added'
    data = data.merge(data.apply(
        lambda x: get_mean_from_last_n_matches(x['l_team'], help_data, x['daynum'], features_m, n=3), axis=1),
        left_index=True, right_index=True)
    print str(datetime.now()) + ' mean 3 added'
    data = data.merge(data.apply(
        lambda x: get_mean_from_last_n_matches(x['l_team'], help_data, x['daynum'], features_m, n=1), axis=1),
        left_index=True, right_index=True)
    print str(datetime.now()) + ' mean 4 added'
    data = data.merge(data.apply(
        lambda x: get_mean_from_last_n_matches(x['h_team'], help_data, x['daynum'], features_m, is_l=False), axis=1),
        left_index=True, right_index=True)
    print str(datetime.now()) + ' mean 5 added'
    data = data.merge(data.apply(
        lambda x: get_mean_from_last_n_matches(x['h_team'], help_data, x['daynum'], features_m, is_l=False, n=5),
        axis=1),
        left_index=True, right_index=True)
    print str(datetime.now()) + ' mean 6 added'
    data = data.merge(data.apply(
        lambda x: get_mean_from_last_n_matches(x['h_team'], help_data, x['daynum'], features_m, is_l=False, n=3),
        axis=1),
        left_index=True, right_index=True)
    print str(datetime.now()) + ' mean 7 added'
    data = data.merge(data.apply(
        lambda x: get_mean_from_last_n_matches(x['h_team'], help_data, x['daynum'], features_m, is_l=False, n=1),
        axis=1),
        left_index=True, right_index=True)
    print str(datetime.now()) + ' mean 8 added'

    return data


def create_and_train_model(dtrain, dtest):
    param = {'bst:max_depth': 10, 'bst:eta': 0.1, 'silent': 1, 'objective': 'binary:logistic', 'nthread': 7,
             'eval_metric': ['logloss', 'error']}
    num_round = 12
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, evallist)
    return bst


def predict(model, sample_submission):
    id = sample_submission.id
    sample_submission = sample_submission[model.feature_names]
    dpredict = xgb.DMatrix(sample_submission)
    pred = model.predict(dpredict)
    return pd.DataFrame(dict(Id=id, Pred=pred))


def save_submission_and_model(sample_submission, model):
    next_nr = max([int(f.replace('sub', '').replace('.csv', '')) for f in os.listdir('submissions')]) + 1
    sample_submission.to_csv('submissions/sub' + str(next_nr) + '.csv', sep=',', index=None)
    print 'submission nr: ' + str(next_nr)
    model.save_model('models/model' + str(next_nr) + '.mod')


# care about this is not copy
def add_rankings_diff(detailed):
    for ranking in ranking_names:
        ranking = ranking.lower()
        detailed.loc[:, 'diff_' + ranking] = (detailed['l_rank_' + ranking] - detailed['h_rank_' + ranking]).astype(np.float32)
    return detailed


def add_diffs_for_columns(detailed, columns):
    for column in columns:
        detailed.loc[:, 'diff_' + column] = (detailed['l_' + column] - detailed['h_' + column]).astype(np.float32)
    return detailed
