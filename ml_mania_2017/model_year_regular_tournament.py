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


def create_train_set_for_year(rs_detailed, ordinals, t_detailed, is_train=True, year=2016):
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
    data = change_data_l_h(data)
    help_data = change_data_l_h(help_data)
    print str(datetime.now()) + ' data changed'
    for ranking in ranking_names:
        data = add_info_about_rank_before_match(data, ordinals, year, ranking)
        print str(datetime.now()) + ' ' + ranking + ' added'

    data = add_statistics_to_l_h_model(data)
    help_data = add_statistics_to_l_h_model(help_data)
    print str(datetime.now()) + ' statistics added'

    features_m10 = ['score', 'fgm', 'fga', 'fgm3', 'fga3', 'ftm', 'fta', 'or', 'dr', 'ast', 'to', 'stl', 'blk', 'pf',
                    'fgm2', 'fga2', 'tr', 'fgm3r', 'fgm2r', 'fga3r', 'fga2r', 'fgmar', 'fgma2r', 'fgma3r', 'odrr',
                    'orr', 'drr']
    data = data.merge(data.apply(
        lambda x: get_mean_from_last_n_matches(x['l_team'], help_data, x['daynum'], features_m10), axis=1),
        left_index=True, right_index=True)
    print str(datetime.now()) + ' mean 1 added'
    data = data.merge(data.apply(
        lambda x: get_mean_from_last_n_matches(x['l_team'], help_data, x['daynum'], features_m10, n=5), axis=1),
        left_index=True, right_index=True)
    print str(datetime.now()) + ' mean 2 added'
    data = data.merge(data.apply(
        lambda x: get_mean_from_last_n_matches(x['l_team'], help_data, x['daynum'], features_m10, n=3), axis=1),
        left_index=True, right_index=True)
    print str(datetime.now()) + ' mean 3 added'
    data = data.merge(data.apply(
        lambda x: get_mean_from_last_n_matches(x['l_team'], help_data, x['daynum'], features_m10, n=1), axis=1),
        left_index=True, right_index=True)
    print str(datetime.now()) + ' mean 4 added'
    data = data.merge(data.apply(
        lambda x: get_mean_from_last_n_matches(x['h_team'], help_data, x['daynum'], features_m10, is_l=False), axis=1),
        left_index=True, right_index=True)
    print str(datetime.now()) + ' mean 5 added'
    data = data.merge(data.apply(
        lambda x: get_mean_from_last_n_matches(x['h_team'], help_data, x['daynum'], features_m10, is_l=False, n=5),
        axis=1),
        left_index=True, right_index=True)
    print str(datetime.now()) + ' mean 6 added'
    data = data.merge(data.apply(
        lambda x: get_mean_from_last_n_matches(x['h_team'], help_data, x['daynum'], features_m10, is_l=False, n=3),
        axis=1),
        left_index=True, right_index=True)
    print str(datetime.now()) + ' mean 7 added'
    data = data.merge(data.apply(
        lambda x: get_mean_from_last_n_matches(x['h_team'], help_data, x['daynum'], features_m10, is_l=False, n=1),
        axis=1),
        left_index=True, right_index=True)
    print str(datetime.now()) + ' mean 8 added'

    return data


def create_and_train_model(dtrain, dtest):
    param = {'bst:max_depth': 10, 'bst:eta': 0.1, 'silent': 1, 'objective': 'binary:logistic', 'nthread': 7,
             'eval_metric': ['logloss', 'error']}
    num_round = 100
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, evallist)
    return bst


def create_submission_file_and_save_model(model, dpredict, sample_submission):
    output = model.predict(dpredict)

    sub = pd.DataFrame({'id': sample_submission.id, 'pred': output})

    next_nr = max([int(f.replace('sub', '').replace('.csv', '')) for f in os.listdir('submissions')]) + 1
    sub.to_csv('submissions/sub' + str(next_nr) + '.csv', sep=',', index=None)
    print 'submission nr: ' + str(next_nr)
    model.save_model('models/model' + str(next_nr) + '.mod')
    return sub
