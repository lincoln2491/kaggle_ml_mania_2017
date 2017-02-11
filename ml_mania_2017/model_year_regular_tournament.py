from models_utils import *
import pandas as pd
from datetime import datetime

ranking_names = ['PGH', 'SAG', 'DOK',
                 # 'ACU',
                 # 'BWE',
                 'MAS', 'KPK', 'MOR', 'PIG',
                 # 'TRP',
                 # 'TRK',
                 'BBT',
                 'POM']


def create_train_set_for_year(rs_detailed, ordinals, year=2016):
    rs_detailed = rs_detailed[rs_detailed.season == year].copy()
    ordinals = ordinals[ordinals.season == year].copy()
    rs_detailed = change_data_l_h(rs_detailed)
    for ranking in ranking_names:
        rs_detailed = add_info_about_rank_before_match(rs_detailed, ordinals, year, ranking)
    return rs_detailed


# TODO optimize
def create_predict_set_for_year(sample_submission, ordinals, year=2016):
    sample_submission[['season', 'l_team', 'h_team']] = sample_submission.id.apply(
        lambda x: pd.Series([int(i) for i in x.split('_')]))
    print 'splitted'
    sample_submission = sample_submission[sample_submission.season == year].copy()
    ordinals = ordinals[ordinals.season == year].copy()
    ordinals = ordinals.sort_values('rating_day_num', ascending=False).groupby(
        ['sys_name', 'team']).first().reset_index().copy()
    sample_submission = sample_submission.assign(daynum=134)
    for ranking in ranking_names:
        print ranking + ' ' + str(year) + ' ' + str(datetime.now())
        sample_submission = add_info_about_rank_before_match(sample_submission, ordinals, year, ranking)

    return sample_submission

