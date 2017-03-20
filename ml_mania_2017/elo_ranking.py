import pandas as pd
import numpy as np
from datetime import datetime

start_point = 1500
width = 400
k_factor = 32


def add_elo_ranking(rs_detailed, t_detailed, sample_submission):
    rs_detailed = rs_detailed.assign(type='regular_season')
    t_detailed = t_detailed.assign(type='tournament')
    data = rs_detailed.append(t_detailed)
    ids = data.lteam.append(data.wteam).unique()
    elo_ranking = dict()
    elo_ranking_season = dict()
    for id in ids:
        elo_ranking[id] = start_point
        elo_ranking_season[id] = start_point

    data = data.assign(welo=np.NaN)
    data = data.assign(lelo=np.NaN)

    data = data.assign(weloseason=np.NaN)
    data = data.assign(leloseason=np.NaN)
    data = data.sort_values(['season', 'daynum'])
    n = 0
    curr_season = None
    for i, row in data.iterrows():
        n = n + 1
        if n % 1000 == 0:
            print str(datetime.now()) + ' ' + str(n)
        if row['season'] != curr_season:
            for id in ids:
                elo_ranking_season[id] = start_point
            curr_season = row['season']
        w_score = elo_ranking[row['wteam']]
        l_score = elo_ranking[row['lteam']]
        w_score_season = elo_ranking_season[row['wteam']]
        l_score_season = elo_ranking_season[row['lteam']]
        data.set_value(i, 'welo', w_score)
        data.set_value(i, 'lelo', l_score)
        data.set_value(i, 'weloseason', w_score_season)
        data.set_value(i, 'leloseason', l_score_season)
        change = k_factor * (1 - expected_score(w_score, l_score))
        w_score += change
        l_score -= change
        elo_ranking[row['wteam']] = w_score
        elo_ranking[row['lteam']] = l_score
        if row['type'] == 'regular_season':
            change_season = k_factor * (1 - expected_score(w_score_season, l_score_season))
            w_score_season += change_season
            l_score_season -= change_season
            elo_ranking_season[row['wteam']] = w_score_season
            elo_ranking_season[row['lteam']] = l_score_season

    rs_detailed = data[data.type == 'regular_season']
    t_detailed = data[data.type == 'tournament']

    sample_submission = sample_submission.assign(l_elo_all=sample_submission.l_team.map(elo_ranking))
    sample_submission = sample_submission.assign(h_elo_all=sample_submission.h_team.map(elo_ranking))

    sample_submission = sample_submission.assign(l_elo_season=sample_submission.l_team.map(elo_ranking_season))
    sample_submission = sample_submission.assign(h_elo_season=sample_submission.h_team.map(elo_ranking_season))
    return rs_detailed, t_detailed, sample_submission


def expected_score(score_A, score_B):
    return 1.0 / (1.0 + 10 ** ((score_B - score_A) / width))
