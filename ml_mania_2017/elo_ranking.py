import pandas as pd
import numpy as np
from datetime import datetime

start_point = 1500
width = 400
k_factor = 32


def add_elo_ranking(rs_detailed, t_detailed):
    rs_detailed = rs_detailed.assign(type='regular_season')
    t_detailed = t_detailed.assign(type='tournament')
    data = rs_detailed.append(t_detailed)
    ids = data.lteam.append(data.wteam).unique()
    elo_ranking = dict()
    for id in ids:
        elo_ranking[id] = start_point
    data = data.assign(welo=np.NaN)
    data = data.assign(lelo=np.NaN)
    data = data.sort_values(['season', 'daynum'])
    n = 0
    for i, row in data.iterrows():
        n = n + 1
        if n % 1000 == 0:
            print str(datetime.now()) + ' ' + str(n)
        w_score = elo_ranking[row['wteam']]
        l_score = elo_ranking[row['lteam']]
        data.set_value(i, 'welo', w_score)
        data.set_value(i, 'lelo', l_score)
        change = k_factor * (1 - expected_score(w_score, l_score))
        w_score += change
        l_score -= change
        elo_ranking[row['wteam']] = w_score
        elo_ranking[row['lteam']] = l_score
    rs_detailed = data[data.type == 'regular_season']
    t_detailed = data[data.type == 'tournament']
    return rs_detailed, t_detailed


def expected_score(score_A, score_B):
    return 1.0 / (1.0 + 10 ** ((score_B - score_A) / width))
