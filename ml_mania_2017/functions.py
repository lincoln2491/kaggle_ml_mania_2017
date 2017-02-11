import pandas as pd

from script import ordinals


def get_PGH_rank_before_mathch(w_team, l_team, season, day_num):
    tmp = ordinals[((ordinals.season == season) & (ordinals.rating_day_num <= day_num) & (ordinals.sys_name == 'PGH'))]
    tmp_o.loc[tmp_o.loc[((tmp_o.team == 1462) & (tmp_o.season == 2016) & (ordinals.sys_name == 'PGH') & (
    tmp_o.rating_day_num <= 130)), 'rating_day_num'].idxmax(), 'orank']


tmp_r.apply(lambda x: tmp_o.loc[tmp_o.loc[((tmp_o.team == x['wteam']) & (tmp_o.season == 2016) & (ordinals.sys_name == 'PGH') & (tmp_o.rating_day_num <= 130)), 'rating_day_num'].idxmax(), 'orank'], axis =1)

tmp_r.iloc[500:510].apply(lambda x: tmp_o.loc[tmp_o.loc[((tmp_o.team == x['wteam']) & (tmp_o.season == 2016) & (tmp_o.sys_name == 'PGH') & (tmp_o.rating_day_num <= x['daynum'])), 'rating_day_num'].idxmax(), 'orank'], axis =1)