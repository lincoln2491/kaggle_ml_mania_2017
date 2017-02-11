# train = data.loc[:, ['low_rank', 'high_rank']]
# labels = data.result

# tmp_r['l_rank'] = np.nan
# tmp_r['w_rank'] = np.nan
#
# tmp_r.w_rank = tmp_r.apply(lambda x: tmp_o.loc[tmp_o.loc[((tmp_o.team == x['wteam']) & (tmp_o.season == 2016) & (
#     tmp_o.sys_name == 'PGH') & (tmp_o.rating_day_num <= x['daynum'])), 'rating_day_num'].idxmax(), 'orank'] if x[
#                                                                                                                    'daynum'] >= 16 else np.nan,
#                            axis=1)
# tmp_r.l_rank = tmp_r.apply(lambda x: tmp_o.loc[tmp_o.loc[((tmp_o.team == x['lteam']) & (tmp_o.season == 2016) & (
#     tmp_o.sys_name == 'PGH') & (tmp_o.rating_day_num <= x['daynum'])), 'rating_day_num'].idxmax(), 'orank'] if x[
#                                                                                                                    'daynum'] >= 16 else np.nan,
#                            axis=1)
#
# trace_line = go.Scatter(x=range(352), y=range(352), mode="lines")
# trace_away = go.Scatter(x=tmp_r.w_rank[tmp_r.wloc == 'A'], y=tmp_r.l_rank[tmp_r.wloc == 'A'], mode='markers',
#                         name='away')
# trace_home = go.Scatter(x=tmp_r.w_rank[tmp_r.wloc == 'H'], y=tmp_r.l_rank[tmp_r.wloc == 'H'], mode='markers',
#                         name='home')
# trace_neutral = go.Scatter(x=tmp_r.w_rank[tmp_r.wloc == 'N'], y=tmp_r.l_rank[tmp_r.wloc == 'N'], mode='markers',
#                            name='neutral')
# offline.plot([trace_home, trace_away, trace_neutral, trace_line], filename='ranks')
# np.minimum()
#
#