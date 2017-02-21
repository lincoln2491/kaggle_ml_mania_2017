import pandas as pd


def load_data():
    teams = pd.read_csv('data/Teams.csv')
    teams.columns = [i.lower() for i in teams.columns]

    seasons = pd.read_csv('data/Seasons.csv')
    seasons.columns = [i.lower() for i in seasons.columns]

    rs_detailed = pd.read_csv('data/rs_detailed-elo.csv', sep = ';')
    rs_detailed.columns = [i.lower() for i in rs_detailed.columns]

    t_detailed = pd.read_csv('data/t_detailed-elo.csv', sep = ';')
    t_detailed.columns = [i.lower() for i in t_detailed.columns]

    t_seeds = pd.read_csv('data/TourneySeeds.csv')
    t_seeds.columns = [i.lower() for i in t_seeds.columns]

    t_slots = pd.read_csv('data/TourneySlots.csv')
    t_slots.columns = [i.lower() for i in t_slots.columns]

    ordinals = pd.read_csv('data/massey_ordinals_2003-2016.csv')

    return teams, seasons, rs_detailed, t_detailed, t_seeds, t_slots, ordinals
