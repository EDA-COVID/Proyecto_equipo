import argparse
import os, sys
import json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-x", "--x", type=str, default='j18', required=True)
args = vars(parser.parse_args())

print(args) 

base = args["x"]

if base == 'j18':

    z_file = __file__

    for i in range(4):
        z_file = os.path.dirname(z_file)
        sys.path.append(z_file)

    from src.utils.folders_tb import jsonlink_df
    from src.utils.mining_data_tb import filter_df, df_covid

    covid = jsonlink_df('https://covid.ourworldindata.org/data/owid-covid-data.json').T
    covid = filter_df(covid,'location','Argentina','Russia', 'Colombia', 'Chile', 'Spain')
    covid = df_covid(covid,val1="data")

    covid_grouped = covid.groupby('data.date').mean().loc[: , ['data.new_cases']]
    covid_grouped = covid_grouped.rename(columns={"data.new_cases": "n_c_averages"})
    n_c_averages = covid_grouped.to_json()
    print('-------\n')
    print(n_c_averages)

else:
    print('not correct')

 