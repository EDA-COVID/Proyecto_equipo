import json
from flask import Flask, request, render_template
import os,sys
import pandas as pd

z_file = __file__

for i in range(4):
    z_file = os.path.dirname(z_file)
sys.path.append(z_file)

from src.utils.folders_tb import jsonlink_df
from src.utils.mining_data_tb import filter_df, df_covid

covid = jsonlink_df('https://covid.ourworldindata.org/data/owid-covid-data.json').T
covid = filter_df(covid,'location','Argentina','Russia', 'Colombia', 'Chile', 'Spain')
covid = df_covid(covid,val1="data")

covid_grouped = covid.groupby('location').mean().loc[: , ['data.new_cases']]
covid_grouped = covid_grouped.astype(int).rename(columns={"data.new_cases": "n_c_averages"})
covid_grouped.to_json('n_c_averages.json')

app = Flask(__name__) 
app.config["DEBUG"] = True

@app.route('/c_json')
def home():
    n_c_averages = covid_grouped.to_json()
    return n_c_averages

@app.route('/give_me_id', methods=['GET'])
def group_id():
    n = request.args['id']
    if n == "A86":
        appDict = {'token': 'A43649037'}
        app_json = json.dumps(appDict)
        return app_json
    else:
        return "No es el identificador correcto"

@app.route('/give_me_token', methods=['GET'])
def group_token():
    s = request.args["token"]
    if s == "A43649037":
        return home()
    else:
        return "No es el identificador correcto"

app.run() #ip 0000 crear archivo settings