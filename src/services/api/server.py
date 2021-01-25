import json
from flask import Flask, request, render_template
import os, sys
import pandas as pd

z_file = __file__

for i in range(4):
    z_file = os.path.dirname(z_file)
sys.path.append(z_file)

from src.utils.folders_tb import jsonlink_df, read_json
from src.utils.mining_data_tb import filter_df, df_covid

covid = jsonlink_df('https://covid.ourworldindata.org/data/owid-covid-data.json').T
covid = filter_df(covid,'location','Argentina','Russia', 'Colombia', 'Chile', 'Spain')
covid = df_covid(covid,val1="data")

covid_grouped = covid.groupby('location').mean().loc[: , ['data.new_cases']]
covid_grouped = covid_grouped.astype(int).rename(columns={"data.new_cases": "n_c_averages"})
covid_grouped.to_json('n_c_averages.json')

app = Flask(__name__) 

@app.route("/")
def home():
    return app.send_static_file('greet.html')

@app.route("/greet")
def greet():
    a = request.args.get('id')
    if a == "A86":
        appDict = {'token': 'A43649037'}
        app_json = json.dumps(appDict)
        return app_json
    else:
        return "No es el identificador correcto"

def json_A():
    n_c_averages = covid_grouped.to_json()
    return n_c_averages

@app.route('/give_me_token', methods=['GET'])
def group_token():
    s = request.args["token"]
    if s == "A43649037":
        return json_A()
    else:
        return "No es el identificador correcto"

def main():
    print("---------STARTING PROCESS---------")
    print(__file__)

    settings_file = os.path.dirname(__file__) + os.sep + "settings.json"
    print(settings_file)
    # Load json from file
    json_readed = read_json(fullpath=settings_file)
    
    # Load variables from jsons
    SERVER_RUNNING = json_readed["server_running"]
    print("SERVER_RUNNING", SERVER_RUNNING)
    if SERVER_RUNNING:
        DEBUG = json_readed["debug"]
        HOST = json_readed["host"]
        PORT_NUM = json_readed["port"]
        app.run(debug=DEBUG, host=HOST, port=PORT_NUM)
    else:
        print("Server settings.json doesn't allow to start server. " + 
              "Please, allow it to run it.")

if __name__ == "__main__":
    main()