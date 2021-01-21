import json
from flask import Flask, request, render_template
import os 

app = Flask(__name__) 
app.config["DEBUG"] = True

def read_json(fullpath):
    with open(fullpath,"r") as json_file_readed:
        json_readed = json.load(json_file_readed)
    return json_readed

@app.route('/c_json')
def home():
    settings_file = os.path.dirname(__file__) + os.sep + "json1.json"
    print(settings_file)
    json_readed = read_json(fullpath= r"C:\Users\Usuario\Desktop\MY_GIT_HUB\Proyecto_equipo - copia\src\utils\covid_group_A.json")

    return json_readed


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

app.run()