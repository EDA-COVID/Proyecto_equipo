import json
from flask import Flask, request, render_template
import os 

app = Flask(__name__) 
app.config["DEBUG"] = True


def read_json():
    with open(r"C:\Users\Usuario\Desktop\MY_GIT_HUB\Proyecto_equipo - copia\src\utils\covid_group_A.json","r") as json_file_readed:
        json_readed = json.load(json_file_readed)
    return json_readed

@app.route('/c_json')
def home():
    settings_file = os.path.dirname(__file__) + os.sep + "json1.json"
    print(settings_file)
    json_readed = read_json()
    return json_readed
app.run()
