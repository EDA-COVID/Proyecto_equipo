import argparse
import os
import json

def read_json(fullpath):
    with open(fullpath,"r") as json_file_readed:
        json_readed = json.load(json_file_readed)
    return json_readed

parser = argparse.ArgumentParser()
parser.add_argument("-x", "--x", type=str, default='j18', required=True)
args = vars(parser.parse_args())

print("####################\n")
print(type(args))
print(args)

base = args["x"]

if base == 'j18':
    # result = 'json_nuestro'
    # print("\nThe final json is:", result)

    # settings_file = os.path.dirname('covid_group_A') + os.sep + "json1.json"
    settings_file = os.path.dirname(__file__) + os.sep + "json1.json"
    print(settings_file)
    json_readed = read_json(fullpath= r"C:\Users\Anais\Documents\BRIDGE\COVID\src\covid_group_A.json")
    print(json_readed)

else:
    print('Not correct')


# TO RUN: 
#python Z:\Data_Science\TheBridge\Content\Contenido_Curso\data_science_nov_2020\week7\day1\python\arg_parse\1_arg_parse.py -x 2 -y 4 -v 2


# 1
# python o python3 
# 2
# ruta al fichero 
# 3
# args
# --help

# 'python' 'ruta' 'args'
 