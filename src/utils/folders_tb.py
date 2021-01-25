import pandas as pd
import requests
import json

def jsonlink_df(url_json):
    
    """
    What it does:
        # This function opens a json in python from a given url/link

    What it needs:
        # The url of the destination file, must be a string.

    What it returns:
        # The opened json as dataframe 
        GITHUB ID: @anaisvh
        """

    r = requests.get(url=url_json)
    json_readed = json.loads(r.text)
    df = pd.DataFrame(json_readed)
    return df

def read_json(fullpath):

    with open(fullpath,"r") as json_file_readed:
        json_readed = json.load(json_file_readed)
    return json_readed