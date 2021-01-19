import pandas as pd 
import json

def extract_dict_values(df_name, column_df):
    df_final = None
    for i, x in enumerate(df_name[column_df]):
            if i == 0:
                df_final = pd.DataFrame(x)
            else:
                df_final = pd.concat([df_final, pd.DataFrame(x)], axis=0)
    return df_final

def filter_df(df,filter_column,val1,val2=None,val3=None,val4=None,val5=None):
    """
    What it does:
        # This function filters a dataframe by column, with a mask by 1 value.It admits up to 5 values. 

    What it needs:
        # The dataframe which needs to be filtered, the column condition for the mask and the values for the column filtering.

    What it returns:
        # the filtered dataframe
        GITHUB ID: @andreeaman
        """
    mask=(df[filter_column] == val1) | (df[filter_column] == val2)|(df[filter_column] == val3)|(df[filter_column] == val4)|(df     [filter_column] == val5)
    filtered_df=df.loc[mask]
    return filtered_df


def df_covid(dt, val1=None):
    """
    What it does:
        # This function extracts the dictionary from the list in the "data" column 

    What it needs:
        # to get the final data set we use the explode function to extract the data from "data" and json_normalize to extract the columns inside "data" 

    What it returns:
        # the complete dataframe
        GITHUB ID: @mardeldom
    """

    first = dt.explode(val1)
    dt_covid= pd.json_normalize(json.loads(first.to_json(orient="records")))
    return dt_covid
    
def datetime(dt=None, val1=None):
    """
    What it does:
        # This function changes the date column from object to datetime. 

    What it needs:
        #  The function to_datetime change the data type

    What it returns:
        # Return cleaned data in a new dataframe
        GITHUB ID: @mardeldom
    """
    dt[val1]= dt[val1].apply(pd.to_datetime)
    return dt

# %%
def remove_outlier(df_in, col_name):
    """
    What it does:
        # This function accept a dataframe, remove outliers, return cleaned data in a new dataframe.

    What it needs:
        # The dataframe and the column to check.

    What it returns:
        # Return cleaned data in a new dataframe
 
    GITHUB ID: @andreeaman
    """
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

