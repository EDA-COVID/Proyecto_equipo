import pandas as pd 

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