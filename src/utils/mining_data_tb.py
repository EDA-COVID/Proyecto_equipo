import pandas as pd 

def extract_dict_values(df_name, column_df):
    df_final = None
    for i, x in enumerate(df_name[column_df]):
            if i == 0:
                df_final = pd.DataFrame(x)
            else:
                df_final = pd.concat([df_final, pd.DataFrame(x)], axis=0)
    return df_final