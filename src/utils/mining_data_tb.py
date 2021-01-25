import pandas as pd 
import json
import os, sys

z_file = __file__

for i in range(2):
    z_file = os.path.dirname(z_file)
sys.path.append(z_file)

from utils.folders_tb import jsonlink_df

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

def data_complete_clean():
    """
    What it does:
        # This function includes all the steps necessary to obtain the definitive dataset. 
        #It opens the json in python from the url to keep it up to date, extracts the data column, which is made up of a dictionary, eliminates the columns with too many nan and modifies the type of the data.date column from object to datetime. 

    What it needs:
        # we need to know which columns dominate the nan values in order to eliminate them

    What it returns:
        # Return cleaned data in a new dataframe
 
    GITHUB ID: @mardeldom
    """
    covid_complete = jsonlink_df('https://covid.ourworldindata.org/data/owid-covid-data.json').T
    covid_complete= df_covid(covid_complete,val1="data")
    covid_complete= covid_complete.drop(["continent", "extreme_poverty", "human_development_index","hospital_beds_per_thousand", "diabetes_prevalence","female_smokers","cardiovasc_death_rate", "aged_70_older", "aged_65_older", "median_age", "population_density", "gdp_per_capita", "male_smokers","data.new_vaccinations","data.total_vaccinations_per_hundred","data.total_vaccinations","data.weekly_hosp_admissions_per_million","data.weekly_hosp_admissions","data.weekly_icu_admissions_per_million","data.weekly_icu_admissions","data.new_vaccinations_smoothed","data.new_vaccinations_smoothed_per_million","data.hosp_patients","data.hosp_patients_per_million","data.icu_patients","data.icu_patients_per_million","handwashing_facilities"],axis=1)
    covid_complete= datetime(dt=covid_complete, val1="data.date")
    return covid_complete

def data_paises_clean():
    """
    What it does:
        #This function includes all the steps necessary to obtain the definitive dataset, by filtering the countries we are going to analyse.
        #It opens the json in python from the url to keep it up to date, extracts the data column, which is made up of a dictionary, eliminates the columns with too many nan and modifies the type of the data.date column from object to datetime. 

    What it needs:
        # we need to know which columns dominate the nan values in order to eliminate them

    What it returns:
        # Return cleaned data in a new dataframe
 
    GITHUB ID: @mardeldom
    """
    covid_paises = jsonlink_df('https://covid.ourworldindata.org/data/owid-covid-data.json').T
    covid_paises= filter_df(covid_paises,'location','Argentina','Russia', 'Colombia', 'Chile', 'Spain')
    covid_paises= df_covid(covid_paises,val1="data")
    covid_paises= covid_paises.drop(["continent", "extreme_poverty", "human_development_index","hospital_beds_per_thousand", "diabetes_prevalence","female_smokers","cardiovasc_death_rate", "aged_70_older", "aged_65_older", "median_age", "population_density", "gdp_per_capita", "male_smokers","data.new_vaccinations","data.total_vaccinations_per_hundred","data.total_vaccinations","data.weekly_hosp_admissions_per_million","data.weekly_hosp_admissions","data.weekly_icu_admissions_per_million","data.weekly_icu_admissions","data.new_vaccinations_smoothed","data.new_vaccinations_smoothed_per_million","data.hosp_patients","data.hosp_patients_per_million","data.icu_patients","data.icu_patients_per_million","handwashing_facilities"],axis=1)
    covid_paises= datetime(dt=covid_paises, val1="data.date")
    return covid_paises

def group(dt=data_complete_clean(),col1=None, col2=None, col3=None, col4=None):
    """
    What it does:
        #This function groups all countries with respect to total infected, total deaths and life expectancy. 
    
    What it needs:
        # we need to filter by the columns we want to analyse

    What it returns:
        # Return cleaned data in a new dataframe
 
    GITHUB ID: @mardeldom
    """
    covid = dt.loc[:,[col1, col2, col3, col4]]
    
    group_col= covid.groupby(col1).mean()
    return group_col

def sort_columns(dt=None, col_name=None):
     """
    What it does:
        #This function sorts the countries from smallest to largest according to the values offered by each column. 
    
    What it needs:
        # we need to sort by the columns we want to analyse

    What it returns:
        # Return cleaned data in a new dataframe
 
    GITHUB ID: @mardeldom
    """
    dt_col= dt.sort_values(col_name, ascending=True).reset_index()
    mask=dt_col.loc[(dt_col["location"]=="Argentina") | (dt_col["location"]=="Chile") | (dt_col["location"]=="Spain") | (dt_col["location"]=="Colombia") |(dt_col["location"]=="Russia")]
    return mask


def pivot_table_from_df (df,col1,col2,col3):  
    """
    What it does:
        # This function accepts dataframe and transform's it into a pivot table.

    What it needs:
        # The dataframe, the columns which will apear in the pivot

    What it returns:
        # Returns the reindexed dataframe
 
    GITHUB ID: @andreeaman
    """

    dont_travel=df.pivot_table(index=['data.date','location'], 
                                values=[col1,col2,col3])
    return dont_travel.reset_index()

def dont_travel_to(df,column):
    """
    What it does:
        # This function indetifies when the column reaches it maximum value and prints the date corresponding to it 

    What it needs:
        # The dataframe, the column which will be analized 

    What it returns:
        # Returns the max values detailed by country and date
 
    GITHUB ID: @andreeaman
    """

    for i in list(set(df.iloc[:,1])):         
        reg = df.loc[df['location'] == i]

        max_value =reg[column].max()
        df_max_value=df[df[column]==max_value]

        loc=df_max_value['location'].tolist()
        date=df_max_value['data.date'].tolist()
        value=df_max_value[column].tolist()



        print('*************************************')
        print("Don't travel to: ")
        print(str([elem for elem in loc])+'\n')
        print("On:")
        print(str([elem for elem in date])+'\n')
        print("{}".format(column))
        print(str([elem for elem in value])+'\n')

def replace_outlier_with_nan(df_in, col_name):
    """
    What it does:
        # This function accepts a dataframes , remove outliers, return cleaned data in a new dataframe.

    What it needs:
        # The dataframes and the column to check.

    What it returns:
        # Return cleaned data in a new dataframe
 
    GITHUB ID: @andreeaman
    """

    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr

    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)] #seleciono los valores que no tienen outliers
    df_outliers=df_in.loc[(df_in[col_name] < fence_low) | (df_in[col_name] > fence_high)] # seleciono los valores que si tienen
    df_outliers[col_name]=None
    df_outliers[col_name]=pd.to_numeric(df_outliers[col_name])
    df_final= pd.concat([df_out,df_outliers])
    
    return df_final