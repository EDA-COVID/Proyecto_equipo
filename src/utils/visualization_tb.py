# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import sys  
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def remove_outlier_con_filtro(df_inicial, df_in, col_name):
    """
    What it does:
        # This function accepts 2 dataframes df_initial(without the filtered values) &df_in(with the filtered values), remove outliers, return cleaned data in a new dataframe.

    What it needs:
        # The dataframes and the column to check.

    What it returns:
        # Return cleaned data in a new dataframe
 
    GITHUB ID: @andreeaman
    """
    # df_inicial=todos los valores menos el filtro
    # df_in= valores con filtro
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out_filtro = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    df_out=pd.concat([df_inicial,df_out_filtro])
    return df_out



# %%
def heatmap_df(df,file_name):
    """
    What it does:
        # This function shows the heatmap of the dataframe, and saves the image under the name you input.
    What it needs:
        # The dataframe and the filename
    What it returns:
        # Shows the heatmap
    GITHUB ID: @andreeaman
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.subplots(figsize=(12,8))
    sns.heatmap(df.corr())
    plt.savefig('..\\resources\\plots\\heatmap_{}.png'.format(file_name))


# %%
def detect_outliers_df(df,col_to_check,file_name):
    """
    What it does:
        # This function shows the outliers of a dataframe column and saves the image under the name you input.
    What it needs:
        # The dataframe,the column and the filename
    What it returns:
        # Shows the plot 
    GITHUB ID: @andreeaman
    """

    plt.subplots(figsize=(15,6))
    df.boxplot(patch_artist=True, sym="k.",column=col_to_check)
    plt.xticks(rotation=90)
    plt.savefig('..\\resources\\plots\\outliers_{}.png'.format(file_name))


# %%
def plot_clean_columns(df_name, df_column, y_label):

    created_df = df_name.loc[:, ['location', df_column]]
    created_df = created_df.dropna()
    created_df = created_df.loc[created_df[df_column]!=0.0]
    print(created_df.nunique())
    print('------')

    b = created_df['location'].value_counts()
    print(b)
    print('------')
    b.plot(kind='pie', autopct = "%1.0f%%", colors=['pink', 'lightblue','violet', 'lightgreen', 'gold'])
    plt.xlabel('total data')
    plt.ylabel('')
    print('------')

    for f in set(created_df['location']):
        ax = created_df[created_df['location']==f].plot(y=df_column, title=f, legend=False, xlabel='', color='lightgreen', ylabel=y_label)
        ax.set_xlim(pd.Timestamp('2020-02-01'), pd.Timestamp('2021-02-01'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gcf().autofmt_xdate()
        if f == 'Spain':
            plt.axvline('2020-03-14')
            plt.axvline('2020-06-21', color='lightblue')
            plt.axvline('2020-10-25')
        if f == 'Colombia':
            plt.axvline('2020-03-12')
            plt.axvline('2020-08-31', color='lightblue')
        if f == 'Chile':
            plt.axvline('2020-03-18')
    
def boxplots_per_country(df,file_name):
    """
    What it does:
        # This function accepts covid dataframe and shows the outliers for all columns, except date and .

    What it needs:
        # The dataframe 

    What it returns:
        # Returns a the subplots/country
 
    GITHUB ID: @andreeaman
    """


    fig, axes = plt.subplots(8, 3, figsize=(20, 30))
    #no puedo plotear las columnas: 'data.tests_units','data.date','data.total_tests'
    #27 columnas
    # 3 filas y 3 columnas
    fig.suptitle('Outliers by Country',fontsize=20)

    sns.boxplot(ax=axes[0, 0], data=df, x='location', y='population')
    sns.boxplot(ax=axes[0, 1], data=df, x='location', y='life_expectancy')
    sns.boxplot(ax=axes[0, 2], data=df, x='location', y='data.total_tests')
    sns.boxplot(ax=axes[1, 0], data=df, x='location', y='data.new_tests')
    sns.boxplot(ax=axes[1, 1], data=df, x='location', y='data.total_tests_per_thousand')
    sns.boxplot(ax=axes[1, 2], data=df, x='location', y='data.new_tests_per_thousand')
    sns.boxplot(ax=axes[2, 0], data=df, x='location', y='data.stringency_index')
    sns.boxplot(ax=axes[2, 1], data=df, x='location', y='data.new_tests_smoothed')
    sns.boxplot(ax=axes[2, 2], data=df, x='location', y='data.new_tests_smoothed_per_thousand')
    sns.boxplot(ax=axes[3, 0], data=df, x='location', y='data.total_cases')
    sns.boxplot(ax=axes[3, 1], data=df, x='location', y='data.new_cases')
    sns.boxplot(ax=axes[3, 2], data=df, x='location', y='data.total_cases_per_million')
    sns.boxplot(ax=axes[4, 0], data=df, x='location', y='data.new_cases_per_million')
    sns.boxplot(ax=axes[4, 1], data=df, x='location', y='data.new_cases_smoothed')
    sns.boxplot(ax=axes[4, 2], data=df, x='location', y='data.total_deaths')
    sns.boxplot(ax=axes[5, 0], data=df, x='location', y='data.new_deaths')
    sns.boxplot(ax=axes[5, 1], data=df, x='location', y='data.new_deaths_smoothed')
    sns.boxplot(ax=axes[5, 2], data=df, x='location', y='data.new_cases_smoothed_per_million')
    sns.boxplot(ax=axes[6, 0], data=df, x='location', y='data.total_deaths_per_million')
    sns.boxplot(ax=axes[6, 1], data=df, x='location', y='data.new_deaths_per_million')
    sns.boxplot(ax=axes[6, 2], data=df, x='location', y='data.new_deaths_smoothed_per_million')
    sns.boxplot(ax=axes[7, 0], data=df, x='location', y='data.positive_rate')
    sns.boxplot(ax=axes[7, 1], data=df, x='location', y='data.tests_per_case')
    sns.boxplot(ax=axes[7, 2], data=df, x='location', y='data.reproduction_rate')
    plt.savefig('..\\resources\\plots\\Outliers by Country{}.png'.format(file_name))

def heatmap_with_column_filters (df,col_to_filter,filter_value,col1,col2,col3,col4,col5,col6):
    """
    What it does:
        # This function shows the heatmap of the dataframe(selection of 6 columns) and saves the image under the name you input.
    What it needs:
        # The dataframe and the filename
    What it returns:
        # Shows the heatmap
    GITHUB ID: @andreeaman
    """
    df2 = df[[col_to_filter,col1,col2,col3,col4,col5,col6]]
    # correlation 
    df3 = df2[(df2[col_to_filter] == filter_value)]                       
    dfCorr = df3.drop([col_to_filter], axis=1).corr()
    fig = plt.figure(figsize=(22,10))
    plt.subplot(121)   #  subplot 1 - female
    plt.title('{}'.format(filter_value))
    sns.heatmap(dfCorr, annot=True, fmt='.2f', square=True, cmap = 'Reds_r')
    plt.savefig('..\\resources\\plots\\heatmap_{}.png'.format(filter_value))

def remove_outlier_filtro(df_inicial, df_in, col_name):
    """
    What it does:
        # This function accepts 2 dataframes df_initial(without the filtered values) &df_in(with the filtered values), remove outliers, return cleaned data in a new dataframe.

    What it needs:
        # The dataframes and the column to check.

    What it returns:
        # Return cleaned data in a new dataframe
 
    GITHUB ID: @andreeaman
    """

    # df_inicial=todos los valores menos el filtro
    # df_in= valores con filtro
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out_filtro = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    df_out=pd.concat([df_inicial,df_out_filtro])
    return df_out
# %%
