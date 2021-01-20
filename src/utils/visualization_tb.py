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




# %%
def plot_per_column_distribution(df, n_graph_shown, n_graph_per_row,figure_name):
    """
    What it does:
        # This function accept a dataframe and plots the histogram of each column.

    What it needs/What it returns:
        # The dataframe, the number of columns to show(n_graph_shown), also it needs the distribution of how the suplots will           be shown (n_graph_per_row) and finally the name of the file(figure_name) in order to save the image.

    GITHUB ID: @andreeaman
    """
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nrow, ncol = df.shape
    column_names = list(df)
    n_graph_row = (ncol + n_graph_per_row - 1) / n_graph_per_row
    plt.figure(num = None, figsize = (6 * n_graph_per_row, 8 * n_graph_row), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(ncol, n_graph_shown)):
        plt.subplot(n_graph_row, n_graph_per_row, i + 1)
        column_df = df.iloc[:, i]
        if (not np.issubdtype(type(column_df.iloc[0]), np.number)):
            value_counts = column_df.value_counts()
            value_counts.plot.bar()
        else:
            column_df.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{column_names[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.savefig('..\\resources\\plots\\{}.png'.format(figure_name))
    plt.show()


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