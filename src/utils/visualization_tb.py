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


def remove_outlier(df_in, col_name):
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
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
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
def plot_clean_columns(df_name, column_1, y_label, column_2=None):

    if column_2==None:
        created_df = df_name.loc[:, ['location', column_1]]
    else:
        created_df = df_name.loc[:, ['location', column_1, column_2]] 
        created_df = created_df.loc[created_df[column_2]!=0.0]   
    
    created_df = created_df.dropna()
    created_df = created_df.loc[created_df[column_1]!=0.0]
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
        
        if column_2==None:
            ax = created_df[created_df['location']==f].plot(y=column_1, title=f, legend=False, xlabel='', color='lightgreen', ylabel=y_label)
        else: 
            ax = created_df[created_df['location']==f].plot(y=[column_1,column_2], title=f, legend=False, xlabel='', color=['lightgreen','purple'], ylabel=y_label)

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
    fig, axes = plt.subplots(7, 3, figsize=(20, 30))
    #no puedo plotear las columnas: 'data.tests_units','data.date','data.total_tests'
    #27 columnas
    # 3 filas y 3 columnas
    fig.suptitle('Outliers by Country',fontsize=20)
    sns.boxplot(ax=axes[0, 0], data=df, x='location', y='data.total_tests')
    sns.boxplot(ax=axes[0, 1], data=df, x='location', y='data.new_tests')
    sns.boxplot(ax=axes[0, 2], data=df, x='location', y='data.total_tests_per_thousand')
    sns.boxplot(ax=axes[1, 0], data=df, x='location', y='data.new_tests_per_thousand')
    sns.boxplot(ax=axes[1, 1], data=df, x='location', y='data.stringency_index')
    sns.boxplot(ax=axes[1, 2], data=df, x='location', y='data.new_tests_smoothed')
    sns.boxplot(ax=axes[2, 0], data=df, x='location', y='data.new_tests_smoothed_per_thousand')
    sns.boxplot(ax=axes[2, 1], data=df, x='location', y='data.total_cases')
    sns.boxplot(ax=axes[2, 2], data=df, x='location', y='data.new_cases')
    sns.boxplot(ax=axes[3, 0], data=df, x='location', y='data.total_cases_per_million')
    sns.boxplot(ax=axes[3, 1], data=df, x='location', y='data.new_cases_per_million')
    sns.boxplot(ax=axes[3, 2], data=df, x='location', y='data.new_cases_smoothed')
    sns.boxplot(ax=axes[4, 0], data=df, x='location', y='data.total_deaths')
    sns.boxplot(ax=axes[4, 1], data=df, x='location', y='data.new_deaths')
    sns.boxplot(ax=axes[4, 2], data=df, x='location', y='data.new_deaths_smoothed')
    sns.boxplot(ax=axes[5, 0], data=df, x='location', y='data.new_cases_smoothed_per_million')
    sns.boxplot(ax=axes[5, 1], data=df, x='location', y='data.total_deaths_per_million')
    sns.boxplot(ax=axes[5, 2], data=df, x='location', y='data.new_deaths_per_million')
    sns.boxplot(ax=axes[6, 0], data=df, x='location', y='data.new_deaths_smoothed_per_million')
    sns.boxplot(ax=axes[6, 1], data=df, x='location', y='data.tests_per_case')
    sns.boxplot(ax=axes[6, 2], data=df, x='location', y='data.reproduction_rate')
    plt.savefig('..\\resources\\plots\\Outliers by Country{}.png'.format(file_name))

def heatmap_with_column_filters (df,file_name,col1,col2,col3,col4,col5,col6):
    """
    What it does:
        # This function shows the heatmap of the dataframe(selection of 6 columns) and saves the image under the name you input.
    What it needs:
        # The dataframe and the filename
    What it returns:
        # Shows the heatmap
    GITHUB ID: @andreeaman
    """
    df2 = df[[col1,col2,col3,col4,col5,col6]]                
    df_corr = df2.corr()
    matrix = np.triu(df2.corr())
    fig = plt.figure(figsize=(22,10))
    plt.subplot(121)   #  subplot 1 - female
    plt.title('{}'.format(file_name))
    sns.heatmap(df_corr,  annot=True, fmt='.2f', square=True, cmap='Blues_r',vmin=-1, vmax=1)
    plt.savefig('..\\resources\\plots\\heatmap_{}.png'.format(file_name))



# %%
def position_countries(dt1=None, dt2=None, dt3=None):
    fig, (ax1, ax2,ax3) = plt.subplots(3, figsize=(10, 10))
    bar_width = 0.4
    def label(ax=None, dt=None):
        rects = ax.patches
        list_index = dt.index.values.tolist()
        labels = list_index
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
                    ha='center', va='bottom')
    #Primer gráfico
    ax1.bar(dt1["location"],dt1["data.total_cases_per_million"], width=bar_width, color= 'lightblue')
    label(ax=ax1, dt=dt1)
    ax1.set_xlabel("Country")
    ax1.set_ylabel("Total cases per million")
    ax1.set_ylim([0,18000])
    #Segundo gráfico
    ax2.bar(dt2["location"],dt2["data.total_deaths_per_million"], width=bar_width, color= 'violet')
    label(ax=ax2, dt=dt2)
    ax2.set_xlabel("Country")
    ax2.set_ylabel("Total deaths per million")
    ax2.set_ylim([0,800])
    #Tercer gráfico
    ax3.bar(dt3["location"],dt3["life_expectancy"], width=bar_width, color= 'lightgreen')
    label(ax=ax3, dt=dt3)
    ax3.set_xlabel("Country")
    ax3.set_ylabel("Life expectancy")
    ax3.set_ylim([0,100])
    return fig.show()

# %%
def heatmap_per_country (df):
    """
    What it does:
        # This function shows the heatmap of the dataframe,it filters it by location and saves the image under the name you input.
    What it needs:
        # The dataframe and the filename
    What it returns:
            # Shows the heatmap
    GITHUB ID: @andreeaman
    """
    fig = plt.figure(figsize = (30,50))
    cont=1
    for i in list(set(df.iloc[:,0])):         
        reg = df.loc[df['location'] == i]
        reg= reg.drop(columns=['location','life_expectancy','population'])   #we get the dataframe filter by region
        ax1 = fig.add_subplot(5, 1, cont)   #we are adding a subplot to the general figure
        ax1.title.set_text("Heatmap per country: "+i) #we set the title of the of ax1 with the current region name
        matrix = np.triu(reg.corr())
        sns.heatmap(reg.corr(),   mask=matrix,ax=ax1, cmap='Reds', annot=True, vmin=-1, vmax=1, linewidths=1)  
        #By doing ax=ax1, we are assigning the subplot ax1 the current heatmap figure
        fig.subplots_adjust(left=None, bottom=0.2, right=None, top=0.9, wspace= 1.1, hspace=0.9)
        cont=cont+1

    plt.savefig('..\\resources\\plots\\heatmap_per_country.png')
        #standard configuration
        #left = 0.125  # the left side of the subplots of the figure
        #right = 0.9   # the right side of the subplots of the figure
        #bottom = 0.1  # the bottom of the subplots of the figure
        #top = 0.9     # the top of the subplots of the figure
        #wspace = 0.2  # the amount of width reserved for space between subplots,
                    # expressed as a fraction of the average axis width
        #hspace = 0.2  # the amount of height reserved for space between subplots,
                    # expressed as a fraction of the average axis height

def columns_correlation_pivot(df,upper_val,lower_val): 
    """
    What it does:
        # Filters pairwise correlation of columns, given the reference levels
    What it needs:
        # The dataframe,and the filter values
    What it returns:
        # It return the pivot of the dataframe with the corr() for each column.
    GITHUB ID: @andreeaman
    """
    
    corr_matrix = df.corr().abs()
    sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False)).reset_index().rename(
        columns={'level_0':'column_1','level_1':'column_2',0:'Corr'})
    mayor_corr=sol[(sol['Corr']<upper_val) & (sol['Corr']>lower_val)]
    mayor_corr_pivot=mayor_corr.sort_values(by='Corr',ascending=False).pivot_table(index=['column_1','column_2'], values='Corr')
    pivot=mayor_corr_pivot.reset_index().groupby('column_1').sum().sort_values(by='Corr',ascending=False)
    return pivot

def daily_deaths_cases(dt=None):
    fig1= plt.figure(figsize=(15,8))
    fig1= sns.scatterplot(x= dt["data.date"], y=dt["data.new_cases_per_million"], hue=dt["location"], data=dt, palette="Set2")
    fig1.set_ylim([0,600])
    fig2= plt.figure(figsize=(15,8))
    fig2= sns.scatterplot(x= dt["data.date"], y=dt["data.new_deaths_per_million"], hue=dt["location"], data=dt, palette="Set2")
    fig2.set_ylim([0,15])
    fig1.set_xlabel("Date")
    fig1.set_ylabel("Daily cases per million")
    fig1.legend(title="Location")
    fig2.set_xlabel("Date")
    fig2.set_ylabel("Daily deaths per million")
    fig2.legend(title="Location")
    return fig1, fig2

def daily_deaths_cases2(dt=None):
    fig1= plt.figure(figsize=(15,8))
    fig1= sns.lineplot(x= dt["data.date"], y=dt["data.new_cases_per_million"], hue=dt["location"], data=dt, palette="Set2")
    fig1.set_ylim([0,600])
    fig2= plt.figure(figsize=(15,8))
    fig2= sns.lineplot(x= dt["data.date"], y=dt["data.new_deaths_per_million"], hue=dt["location"], data=dt, palette="Set2")
    fig2.set_ylim([0,15])
    fig1.set_xlabel("Date")
    fig1.set_ylabel("Daily cases per million")
    fig1.legend(title="Location")
    fig2.set_xlabel("Date")
    fig2.set_ylabel("Daily deaths per million")
    fig2.legend(title="Location")
    return fig1, fig2
def daily_deaths_cases3(dt=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(dt.location.unique()))
    bar_width = 0.4
    b1 = ax.bar(x, dt["data.new_cases_per_million"],
                width=bar_width)
    b2 = ax.bar(x + bar_width,  dt["data.new_deaths_per_million"],
                width=bar_width)
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(dt.location.unique())
    ax.set_ylim([0,400])
    return fig.show()

def daily_deaths_cases4(dt=None, col_name=None, title=None):
    fig1 = dt.groupby("location")[col_name].mean()
    fig1.plot(kind="pie",autopct = "%1.0f%%", colors=['pink', 'lightblue','violet', 'lightgreen', 'gold'])
    fig1= plt.xlabel(title)
    fig1= plt.ylabel('')
    return fig1

    
def plotly_dont_travel_to(df,column,chart_name):
    """
    What it does:
        # This function accepts pivot dataframe and shows the evolution in time of the column you input

    What it needs:
        # The dataframe, the column which will be shown and the chart name. 

    What it returns:
        # Returns a plotly chart detailed by country
 
    GITHUB ID: @andreeaman
    """
    fig = px.line(df, x="data.date", y=column, color='location', 
                hover_data={"data.date": "|%B %d, %Y"},
                title='Evolution per month', labels=dict(x="month", column=chart_name, location="Country"),
                width=1300, height=600)
    fig.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y")
    fig.update_traces(connectgaps=True)
    #fig.write_image("..\\resources\\plots\\Evolution_{}.png".format(chart_name))
    fig.write_html('..\\reports\\Evolution_{}.html'.format(chart_name))
    fig.show()  
   
