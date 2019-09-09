import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def funcion(dataFrame,var,print_list = False):
    ''' Funcion que permite retornar la cantidad de casos perdidos y su % correspondiente'''
    casos_perdidos = dataFrame[var].isnull().sum()
    porcentaje = casos_perdidos/len(dataFrame)
    
    if(print_list == True):
        list = [dataFrame[dataFrame[var].isna()]['ccodealp']]
        print(list)
    return casos_perdidos,porcentaje
def graph(dataframe,var,sample_mean=False,true_mean=False):
    '''Obtiene el plot de la media, ya sea por la del ejemplo o completa'''
    plt.figure()
    if(sample_mean == True):
        plt.hist(dataframe[var],bins= 30)
        plt.axvline(dataframe[var].mean(),color='tomato',linestyle='--',)
    if(true_mean == True):
        plt.axvline(df[var].mean(),color='green',linestyle='--',)
    return 
def dotplot(dataFrame, plot_var,plot_by,global_stat = False,statistic = 'mean'):
    plt.figure()
    if(statistic == 'mean'):
        plt.plot(dataFrame[plot_var],dataFrame[plot_by],'o')
        plt.axvline(dataFrame[plot_var].mean(),color='tomato',linestyle='--',)
    
    elif(statistic == 'median'):
        plt.plot(dataFrame[plot_var],dataFrame[plot_by],'o')
        plt.axvline(dataFrame[plot_var].median(),color='tomato',linestyle='--',)
	
    elif(statistic == 'zscore'):
        pb = dataFrame.groupby(plot_by)['Z_'+plot_var].mean()
        plt.plot(pb.values,pb.index,'o')
        plt.axvline(pb.mean(),color='tomato',linestyle='--',)
    return 