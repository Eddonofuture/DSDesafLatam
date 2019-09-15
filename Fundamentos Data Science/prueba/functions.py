def printUniques(df, columnas):
    '''
    Funcion obtiene el listado de datos unicos del DataFrame
    Input (DataFrame , Array[])
    Output none
    '''
    for col in columnas:
        print(col , df[col].unique())

def replaceAll(df,listado, strObjetivo):
    import pandas as pd
    df = df.replace(listado,strObjetivo)
    return df

def concise_summary(mod,print_fit=True):
    '''
    Funcion permite entregar el sumario de una regresion
    Input (Regresion , Boolean)
    Output print
    '''
    import pandas as pd
    fit = pd.DataFrame({'Statistics': mod.summary2().tables[0][0][2:],
                       'Value': mod.summary2().tables[0][3][2:]})
    estimates = pd.DataFrame(mod.summary2().tables[1].loc[:,'Coef.':'Std.Err.'])
    if print_fit is True:
        print("\nGoodnes of Fit statistics\n", fit)
    print("\nPoint Estimates\n\n", estimates)
    
def logitCalculate(columns, df):
    '''
    Funcion que permite calculo de parametros de statsmodel con su logit incluyendo su interceptor
    Input: (List, DataFrame)
    Output: (int)
    '''
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    estimate_y = 0
    for i in columns:
        estimate_y += df.params[i]
    return invlogit(estimate_y)

def invlogit(x):
    '''
    Funcion apoyo, permite calculo logaritmico
    input: int
    output: int
    '''
    import numpy as np
    return 1/(1+ np.exp(-x))

def report_scores(Y_test,predict):
    '''
    Reporteria Scores en regresiones
    input: (List , List)
    output: none
    '''
    from sklearn.metrics import mean_squared_error, r2_score
    mse_modelo = mean_squared_error(Y_test, predict).round(1)
    r2_modelo = r2_score(Y_test,predict).round(1)
    print(mse_modelo,r2_modelo)

def fetch_features(df , vector = 'income'):
    '''
    Entrega features de un dataframe
    input: (Dataframe, string)
    output: Dictionary
    '''
    import pandas as pd
    columns = df.columns
    attr_name,pearson_r,abs_pearson_r = [],[],[]

    for col in columns:
        if col != vector:
            attr_name.append(col)
            pearson_r.append(df[col].corr(df[vector]))
            abs_pearson_r.append(abs(df[col].corr(df[vector])))
    features = pd.DataFrame(
        {'attribute': attr_name,
        'corr': pearson_r,
        'abs_corr': abs_pearson_r}
    )   

    features = features.set_index('attribute')
    return features.sort_values(by=['abs_corr'], ascending =False)
