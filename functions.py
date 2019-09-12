def fetch_descriptives(x):
    '''
    Variables descriptivas media,mediana, varianza
    - X -> Int/Float
    Return
    -> mean, median, var,mode
    '''
    import numpy as np
    from scipy import stats
    if isinstance(x, pd.Series) is True:
        if x.dtype != 'object':
            x = x.dropna()
            tmp_mean = np.mean(x)
            tmp_median = np.median(x)
            tmp_var = np.var(x)
            tmp_mode = stats.mode(x)
        else:
            raise TypeError('La serie no contiene datos numericos')
    else:
        raise TypeError('El dato ingresado no es una serie.')

    return tmp_mean, tmp_median,tmp_var, tmp_mode

def get_corrs_ml(df, attr):
    '''
    Obtiene la correlacion en base al vector objetivo
    '''
    columns = df.columns

    attr_name , pearson_r , abs_pearson_r  = [], [], []

    for col in columns:
        if col != attr:
            attr_name.append(col)
            pearson_r.append(df[col].corr(df[attr]))
            abs_pearson_r.append(abs(df[col].corr(df[attr])))

    features = pd.DataFrame({
        'attribute': attr_name,
        'corr': pearson_r,
        'abs_corr': abs_pearson_r
    })

    feature = features.set_index('attribute')
    return features.sort_values(by=['abs_corr'],ascending=False)

def print_df(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    '''
    Plot DataFrame en base a sus tipos de variables
    '''
    for n, i in enumerate(df):
        plt.subplot(2, 3, n+1)
        if len(df[i].value_counts()) > 2:
            sns.distplot(df[i])
            plt.title(i)
            plt.xlabel("")
        else:
            sns.countplot(y=df[i])
            plt.title(i)
            plt.xlabel("")
            plt.tight_layout()
    return

def roc_graphic(y_test, y_hat):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    '''
    Plot Curva ROC
    -> Input
    y_test , y_hat
    -> Output
    Plot
    '''
    false_positive , true_positive , threshold = roc_curve(y_test,y_hat[:,1])
    plt.plot(false_positive,true_positive,lw = 1)
    plt.plot([0,1], linestyle='--', lw=1, color='tomato')
    plt.ylabel('Verdaderos Positivos')
    plt.xlabel('Falsos Positivos')
    return