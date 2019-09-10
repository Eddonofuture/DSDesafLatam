import numpy as np
from scipy import stats
def fetch_descriptives(x):
    '''
    Variables descriptivas media,mediana, varianza
    - X -> Int/Float
    Return
    -> mean, median, var,mode
    '''
    x = x.dropna()
    tmp_mean = np.mean(x)
    tmp_median = np.median(x)
    tmp_var = np.var(x)
    tmp_mode = stats.mode(x)
    return tmp_mean, tmp_median,tmp_var, tmp_mode