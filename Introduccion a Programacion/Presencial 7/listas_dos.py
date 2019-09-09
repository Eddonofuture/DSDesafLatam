import numpy as np

def promedio(auto):
    autoMean = []
    campo1 = []
    campo2 = []
    campo4 = []
    
    for index , element in enumerate(auto):
        
        campo1.append(element[1])
        campo2.append(element[2])
        campo4.append(element[4])

    campo1 = np.mean(campo1)
    campo2 = np.mean(campo2)
    campo4 = np.mean(campo4)
    return campo1,campo2,campo4