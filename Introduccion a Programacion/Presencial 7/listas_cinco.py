import numpy as np

def promedio(auto):
    listado = []
    promedio = []
    lastListado = []
    for index , element in enumerate(auto):
        listado.append(element[1])
        promedio = np.mean(listado)
    
    for i,_ in enumerate(auto):   
        if(auto[i][1] >= promedio):
            lastListado.append(auto[i])
    return lastListado
