import numpy as np

def promedio(auto):
    listado = []
    
    for i,_ in enumerate(auto):   
        if(auto[i][3] == True):
            listado.append(auto[i][0])
    return listado
