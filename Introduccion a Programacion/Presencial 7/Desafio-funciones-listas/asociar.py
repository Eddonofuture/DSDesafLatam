import numpy as np

def zip():

    velocidad = [4, 4, 7, 7, 8, 9, 10, 10, 10,
    11, 11, 12, 12, 12, 12, 13, 13,
    13, 13, 14, 14, 14, 14, 15, 15,
    15, 16, 16, 17, 17, 17, 18, 18,
    18, 18, 19, 19, 19, 20, 20, 20,
    20, 20, 22, 23, 24, 24, 24, 24, 25]
    distancia = [2, 10, 4, 22, 16, 10, 18, 26, 34,
    17, 28, 14, 20, 24, 28, 26, 34, 34,
    46, 26, 36, 60, 80, 20, 26, 54, 32,
    40, 32, 40, 50, 42, 56, 76, 84, 36,
    46, 68, 32, 48, 52, 56, 64, 66, 54,
    70, 92, 93, 120, 85]
    asociar = []
    asociacion1 = []
    asociacion2 = []
    asociacion3 = []
    asociacion4 = []
    promedioVelocidad = np.mean(velocidad)
    promedioDistancia = np.mean(distancia)
    
    for pair in ((velocidad[i], distancia[i]) for i in range(min(len(velocidad), len(distancia)))):
        asociar.append(pair)

    for index, value in enumerate(asociar):
        if (value[0] <= promedioVelocidad):
            asociacion1.append(value)
        if (value[0] <= promedioVelocidad and value[1] >= promedioDistancia):
            asociacion2.append(value)
        if (value[0] >= promedioVelocidad):
            asociacion3.append(value)
        if (value[0] >= promedioVelocidad and value[1] <= promedioDistancia):
            asociacion4.append(value)

    return ([len(asociacion1)],[len(asociacion2)],[len(asociacion3)],[len(asociacion4)])

zip()
