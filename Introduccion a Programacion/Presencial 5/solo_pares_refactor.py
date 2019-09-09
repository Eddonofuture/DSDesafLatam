numMax = int(input("Ingrese numero Maximo\n"))
#filtro por el modulo para determinar el numero par optimizando
for i in range(1 , numMax+1 ):
    if i % 2 == 0:
        print (i)
