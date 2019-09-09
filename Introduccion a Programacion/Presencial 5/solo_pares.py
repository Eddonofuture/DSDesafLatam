numMax = int(input("Ingrese numero Maximo\n"))
#filtro por el modulo para determinar el numero par
for i in range(0 , numMax+1 ):
    if i % 2 == 0:
        print (i)
