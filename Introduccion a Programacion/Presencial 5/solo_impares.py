numMax = int(input("Ingrese numero Maximo\n"))
#filtro por el modulo para determinar el numero impar
for i in range(0 , numMax+1 ):
    if i % 2 == 1:
        print (i)
