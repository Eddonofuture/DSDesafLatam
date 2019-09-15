numMax = int(input("Ingrese numero Maximo\n"))
#acumulo los registros en la iteracion
acumulador = 0
for i in range(0 , numMax+1 ):
    if i % 2 == 0:
        acumulador += i
print (acumulador)
