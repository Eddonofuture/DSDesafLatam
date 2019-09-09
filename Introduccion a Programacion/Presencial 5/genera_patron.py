numMax = int(input("Ingrese numero Maximo\n"))
#arrastro los datos para realizar una sola impresion
acumulador = ""
acumFinal = ""
for i in range(1 , numMax+1 ):
        #asigno valor nuevo
        acumulador = acumulador + str(i)
        #acumulo valor
        acumFinal += acumulador + "\n"
print (acumFinal)
# en un solo print        
