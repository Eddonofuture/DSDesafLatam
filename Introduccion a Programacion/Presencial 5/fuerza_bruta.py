#reduzco a minusculas
password = str(input("Ingresa contrasena\n")).lower()
#creo abecedario para comparacion
abecedario = 'abcdefghijklmnopqrstuvwxyz'
#genero iterador/acumulador
intentos = 0
for i in range(len(password)):
    for j in range(len(abecedario)):
        intentos += 1 #dsumo intentos por cada iteracion del 2 for
        if password[i] == abecedario[j]:#modificacion de burbuja
            break
print(str(intentos) + ' intentos')