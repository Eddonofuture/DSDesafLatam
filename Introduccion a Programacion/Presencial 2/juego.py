import sys
import random

maquina = random.randrange(3) + 1 
inputConsola = sys.argv[1]

if inputConsola == "piedra":
    if maquina == 1 :
        print ("Computador juega Piedra" )
        print ("Empataste")
    elif maquina == 2:
        print ("Computador juega Papel" )
        print ("Perdiste")
    elif maquina == 3:
        print ("Computador juega Tijera" )
        print ("Ganaste")    
elif inputConsola == "tijera" :
    if maquina == 1 :
        print ("Computador juega Piedra" )
        print ("Perdiste")
    elif maquina == 2:
        print ("Computador juega Papel" )
        print ("Ganaste")
    elif maquina == 3:
        print ("Computador juega Tijera" )
        print ("Empataste")    
elif inputConsola == "papel": 
    if maquina == 1 :
        print ("Computador juega Piedra" )
        print ("Ganaste")
    elif maquina == 2:
        print ("Computador juega Papel" )
        print ("Empataste")
    elif maquina == 3:
        print ("Computador juega Tijera" )
        print ("Perdiste")    
else:
    print ("Argumento invalido: Debe ser piedra,papel o tijera")
