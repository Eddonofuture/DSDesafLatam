cuenta_regresiva = int(input("Ingrese un numero para comenzar la cuenta\n"))
#defino un contador ya que modificamos un for para un while
i = 0
while i < cuenta_regresiva:
    print("Iteracion {}".format(cuenta_regresiva))
    cuenta_regresiva -= 1
    