def depositar(monto , cantidad):
    saldo = monto + cantidad
    return saldo

def girar(monto , cantidad):
    if(cantidad <= monto):
        saldo = monto - cantidad
        return saldo
    else: 
        return False    
  
def mostrar_menu(saldo = 0):
    monto = saldo
    opcion = ""
    print("Bienvenido al portal del Banco Amigo. Escoja una opcion: ")
    print("1. Consultar saldo ")
    print("2. Hacer deposito ")
    print("3. Realizar giro ")
    print("4. Salir")
    opcion = input("")
    if(opcion == "1"):
        print("Su saldo es: " + monto)
        mostrar_menu(monto)
    elif(opcion == "2"):
        cantidad = int(input("Ingrese cantidad a depositar "))
        newSaldo = (depositar(monto,cantidad))
        print ("El nuevo saldo es " + str(newSaldo))
        mostrar_menu(newSaldo)
    elif(opcion == "3"):
        if(monto <= 0 ):
            print("No puede realizar giros. Su saldo es 0")
        else:
            retiro = int(input("Cantidad a Girar"))
           # if(girar(monto , retiro)    
    elif(opcion == "4"):
        return ""
    else:
        print("Opción inválida. Por favor ingrese 1, 2, 3 ó 4.")
        mostrar_menu(monto)

mostrar_menu()