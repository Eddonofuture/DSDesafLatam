import sys

print("Ingrese su necesidad 1.- Sumar 2.- Restar 3.-Multiplicar 4.-Dividir")

accion = int(input("Ingrese Opcion: "))
num1 = float(input("Ingrese Primer Numero: "))
num2 = float(input("Ingrese Segundo Numero: "))

if(accion == 1):
    print(float(num1 + num2))
elif (accion == 2):
    print(float(num1 - num2))
elif (accion == 3):
    print(float(num1 * num2))
elif (accion == 4):
    print(float(num1 / num2))