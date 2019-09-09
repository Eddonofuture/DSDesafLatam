import sys
i = 0.0
resultado = 0
numeroMaximo = float(sys.argv[1])

while i < numeroMaximo:
    i += 1
    resultado += i

print (float(resultado/i))