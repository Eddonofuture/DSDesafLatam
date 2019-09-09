import sys
i = 0.0
numeroMaximo = float(sys.argv[1])
data = 0.0

while i < numeroMaximo:
    i += 1
    data += float(input("Ingresa Dato: "))


print ("{} {}").format(float(data/numeroMaximo),int(numeroMaximo))