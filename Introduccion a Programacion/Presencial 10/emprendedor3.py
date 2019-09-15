import sys

precioVenta = float(sys.argv[1])
numUsuarios = (float(sys.argv[2]))
gasto = float(sys.argv[3])
utilAnterior = 1000

data = ((precioVenta*(numUsuarios))- gasto)

if len(sys.argv) >= 5 :
    utilAnterior = float(sys.argv[4]) 
    data = data + utilAnterior
else:
    data = (data + utilAnterior)

data = (data * 100 ) / utilAnterior

print (data)