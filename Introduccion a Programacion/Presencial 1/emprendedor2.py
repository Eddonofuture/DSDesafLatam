import sys

precioVenta = float(sys.argv[1])
usuariosNormales = (float(sys.argv[2]))
usuariosPremium = (float(sys.argv[3]))
usuariosGratuitos = (float(sys.argv[4]))
gasto = float(sys.argv[5])

utilidades = ( ( (precioVenta*usuariosPremium*2) + (precioVenta*usuariosNormales) ) - gasto)

if(utilidades >= 0 ):
    utilidades = utilidades*0.65
    print(utilidades)
else:
    print(utilidades)