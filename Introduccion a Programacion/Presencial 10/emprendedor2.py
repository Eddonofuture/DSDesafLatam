import sys

precioVenta = float(sys.argv[1])
usuariosTotales = (float(sys.argv[2]))
usuariosPremium = (float(sys.argv[3]))
usuariosGratuitos = (float(sys.argv[4]))
gasto = float(sys.argv[5])

usuariosNormales = usuariosTotales - usuariosPremium - usuariosGratuitos

utilidades = ( ( (precioVenta*usuariosPremium*2) + (precioVenta*usuariosNormales) ) - gasto)
print (utilidades)
if(utilidades >= 0 ):
    utilidades = utilidades*0.65
    print("Las utilidades son" + utilidades)
else:
    print("Las utilidades son" + utilidades)