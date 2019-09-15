def letra_i(n):
    varFinal = "*"*n
    varFinal += "\n"
    for i in range(1 , n-1):
        linea = ""
        for j in range(1,n):
            #Si es par, debo dejarlo en posicion n/2
            if(n%2 == 0):
                if(n//2 == j):
                    linea += "*"
                else:
                    linea += " "
            #Si es impar, dejo en posicion n+1    
            elif((n%2 == 1)):
                if(((n//2)+1) == j):
                    linea += "*"
                else:
                    linea += " "
        varFinal += linea+"\n"   
    varFinal += "*"*n       
    return (varFinal)
  