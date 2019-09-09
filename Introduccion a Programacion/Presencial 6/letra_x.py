def letra_x(n):
    varFinal = ""
    rango = n-1
    for i in range(0 , n):
        lineaVertical = ""
        for j in range(0 , n):
            if(i==j):
                lineaVertical += "*"
            elif((i+j)== rango):
                lineaVertical += "*"
            else:
                lineaVertical += " "    
        varFinal += lineaVertical+"\n"
    return (varFinal)