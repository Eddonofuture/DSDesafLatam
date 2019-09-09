def letra_o(n):
    varFinal = "*"*n
    varFinal += "\n"
    for i in range(0 , n-2):
        lineaVertical = ""
        for j in range(0,n):
            lineaHorizontal = ""
            if(j==0):
                lineaHorizontal = "*"
                lineaVertical += lineaHorizontal
            elif (j==n-1):
                lineaHorizontal = "*"
                lineaVertical += lineaHorizontal
            else:
                lineaHorizontal = " "
                lineaVertical += lineaHorizontal
        varFinal += lineaVertical+"\n"
    varFinal += "*"*n
    return (varFinal)           