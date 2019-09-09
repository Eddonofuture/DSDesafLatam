import sys

data = (float(sys.argv[1])*float(sys.argv[2])-float(sys.argv[3]))

if(data >= 0 ):
    data = data*0.65
    print(data)
else:
    print(data)
