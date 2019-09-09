def filter(dic, valor):

    new_dic = {}
    for k,v in dic.items():
        if v > valor:
            new_dic[k] = v
    return new_dic