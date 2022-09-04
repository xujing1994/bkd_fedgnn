import math

def encode(indx_sign, BASE=2):
    POW = int(32 / math.log(BASE,2)) 
    length = len(indx_sign)
    n = length % POW
    if n != 0:
        length += POW - n
        add = [0 for i in range(POW - n)]
        indx_sign += add
    tmp = []
    for j in range(0, length, POW):
        res = 0
        for t in range(POW):
            res += indx_sign[j + t] * (BASE ** (POW - 1 - t))
        tmp.append(res)

    return tmp


def decode(bf, BASE=2):
    POW = int(32 / math.log(BASE,2)) 
    indx_sign = []
    for a in bf:
        tmp = []
        for t in range(POW):
            tmp.append(a // (BASE ** (POW - 1 - t)))
            a %= (BASE ** (POW - 1 - t))
        indx_sign.extend(tmp)

    return indx_sign



#
if __name__ == "__main__":
    import numpy as np
    size = 10
    a = np.random.randint(256, size=size).tolist()
    print(a)
    b = encode(a,256)
    print(b)
    c = decode(b,256)[:10]
    print(c)
