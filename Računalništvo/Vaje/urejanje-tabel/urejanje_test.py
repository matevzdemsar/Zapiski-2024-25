def zlij1(a, b):
    c = [None] * (len(a) + len(b))
    i, j, k = 0, 0, 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            c[k] = a[i]
            i = i + 1
        else:
            c[k] = b[j]
            j = j + 1
        k = k + 1
    while i < len(a):
        c[k] = a[i]
        i = i + 1
        k = k + 1
    while j < len(b):
        c[k] = b[j]
        j = j + 1
        k = k + 1
    return c

def zlij2(a, b):
    c = []
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            c.append(a[i])
            i += 1
        else:
            c.append(b[j])
            j += 1
    while i < len(a):
        c.append(a[i])
        i += 1
    while j < len(b):
        c.append(b[j])
        j += 1
    return c

def uredi_z_izbiranjem(a):
    for i in range(0, len(a)-1):
        for j in range(i+1, len(a)):
            if a[j] < a[i]:
                a[i], a[j] = a[j], a[i]

def uredi_z_vstavljanjem(a):
    for i in range(len(a)):
        for j in range(i + 1):
            if a[j] >= a[i]:
                a[j], a[i] = a[i], a[j]

def sort1(a):
    divide = len(a) // 2
    if not divide:
        return a
    else:
        return zlij1(sort1(a[:divide]), sort1(a[divide:]))

def sort2(a):
    divide = len(a) // 2
    if not divide:
        return a
    else:
        return zlij2(sort2(a[:divide]), sort2(a[divide:]))

if __name__ == '__main__':
    import timeit
    import random
    t = list(range(0,1000))
    random.shuffle(t)
    print(timeit.timeit("uredi_z_izbiranjem(t)",
                        number=100,
                        setup="from __main__ import t, uredi_z_izbiranjem"))
    random.shuffle(t)
    print(timeit.timeit("uredi_z_vstavljanjem(t)",
                        number=100,
                        setup="from __main__ import t, uredi_z_vstavljanjem"))
    random.shuffle(t)
    print(timeit.timeit("sort1(t)",
                        number=100,
                        setup="from __main__ import t, sort1"))
    random.shuffle(t)
    print(timeit.timeit("sort2(t)",
                        number=100,
                        setup="from __main__ import t, sort2"))