"""
Gre za vodeno preiskovanje. Cilj je najti rešitev, ki je hitrejša od običajnega preiskovanja,
npr. v širino ali globino.
"""
import random
def sestavi(n):
    out = []
    tmp = []
    for i in range(1, n ** 2):
        tmp.append(i)
    tmp.append(None)
    random.shuffle(tmp)
    for i in range(n):
        row = tmp[:n]
        out.append(row)
        tmp = tmp[n:]
    return out

def resitev(n):
    r = [[n * i + j for j in range(1, n + 1)]for i in range(n)]
    r[-1][-1] = None
    return r


class Stanje:

    def __init__(self, stanje):
        self.stanje = stanje
        self.prazno_mesto = [(i, j) for i, v in enumerate(self.stanje) for j, vr in enumerate(v) if self.stanje[i][j] is None][0]
    
    def __repr__(self):
        return f"Stanje({self.stanje})"
    
    def __str__(self):
        niz = ""
        for vrstica in self.stanje:
            niz += str(vrstica) + "\n"
        return niz.strip()
    
    def __eq__(self, other):
        return self.stanje == other.stanje

    def sosednja_stanja(self):
        i, j = self.prazno_mesto
        smeri = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for x, y in smeri:
            ni = i + x
            nj = j + y
            if 0 <= ni < len(self.stanje) and 0 <= nj < len(self.stanje):
                novo_stanje = [v[:] for v in self.stanje]
                novo_stanje[i][j] = self.stanje[ni][nj]
                novo_stanje[ni][nj] = None
                yield (Stanje(novo_stanje), 1)


class Uganka:
    def __init__(self, matrika):
        self.zacetno_stanje = Stanje(matrika)
    
    def zacetno(self):
        return self.zacetno_stanje
    
    def resitev(self, s):
        return s == Stanje(resitev(len(s.stanje)))
    
    def razvejaj(self, s):
        for sosed in s.sosednja_stanja():
            yield sosed

    '''Tu moramo definirati hevristiko, ki nam oceni kvaliteto stanja. '''

    '''
    Primer: Nekemu elementu seznama v ustreza
    vrstica: v // n
    stolpec: v % n
    '''
    def h(self, s):
        vrednost = 0
        n = len(s.stanje)
        r = resitev(n)
        for i in range(n):
            for j in range(n):
                if s.stanje[i][j] != r:
                    vrednost += 1
        return vrednost


'''Brez kakršne koli hevristike potrebujemo za uganko1 1039 poskusov'''

import preiskovanje

uganka1 = [[4, None, 1], [7, 2, 3], [5, 8, 6]]

'''Ta rešitev ustreza hevristiki h'''

p = preiskovanje.Preisci(Uganka(uganka1), "najprej-najboljsi")
p.preisci()
print(p)

'''Rešitev najde po 1035 poskusih'''


'''Ta rešitev ustreza sledeči hevristiki: Za vsak element pogledamo, kako daleč
je od svojega predvidenega mesta. Nato vrednosti seštejemo in poiščemo najmanjšo.'''

p = preiskovanje.Preisci(Uganka(uganka1), 'a*')
p.preisci()
print(p)

'''Vidimo, da je ta rešitev boljša.'''