class Preisci:
    """Razred isci, ki implementira sploĹĄni preiskovalni algoritem
    in ĹĄtiri strategije iskanja:
    (1) iskanje v globino, si = 'v-globino',
    (2) iskanje v ĹĄirino, si = 'v-ĹĄirino',
    (3) najprej najboljĹĄi, si = 'najprej-najboljsi',
    (4) A*, si = 'a*'."""

    def __init__(self, pr, si):
        """Naredimo novo instanco preiskovalnega algoritma za
        prostor moĹžnih reĹĄitev pr in strategijo iskanja si.
        Predpostavka je, da je pr objekt razreda, ki definira
        naslednje ĹĄtiri metode:
        (1) zacetno(): vrne zaÄetno stanje v prostoru moĹžnih reĹĄitev;
        (2) resitev(s): preveri ali je stanje s reĹĄitev;
        (3) razvejaj(s): vrne iterator skozi naslednike stanja s;
        (4) h(s): vrne vrednost hevristiÄne funkcije za stanje s."""

        self.pr = pr
        self.si = si

        # Odprti seznam dvojic
        # (1) stanje s, ki ĹĄe ni razvejano, in
        # (2) cena g dosedanje poti do stanja s
        self.os = [(self.pr.zacetno(), 0)]

        # Zaprti seznam Ĺže razvejanih stanj
        self.zs = []

        # Ĺ tevec razvejanj vozliĹĄÄ
        self.n_razvejana = 0


    def preisci(self):
        """SploĹĄni preiskovalni algoritem."""

        # NeskonÄna zanka preiskovanja
        while True:
            #print(self)

            # Vzamemo prvi element iz odprtega seznama os (s je stanje, g je cena poti do s)
            (s, g) = self.os[0]

            # Äe je s reĹĄitev vrnemo stanje, ceno poti in ĹĄtevilo razvejanih vozliĹĄÄ
            if self.pr.resitev(s):
                return (s, g, self.n_razvejana)
            #else

            # pripravimo se za razvejanje stanja s
            # poveÄamo ĹĄtevilo razvejanj in prestavimo s iz odprtega na zaprti seznam
            self.n_razvejana += 1
            self.os = self.os[1:]
            if s not in self.zs: self.zs.append(s)

            # razvejamo stanje s
            for (ns, cena) in self.pr.razvejaj(s):
                # Äe je naslednje stanje ns na zaprtem seznamu, ga preskoÄimo
                if ns in self.zs: continue
                # else

                # Äe gre za iskanje v ĹĄirino,
                # dodaj novo stanje na konec odprtega seznama
                # sicer pa ga dodaj na zaÄetek
                if self.si == "v-sirino":
                    self.os = self.os + [(ns, g + cena)]
                else:
                    self.os = [(ns, g + cena)] + self.os

            # Äe gre za strategijo najprej najboljĹĄi,
            # uredi odprti seznam po naraĹĄÄajoÄi vrednosti g
            if self.si == "najprej-najboljsi":
                self.os = sorted(self.os, key = lambda sg: sg[1])

            # Äe gre za strategijo a*,
            # uredi odprti seznam po naraĹĄÄajoÄi vrednosti g + h
            if self.si == "a*":
                self.os = sorted(self.os, key = lambda sg: sg[1] + self.pr.h(sg[0]))


    def __str__(self):
        """Vrni berljivo predstavitev stanja preiskovalnega algoritma."""

        return f"Zaprti seznam {self.zs}\nOdprti seznam {self.os}\n{self.n_razvejana} razvejanih vozlišč"