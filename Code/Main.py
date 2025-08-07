import math
import random as rd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from Expressions import *

# markers - linestyle - color

def tv1():
    print("--- Teilversuch 1 ---")

    # transmittiert
    # mm
    # r = [0.0, 5.5, 7.5, 8.5, 10.0, 10.5, 11.0, 12.0, 12.5, 13.0, 13.5]
    r = [4.5, 5.5, 7.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.5, 12.0]
    evalTV1("Transmittiert", r)

    # transmittiert
    # mm
    r = [5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0]
    evalTV1("Reflektiert", r)

def evalTV1(name, r):
    n = np.arange(1, len(r) + 1)
    r2 = []
    for ri in r:
        r2.append(ri * ri)
    coefs = np.polyfit(n, r2, 1)

    # deltaCoefs = linRegUncertainty(n, r2, coefs)
    deltaCoefs = [0.1, 0.1]

    varM = Var(coefs[0], deltaCoefs[0], "m")
    varT = Var(coefs[1], deltaCoefs[1], "t")

    params = [varM]

    cR = Const(12.141e3)
    cL = Const(635e-6) 

    r = Div(varM, cL)
    l = Div(varM, cR)

    print(f"{name}:")

    printVar(varM)
    printVar(varT)

    rVal = r.eval()
    rUnc = gaussian(r, params)
    printExpr("R", rVal * 1e-3, rUnc * 1e-3)
    printDiff(rVal, cR.eval(), rUnc)

    lVal = l.eval()
    lUnc = gaussian(l, params)
    printExpr("lambda", lVal * 1e3, lUnc * 1e3)
    printDiff(lVal, cL.eval(), lUnc)

    print()

    fit = np.polyval(coefs, n)

    pp = PdfPages(f"../Abbildungen/GraphTV1_{name}.pdf")

    plt.figure()
    plt.clf()

    # plt.plot(d, p, "-o")
    plt.plot(n, r2, "o", label=f"Messwerte")
    plt.plot(n, fit, "-", label=f"Ausgleichsgerade")

    plt.title(f"Radius^2 vs. Interfernzordnung")
    plt.xlabel('Interferenzordnung')
    plt.ylabel('Radius^2 (m^2)')
    plt.legend()

    pp.savefig()
    pp.close()

def printExpr(name, value, unc):
    print(f"{name}  = {value}")
    print(f"∆{name} = {unc}")

def printVar(var):
    printExpr(var.name, var.value, var.uncertainty)

def printDiff(val, theo, unc):
    u = abs(val - theo) / unc
    print(f"Abweichung vom theoretischen Wert: {u} * Unsicherheit")

def tv2():
    print("--- Teilversuch 2 ---")

    print("Fresnelbiprisma 1:")
    s = Var(385e-3, 3e-3, "s")
    bB = Var(18e-3, 1e-3, "B")
    b = Var(2180e-3, 5e-3, "b")
    f = Const(300e-3)
    dm = Var(30e-3, 1e-3, "delta_m")
    evalSpiegel(s, bB, b, f, dm, 54)

    print("Fresnelbiprisma 2:")
    s = Var(385e-3, 3e-3, "s")
    bB = Var(12e-3, 1e-3, "B")
    b = Var(2115e-3, 5e-3, "b")
    f = Const(300e-3)
    dm = Var(20e-3, 1e-3, "delta_m")
    evalSpiegel(s, bB, b, f, dm, 25)

    print("Fresnelbiprisma 3:")
    s = Var(385e-3, 3e-3, "s")
    bB = Var(9e-3, 1e-3, "B")
    b = Var(2115e-3, 5e-3, "b")
    f = Const(300e-3)
    dm = Var(15e-3, 1e-3, "delta_m")
    evalSpiegel(s, bB, b, f, dm, 13)

    print("Fresnelbiprisma:")
    s = Var(380e-3, 3e-3, "s")
    bB = Var(21e-3, 1e-3, "B")
    b = Var(2510e-3, 5e-3, "b")
    f = Const(300e-3)
    dm = Var(25e-3, 1e-3, "delta_m")
    evalSpiegel(s, bB, b, f, dm, 35)

def evalSpiegel(s, bB, b, f, dm, m):
    params = [s, bB, b, f, dm]

    a = Div(Mult(bB, f), Sub(b, f))
    printExpr("a", a.eval() * 1e3, gaussian(a, params) * 1e3)

    delta = Div(dm, Const(m))

    lam = Div(Mult(a, delta), Add(s, b))
    lamUnc = gaussian(lam, params)
    lamVal = lam.eval()
    printExpr("lambda", lamVal * 1e6, lamUnc * 1e6)

    theo = 635e-9

    unc = abs(lamVal - theo) / lamUnc
    print("Abweichung vom theoretischen Wert: %.3f * Unsicherheit" % (unc))
    print()

tv1()
tv2()
