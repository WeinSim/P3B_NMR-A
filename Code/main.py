import math
import random as rd
import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from expressions import *


# Source: CODATA (2022)

c = 299_792_458                     # speed of light
e = 1.602_176_634e-19               # elemtary charge
m_e = 9.109_383_7139e-31            # electron mass
m_p = 1.672_621_925_95e-27          # proton mass
h = 6.626_070_15e-34                # Planck's constant
hbar = h / (2 * math.pi)            # reduced planck's constant
u = 1.66e-27                        # atomic mass
epsilon_0 = 8.854_187_8188e-12      # vacuum permittivity
mu_0 = 1.256_637_061_27e-6          # vacuum permeability
N_A = 6.022_140_76e23               # Avogadro's constant
alpha = 7.297_352_5643e-3           # fine structure constant
k_B = 1.380649e-23                  # Blotzman constant
G = 6.67430e-11                     # gravitational constant


def tv3():
    print("--- Teilversuch 3 ---")

    f_1 = Var(17.8207e6, 0.00010e6, "f_1")
    f_2 = Var(17.8470e6, 0.00010e6, "f_2")
    x_1 = Var(-3, 1 / 20, "x_1")
    x_2 = Var(3, 1 / 20, "x_2")
    div_scale = 0.5 # 500 mV / Div

    variables = [ f_1, f_2, x_1, x_2 ]

    df_var = Sub(f_2, f_1)
    scale_var = Div(df_var, Mult(Sub(x_2, x_1), Const(div_scale)))
    scale = scale_var.eval()
    scale_unc = gaussian(scale_var, variables)
    scale_var = Var(scale, scale_unc, "scale")
    
    print_val("df", df_var.eval(), gaussian(df_var, variables))
    print_val("s", scale, scale_unc)

    delta_x = [ 1.75, 0.7, 1.0, 1.6 ]
    voltage_per_div = [ 0.5, 0.5, 0.1, 0.1 ]
    names = [ "PTFE", "Glycerin", "Polystyrol", "Wasser" ]

    for i in range(len(names)):
        dx_var = Var(delta_x[i], 1 / 20, "dx")
        fwhm_var = Mult(Mult(dx_var, Const(voltage_per_div[i])), scale_var)
        print(names[i])
        print_val("FWHM", fwhm_var.eval(), gaussian(fwhm_var, [scale_var, dx_var]))

def tv5():
    print("--- Teilversuch 5 ---")


    # frequencies in MHz
    # magnetic field strength in mT

    glycerin_freq = [ 15.8671, 15.9022, 15.9589, 16.0148, 16.0453, 16.1010, 16.1518, 16.2068, 16.2491, 16.3059, 16.3494, 16.3964, 16.4508, 16.5020, 16.5547, 16.6047, 16.6603, 16.6985, 16.7496, 16.8027, 16.8452, 16.9001, 16.9488, 17.0025, 17.0526, 17.1013, 17.1507, 17.1993, 17.2524, 17.3012, 17.3489, 17.4019, 17.4491, 17.5036, 17.5509, 17.6002, 17.6524, 17.6999, 17.7507, 17.8043, 17.8503, 17.9062, 17.9525, 18.0044, 18.0527, 18.1026, 18.1513, 18.2007, 18.2502, 18.3005, 18.3508, 18.4022, 18.4488, 18.4994, 18.5509, 18.6004, 18.6507, 18.7024, 18.7509, 18.8013, 18.8502, 18.9010, 18.9504, 19.0009 ]
    glycerin_magnet = [ 216, 217, 217, 218, 218, 219, 220, 220, 221, 222, 222, 223, 224, 225, 225, 226, 227, 227, 228, 229, 229, 230, 231, 232, 232, 233, 234, 234, 235, 236, 237, 237, 238, 238, 239, 240, 240, 241, 242, 243, 243, 244, 244, 245, 246, 246, 247, 248, 248, 249, 250, 251, 251, 252, 253, 253, 254, 255, 255, 256, 256, 257, 258, 259 ]

    wasser_freq = [ 16.6569, 17.0090, 17.3316, 17.6675, 18.0021, 18.3331, 18.6634, 18.9001, 19.0045, 19.1068 ]
    wasser_magnet = [ 227, 232, 236, 240, 245, 249, 254, 257, 258, 260 ] 
    
    ptfe_freq = [ 16.6655, 16.9995, 17.3330, 17.6647, 18.0006, 18.3352, 18.6675, 18.9013, 19.0004, 19.1383 ]
    ptfe_magnet = [ 242, 247, 251, 256, 261, 266, 271, 274, 275, 277 ]

    B_inside = 421e-3
    B_outside = 258e-3
    calibration_factor = B_inside / B_outside
    print_val("Kalibrierfaktor", calibration_factor)

    eval_TV1("Glycerin", glycerin_freq, glycerin_magnet, 5.5856912, calibration_factor)
    eval_TV1("Wasser", wasser_freq, wasser_magnet, 5.5856912, calibration_factor)
    eval_TV1("PTFE", ptfe_freq, ptfe_magnet, 5.257732, calibration_factor)

def eval_TV1(name, freq, magnet, literature, calibration_factor):
    print(name)

    x_values = [ b * 1e-3 * calibration_factor for b in magnet ]
    y_values = [ f * 1e6 for f in freq ]

    params, cov = np.polyfit(x_values, y_values, 1, cov=True)
    (m, b) = params
    dm = cov[0][0] ** 0.5
    db = cov[1][1] ** 0.5

    m_var = Var(m, dm, "m")
    g_var = Mult(m_var, Const(4 * math.pi * m_p / e))

    print_val("m", m, dm)
    print_val("b", b, db)
    print_val("g", g_var.eval(), gaussian(g_var, [m_var]), literature)

    x_range = np.linspace(min(x_values), max(x_values), 100)
    fit = np.polyval(params, x_range)

    pp = PdfPages(f"../Abbildungen/Graph_TV5_{name}.pdf")
    fig = plt.figure()

    plt.plot(x_values, y_values, "o", label="Messwerte")
    plt.plot(x_range, fit, "-", label=f"Fit (y = mx + b, m = {m:.4g}, b = {b:.4g})")

    plt.xlabel("$B$ [$T$]")
    plt.ylabel("$\\nu$ [Hz]")
    # plt.title("Zündspannung vs. Abstand * Druck")
    plt.legend()

    fig.tight_layout()
    pp.savefig()
    pp.close()

def print_val(name, value, uncertainty=None, literature=None):
    output = f"{name} = {value:12.4g}"
    if not uncertainty is None:
        output += f", Unsicherheit = {uncertainty:12.4g}"
    if not literature is None:
        difference = abs(value - literature)
        deviation_sigma = difference / uncertainty
        deviation_percent = (difference / literature) * 100
        output += f", Abweichung = {deviation_sigma:.2f} = {deviation_percent:.1f}%"
    print(output)

tv3()
tv5()
