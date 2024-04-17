import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy import sqrt
import pandas as pd
import math
import scipy.constants as const
from scipy.optimize import curve_fit                        # Funktionsfit:     popt, pcov = curve_fit(func, xdata, ydata) 
from uncertainties import ufloat                            # Fehler:           fehlerwert =  ulfaot(x, err)
from uncertainties import unumpy as unp 
from uncertainties.unumpy import uarray                     # Array von Fehler: fehlerarray =  uarray(array, errarray)
from uncertainties.unumpy import (nominal_values as noms,   # Wert:             noms(fehlerwert) = x
                                  std_devs as stds)         # Abweichung:       stds(fehlerarray) = errarray

# Plot 1:

t, U = np.genfromtxt('genvarAmp.txt', unpack=True, skip_header=1) 

# für den initial guess bei curvefit()
#n = len(f)                             # Anzahl der Daten
#mean = sum(r*V)/n                      # Mittelwert
#sigma = np.sqrt(sum(V*(r - mean)**2))   # Standardabweichung

# Ausgleichsrechung nach Gaußverteilung

<<<<<<< HEAD
def f(U0,n, x):
    return U0*np.exp(n*x/(20*10000*22))  # b = 2*sigma**2
||||||| 3984183
def g(U0,n, x):
    return U0*np.exp(n*x/(20*10000*22))  # b = 2*sigma**2
=======
def g(U0,T, x):
    return U0*np.exp(x*10**(-9)/T)  # b = 2*sigma**2
>>>>>>> f4770250402e053435f95e10ddf7e899dc48b11a

<<<<<<< HEAD
para, pcov = curve_fit(f,t,U)
n, U0 = para
||||||| 3984183
para, pcov = curve_fit(g,t,U)
U0, n = para
=======
para, pcov = curve_fit(g,t,U)
U0, T = para
>>>>>>> f4770250402e053435f95e10ddf7e899dc48b11a
pcov = np.sqrt(np.diag(pcov))
<<<<<<< HEAD
fn, fU = pcov
||||||| 3984183
fU, fn = pcov
=======
fU, fT = pcov
>>>>>>> f4770250402e053435f95e10ddf7e899dc48b11a
uU = ufloat(U0, fU) 
uT = ufloat(T, fT)

print('uU:',uU,'un:',uT)

xx = np.linspace(0, 800, 10**4)

plt.plot(t, U, 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
plt.plot(xx, f(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$t \, / \, \mathrm{ns}$')
plt.ylabel(r'$U_A \, / \, \mathrm{V}$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style

plt.savefig('build/genvar.pdf', bbox_inches = "tight")
plt.clf() 
