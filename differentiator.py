import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy import sqrt
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit                        # Funktionsfit:     popt, pcov = curve_fit(func, xdata, ydata) 
from uncertainties import ufloat                            # Fehler:           fehlerwert =  ulfaot(x, err)
from uncertainties import unumpy as unp 
from uncertainties.unumpy import uarray                     # Array von Fehler: fehlerarray =  uarray(array, errarray)
from uncertainties.unumpy import (nominal_values as noms,   # Wert:             noms(fehlerwert) = x
                                  std_devs as stds)         # Abweichung:       stds(fehlerarray) = errarray

# Plot 1:

f, Ue, Ua = np.genfromtxt('differentiator.txt', unpack=True, skip_header=1) 

# für den initial guess bei curvefit()
#n = len(f)                             # Anzahl der Daten
#mean = sum(r*V)/n                      # Mittelwert
#sigma = np.sqrt(sum(V*(r - mean)**2))   # Standardabweichung

# Ausgleichsrechung nach Gaußverteilung

def g(m,x,b):
    return m*x+b  # b = 2*sigma**2

para, pcov = curve_fit(g, np.log(f),np.log(Ua/Ue))
m, b = para
pcov = np.sqrt(np.diag(pcov))
fm, fb = pcov
um = ufloat(m, fm) 
ub = ufloat(b, fb)

print('um:',um,'ub:',ub)

xx = np.linspace(1.5, 6, 10**4)

plt.plot(np.log(f), np.log(Ua/Ue), 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
plt.plot(xx, g(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$f \, / \, \mathrm{Hz}$')
plt.ylabel(r'$U_A \, / \, U_E$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style

plt.savefig('build/differentiator.pdf', bbox_inches = "tight")
plt.clf() 
