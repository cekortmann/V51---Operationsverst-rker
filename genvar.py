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
t1,U1= np.genfromtxt('genvarAmpFreestyle.txt', unpack=True, skip_header=1)
# für den initial guess bei curvefit()
#n = len(f)                             # Anzahl der Daten
#mean = sum(r*V)/n                      # Mittelwert
#sigma = np.sqrt(sum(V*(r - mean)**2))   # Standardabweichung

# Ausgleichsrechung nach Gaußverteilung

def g(x,U0,T,a):
    return U0*np.exp(-x/T)+a  # b = 2*sigma**2

#p0= (1.,1.,1./100,75.)
para, pcov = curve_fit(g,t,U,[1,40,1.6])
U0, T, a= para
pcov = np.sqrt(np.diag(pcov))
fU, fT, fa= pcov
uU = ufloat(U0, fU) 
uT = ufloat(T, fT)
ua = ufloat(a, fa)

print('uU', uU)
print('uT', uT)
print('ua', ua)

xx = np.linspace(0, 800, 10**4)

plt.plot(t, U, 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
plt.plot(xx, g(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
#plt.plot(xx, 1.70*np.exp(-xx/(3.75*10**(2))), '-y', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$t \, / \, \mathrm{ns}$')
plt.ylabel(r'$U_A \, / \, \mathrm{V}$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
#plt.ylim(0, 1.4)

plt.savefig('build/genvar.pdf', bbox_inches = "tight")
plt.clf() 
