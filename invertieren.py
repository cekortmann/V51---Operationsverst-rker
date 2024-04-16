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



f ,V , U, g ,r =np.genfromtxt('Invertieren1.txt',unpack= True, skip_header=1)
#print(U)
#plt.plot(np.log(r),np.log(V), 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
#plt.ylim(0, 1.5)
#plt.xlim(0.00, 1.3)
#plt.savefig('build/Invertieren1.pdf', bbox_inches = "tight")
f2 ,V2 , U2, g2 ,r2 =np.genfromtxt('Invertieren1.txt',unpack= True, skip_header=1, skip_footer=7)
    
# für den initial guess bei curvefit()
n = len(r)                             # Anzahl der Daten
mean = sum(r*V)/n                      # Mittelwert
sigma = np.sqrt(sum(V*(r - mean)**2))  # Standardabweichung

def log(u):
    return np.log(u)

def g(m,x,b):
    return m*x+b  # b = 2*sigma**2

para, pcov = curve_fit(g, log(r2),log (V2/U2))
m, b = para
pcov = np.sqrt(np.diag(pcov))
fm, fb = pcov
um = ufloat(m, fm) 
ub = ufloat(b, fb)

print('um:',um,'ub:',ub)

xx = np.linspace(0,1.3 , 10**4)

plt.plot(log(r),log(V/U), 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
plt.plot(xx, g(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$log(\Delta \phi)}$')
plt.ylabel(r'$^log(U_0 \, / \, U_1)$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
#plt.xlim(22, 40)
#plt.ylim(-0.05, 1.05)

plt.savefig('build/Invertieren1.pdf', bbox_inches = "tight")
plt.clf() 

f3 ,V3 , U3, r3 ,g3 =np.genfromtxt('Invertieren2.txt',unpack= True, skip_header=1)

f4 ,V4 , U4, r4 ,g4 =np.genfromtxt('Invertieren2.txt',unpack= True, skip_header=1, skip_footer=7)
    
# für den initial guess bei curvefit()
n = len(r)                             # Anzahl der Daten
mean = sum(r*V)/n                      # Mittelwert
sigma = np.sqrt(sum(V*(r - mean)**2))  # Standardabweichung


para, pcov = curve_fit(g, log(r3),log (V3/U3))
m, b = para
pcov = np.sqrt(np.diag(pcov))
fm, fb = pcov
um = ufloat(m, fm) 
ub = ufloat(b, fb)

print('um:',um,'ub:',ub)


plt.plot(log(r3),log(V3/U3), 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
plt.plot(xx, g(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$log(\Delta \phi)}$')
plt.ylabel(r'$^log(U_0 \, / \, U_1)$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
#plt.xlim(22, 40)
#plt.ylim(-0.05, 1.05)
plt.savefig('build/Invertieren2.pdf', bbox_inches = "tight")
plt.clf() 

f5 ,V5 , U5, r5 ,g5 =np.genfromtxt('Invertieren3.txt',unpack= True, skip_header=1)

f6 ,V6 , U6, r6 ,g6=np.genfromtxt('Invertieren3.txt',unpack= True, skip_header=1, skip_footer=7)
    
# für den initial guess bei curvefit()
n3 = len(r5)                             # Anzahl der Daten
mean3 = sum(r5*V5)/n3                      # Mittelwert
sigma3 = np.sqrt(sum(V5*(r5 - mean3)**2))  # Standardabweichung

para, pcov = curve_fit(g, log(r5),log (V5/U5))
m, b = para
pcov = np.sqrt(np.diag(pcov))
fm, fb = pcov
um = ufloat(m, fm) 
ub = ufloat(b, fb)

print('um:',um,'ub:',ub)

plt.plot(log(r3),log(V3/U3), 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
plt.plot(xx, g(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$log(\Delta \phi)}$')
plt.ylabel(r'$^log(U_0 \, / \, U_1)$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
#plt.xlim(22, 40)
#plt.ylim(-0.05, 1.05)
plt.savefig('build/Invertieren3.pdf', bbox_inches = "tight")
plt.clf() 