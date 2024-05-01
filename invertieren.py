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
f2 ,V2 , U2, g2 ,r2 =np.genfromtxt('Invertieren1.txt',unpack= True, skip_header=9, skip_footer=1)
    
# für den initial guess bei curvefit()
n = len(r)                             # Anzahl der Daten
mean = sum(r*V)/n                      # Mittelwert
sigma = np.sqrt(sum(V*(r - mean)**2))  # Standardabweichung

def log(u):
    return np.log(u)

def g(m,x,b):
    return m*x+b  # b = 2*sigma**2


#def f(m,b):
#    return np.exp(1/m*(log(92.9/np.sqrt(2))-b))

para, pcov = curve_fit(g, log(f2),log (V2/U2))
m, b = para
pcov = np.sqrt(np.diag(pcov))
fm, fb = pcov
um = ufloat(m, fm) 
ub = ufloat(b, fb)

print('um:',um,'ub:',ub)
Plateau1= log(19.5/0.21)
print('Grenzfrequenz1', (Plateau1/np.sqrt(2)-ub)/um)
#print(np.exp(logfg1))
xx = np.linspace(8.5,12 , 10**4)

plt.plot(log(f),log(V/U), 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
plt.plot(xx, g(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$log(f/{Hz})}$')
plt.ylabel(r'$log(U_0 \, / \, U_{i})$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
#plt.xlim(22, 40)
#plt.ylim(-0.05, 1.05)

plt.savefig('build/Invertieren1.pdf', bbox_inches = "tight")
plt.clf() 

f3 ,V3 , U3, r3 ,g3 =np.genfromtxt('Invertieren2.txt',unpack= True, skip_header=1)

f4 ,V4 , U4, r4 ,g4 =np.genfromtxt('Invertieren2.txt',unpack= True, skip_header=11, skip_footer=1)
    
# für den initial guess bei curvefit()
n = len(r)                             # Anzahl der Daten
mean = sum(r*V)/n                      # Mittelwert
sigma = np.sqrt(sum(V*(r - mean)**2))  # Standardabweichung


para, pcov = curve_fit(g, log(f4),log (V4/U4))
m, b = para
pcov = np.sqrt(np.diag(pcov))
fm, fb = pcov
um = ufloat(m, fm) 
ub = ufloat(b, fb)

print('um:',um,'ub:',ub)
Plateau2= log(27/0.32)
print(Plateau2)
print('Grenzfrequenz2', (Plateau2/np.sqrt(2)-ub)/um)
print('Grenzfrequenz2',np.exp(1/m*(log(61.4/np.sqrt(2))-b)))

plt.plot(log(f3),log(V3/U3), 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
plt.plot(xx, g(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$log(f/{Hz})}$')
plt.ylabel(r'$log(U_0 \, / \, U_{i})$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
#plt.xlim(22, 40)
#plt.ylim(-0.05, 1.05)
plt.savefig('build/Invertieren2.pdf', bbox_inches = "tight")
plt.clf() 

f5 ,V5 , U5, r5 ,g5 =np.genfromtxt('Invertieren3.txt',unpack= True, skip_header=1)

f6 ,V6 , U6, r6 ,g6=np.genfromtxt('Invertieren3.txt',unpack= True, skip_header=10, skip_footer=1)
    
# für den initial guess bei curvefit()
n3 = len(r5)                             # Anzahl der Daten
mean3 = sum(r5*V5)/n3                      # Mittelwert
sigma3 = np.sqrt(sum(V5*(r5 - mean3)**2))  # Standardabweichung

para, pcov = curve_fit(g, log(f6),log (V6/U6))
m, b = para
pcov = np.sqrt(np.diag(pcov))
fm, fb = pcov
um = ufloat(m, fm) 
ub = ufloat(b, fb)

print('um:',um,'ub:',ub)

plt.plot(log(f5),log(V5/U5), 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
plt.plot(xx, g(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$log(f/{Hz})}$')
plt.ylabel(r'$log(U_0 \, / \, U_{i})$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
#plt.xlim(22, 40)
#plt.ylim(-0.05, 1.05)
plt.savefig('build/Invertieren3.pdf', bbox_inches = "tight")
plt.clf() 

Plateau3= log(27/0.32)
print('Grenzfrequenz3', (Plateau3/np.sqrt(2)-ub)/um)
print('Grenzfrequenz3',np.exp(1/m*(log(84.4/np.sqrt(2))-b)))

uf= ufloat(16317.6,157)
print(84.4*uf)