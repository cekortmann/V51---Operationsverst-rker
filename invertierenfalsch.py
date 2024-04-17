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
f1 ,V1 , U1, g1 ,r1 =np.genfromtxt('Invertieren1.txt',unpack= True, skip_header=11, skip_footer=2)
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

def g(x,b,m):
    return m*x**b  # b = 2*sigma**2

para, pcov = curve_fit(g, (f1), (V1/U1))
m, b = para
pcov = np.sqrt(np.diag(pcov))
fm, fb = pcov
um = ufloat(m, fm) 
ub = ufloat(b, fb)

print('um:',um,'ub:',ub)

xx = np.linspace(7,10 , 10**4)

plt.plot((f),(V/U), 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
plt.plot(xx, g(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$f/{Hz}}$')
plt.ylabel(r'$U_0 \, / \, U_i$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
#plt.xlim(22, 40)
#plt.ylim(-0.05, 1.05)

plt.savefig('build/Invertieren1.pdf', bbox_inches = "tight")
plt.clf() 

f3 ,V3 , U3, r3 ,g3 =np.genfromtxt('Invertieren2.txt',unpack= True, skip_header=1)
f35 ,V35 , U35, r35 ,g35 =np.genfromtxt('Invertieren2.txt',unpack= True, skip_header=9, skip_footer=1)
f4 ,V4 , U4, r4 ,g4 =np.genfromtxt('Invertieren2.txt',unpack= True, skip_header=1, skip_footer=8)
    


para, pcov = curve_fit(g,log(f35),log(V35/U35))
m, b = para
pcov = np.sqrt(np.diag(pcov))
fm, fb = pcov
um = ufloat(m, fm) 
ub = ufloat(b, fb)

print('um:',um,'ub:',ub)
xx2 = np.linspace(6000,140000 , 10**4)

plt.plot((f3),(V3/U3), 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
plt.plot(xx2, g(xx2, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$(f/{Hz})}$')
plt.ylabel(r'$U_0 \, / \, U_i$')
plt.legend(loc="best")                  # legend position
plt.grid(True)
plt.xscale('log')
plt.yscale('log')                         # grid style
#plt.xlim(22, 40)
#plt.ylim(-0.05, 1.05)
plt.savefig('build/Invertieren2.pdf', bbox_inches = "tight")
plt.clf() 

f5 ,V5 , U5, r5 ,g5 =np.genfromtxt('Invertieren3.txt',unpack= True, skip_header=1)
f55 ,V55 , U55, r55 ,g55=np.genfromtxt('Invertieren3.txt',unpack= True, skip_header=10, skip_footer=0)
f6 ,V6 , U6, r6 ,g6=np.genfromtxt('Invertieren3.txt',unpack= True, skip_header=1, skip_footer=7)
    

para, pcov = curve_fit(g, log(f55), log(V55/U55))
m, b = para
pcov = np.sqrt(np.diag(pcov))
fm, fb = pcov
um = ufloat(m, fm) 
ub = ufloat(b, fb)

print('um:',um,'ub:',ub)

plt.plot(log(f5),log(V5/U5), 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
plt.plot(xx, g(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$f/{Hz}}$')
plt.ylabel(r'$U_0 \, / \, U_i$')
plt.legend(loc="best")                  # legend position
plt.grid(True)   
#plt.xscale('log')
#plt.yscale('log')                      
#plt.xlim(22, 40)
#plt.ylim(-0.05, 1.05)
plt.savefig('build/Invertieren3.pdf', bbox_inches = "tight")
plt.clf() 