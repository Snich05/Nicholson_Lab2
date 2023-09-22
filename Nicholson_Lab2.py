#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:45:13 2023

@author: shanycenicholson
"""

import numpy as np
from matplotlib import pyplot as plt
import time

x1 = 2
nx = 750
x = np.linspace(0,x1,nx)
dx = x1/(nx-1)
nt = 200
dt = 0.0025
c = 1
g = .01
theta = x/(0.5*x1)
cfl = round(c*dt/dx,2)

if cfl >= 1:
    print('CFL is %s, which is over 1'%(cfl))
else:
    print('CFL = %s'%(cfl))
    
u = np.ones(nx)
un = np.ones(nx)
u = (1/(2*np.sqrt(np.pi*(g))))*np.exp(-(1-theta)**2/(4*g))
ui = u.copy()
plt.plot(x,u);

start = time.process_time()
for n in range(nt):
    un = u.copy()
    for i in range(1,nx-1):
        u[i] = un[i] - c*dt/(dx)*(un[i]-un[i-1])
        u[0] = u[nx-2]
        u[nx-1] = u[1]
        
end = time.process_time()
print(end-start)


import numpy as np
from matplotlib import pyplot as plt

x1 = 2
nx = 100
x = np.linspace(0,2,nx)
dx = x1/(nx-1)
nt = 500
nu = 0.01
sigma = 0.5
dt = sigma*dx**2/nu

u = np.zeros(nx)
un = np.zeros(nx)
u[int(.8/dx):int(1.2/dx+1)]=1
plt.plot(x,u)
plotcond = np.zeros(nt)
p = 1
f = 40

for n in range(nt):
    un = u.copy()
    u[0]=0
    u[nx-1]=0
    u[1:-1] = un[1:-1] + nu*dt/dx**2*(un[2:]-2*un[1:-1]+un[:-2])
    plotcond[n] = np.round((n/f/p),1)
    plotcond[n] = plotcond[n] = plotcond[n].astype(int)
    if plotcond[n]==1:
        plt.figure(1)
        plt.plot(x,u,label='t=%s s'%(np.round(n*dt,1)))
        plt.legend()
        p += 1
