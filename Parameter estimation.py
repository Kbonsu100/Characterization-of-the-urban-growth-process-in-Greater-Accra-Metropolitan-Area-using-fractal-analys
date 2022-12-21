# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:57:05 2022

@author: kbons
"""

#parameter estimation of logistic growth model

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.metrics import r2_score
from scipy.integrate import odeint


data = pd.read_csv('C:\\Users\\kbons\\Desktop\\GAMA Land Cover Change\\new project\\GAMA_Classified\\Maps\\Fractal_Dim.csv', ';')

year= data['Year'].values.reshape(-1, 1)
x1= data['Year']
dimension = data['Fractal Dimension'].values.reshape(-1,1)
y1 = data['Fractal Dimension']



def sim(variables, t, params):
    X = variables[0]
   
    r = params[0]
    K = params[1]
    
    dXdt = r*K * (1-X/K)
    
    return([dXdt])


def loss_function(params, x1, y1):
    y0 = [y1[0]]
    t = np.linspace(x1[0],x1[3], num=len(x1))
    
    output = odeint(sim, y0, t, args=(params, ))
    
    loss=0
    
    for i in range(len(x1)):
        data_count = y1[i]
        model_count = output[i, 0]
    
        res =(data_count - model_count)**2
        loss += res
        #print("loss is", loss)
    return (loss) 
    
   

params0 = np.array([0.001,2])
minimum = scipy.optimize.fmin(loss_function, params0, args=(x1, y1))
print(minimum)


r_fit = minimum[0]
L_fit = minimum[1]

params = [r_fit, L_fit]
y0 = [y1[0]]
t = np.linspace(x1[0],2030)

output= odeint(sim, y0, t, args = (params,),)

# import scipy as sp

# r, p = sp.stats.pearsonr(y1, output)

# odeint





sns.set_theme()
sns.set(font_scale = 1.5)
f, (ax1)= plt.subplots(1)
line1 = ax1.scatter(x1, y1, c='b', s=20)
line1, = ax1.plot(t, output[:,0],color='r')
ax1.set_ylabel('Fractal Dimension')
ax1.set_xlabel('Year')
# ax1.text(0.05, .8, 'r={:.2f}, p ={:.2g}'.format(r,p),
#          transform = ax1.transAxes)
plt.show()
   

# for c in range(len(x1)):
#     observed = y1[c]
#     predicted = output[c, 0]

#     r2 = r2_score(observed,predicted) 


data_1 = pd.read_csv('C:\\Users\\kbons\\Desktop\\GAMA Land Cover Change\\new project\\GAMA_Classified\\Maps\\r2.csv', ';')
obs = data_1['observed']
prd = data_1['predicted']
r2 = r2_score(y1,) 































    