# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 22:23:48 2022

@author: kbons
"""

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 

sns.set(font_scale = 1.5)

data = pd.read_csv('C:\\Users\\kbons\\Desktop\\GAMA Land Cover Change\\new project\\GAMA_Classified\\Maps\\Box_2013.csv', ';')
fig, ax = plt.subplots(1, figsize = (15,7))
sns.set_theme()
#sns.set_style("ticks")
sns.despine()
sns.pointplot(data, x="S", y ="Scaling Behavior")
sns.lineplot(data, x="S", y ="Scaling Behavior")
# sns.pairplot(data, hue = "S")


#land cover 
data = pd.read_csv('C:\\Users\\kbons\\Desktop\\GAMA Land Cover Change\\new project\\GAMA_Classified\\Maps\\land cover.csv', ';')
fig, ax = plt.subplots(1, figsize = (15,7))
#sns.set_theme()
sns.displot(data, x="Built-up")
sns.pairplot(data)

# Evolution of fractal dimension
data = pd.read_csv('C:\\Users\\kbons\\Desktop\\GAMA Land Cover Change\\new project\\GAMA_Classified\\Maps\\Fractal_Dim.csv', ';')
fig, ax = plt.subplots(1, figsize = (15,7))
sns.set_theme()
#sns.set_style("ticks")
#sns.despine()
sns.pointplot(data, x="Year", y ="Fractal Dimension")
sns.regplot(data, x='Year', y='Fractal Dimension')

#sns.pairplot(data)

#Scaling Behavior all years
data = pd.read_csv('C:\\Users\\kbons\\Desktop\\GAMA Land Cover Change\\new project\\GAMA_Classified\\Maps\\Scaling Behavior.csv', ';')
data = data.melt('S', var_name = 'Year', value_name= 'Scaling Behavior')
fig, ax = plt.subplots(1, figsize = (15,7))
sns.set_theme()
sns.despine()
sns.pointplot(data, x="S", y ="Scaling Behavior", hue = "Year")
sns.lineplot(data, x="S", y ="Scaling Behavior")








