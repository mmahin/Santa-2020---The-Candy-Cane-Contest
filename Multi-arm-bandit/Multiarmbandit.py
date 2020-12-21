from arm import  Arm
'''
#widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display

#plots
import matplotlib.pyplot as plt
from plotnine import *

#stats
import scipy as sp
import statsmodels as sm
'''
a = Arm(0.1)
for i in range(100): print(a.pull())
print(a.get_state())