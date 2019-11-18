import os
import numpy as np
from itertools import product
from collections import OrderedDict
from decimal import Decimal
import matplotlib.pyplot as plt
import pickle

with open('results.pck', 'r') as ff:
    limits2D = pickle.load(ff)
    
masses_central   = []
masses_one_sigma = []
masses_two_sigma = []

minus_two = []
minus_one = []
central   = []
plus_one  = []
plus_two  = []

# go through the different mass points first left to right to catch the lower exclusion bound
# then right to left to catch the upper exclusion bound
for mass in sorted(limits2D.keys()):

    if len(limits2D[mass]['exp_central'])>0: 
        central.append( min(limits2D[mass]['exp_central']) )
        masses_central.append(mass)

    if len(limits2D[mass]['exp_minus_one'])>0 and len(limits2D[mass]['exp_plus_one' ])>0: 
        minus_one.append( min(limits2D[mass]['exp_minus_one']) )
        plus_one .append( min(limits2D[mass]['exp_plus_one' ]) )
        masses_one_sigma.append(mass)

    if len(limits2D[mass]['exp_minus_two'])>0 and len(limits2D[mass]['exp_plus_two' ])>0: 
        minus_two.append( min(limits2D[mass]['exp_minus_two']) )
        plus_two .append( min(limits2D[mass]['exp_plus_two' ]) )
        masses_two_sigma.append(mass)
    
for mass in sorted(limits2D.keys(), reverse=True):

    if len(limits2D[mass]['exp_central'])>1: 
        central.append( max(limits2D[mass]['exp_central'  ]) )
        masses_central.append(mass)

    if len(limits2D[mass]['exp_minus_one'])>1 and len(limits2D[mass]['exp_plus_one' ])>1: 
        minus_one       .append( max(limits2D[mass]['exp_minus_one']) )
        plus_one        .append( max(limits2D[mass]['exp_plus_one' ]) )
        masses_one_sigma.append(mass)

    if len(limits2D[mass]['exp_minus_two'])>1 and len(limits2D[mass]['exp_plus_two' ])>1: 
        minus_two       .append( max(limits2D[mass]['exp_minus_two']) )
        plus_two        .append( max(limits2D[mass]['exp_plus_two' ]) )
        masses_two_sigma.append(mass)

# plot the 2D limits
plt.clf()

plt.fill_between(masses_two_sigma, minus_two, plus_two, color='gold'       , label=r'$\pm 2 \sigma$')
plt.fill_between(masses_one_sigma, minus_one, plus_one, color='forestgreen', label=r'$\pm 1 \sigma$')
plt.plot        (masses_central  , central            , color='red'        , label='central expected', linewidth=2)

plt.ylabel('$|V|^2_\mu$')
plt.xlabel('mass (GeV)')
plt.legend()
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
#plt.tight_layout()
plt.yscale('log')
plt.xscale('linear')
plt.grid(True)
plt.savefig('2d_hnl_limit.pdf')





    

