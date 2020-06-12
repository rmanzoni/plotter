# https://sukhbinder.wordpress.com/2017/06/13/intersection-of-two-curves-in-pure-numpy/
# handle python subprocess, it might need to be updated when switching to python3
# https://stackoverflow.com/questions/4760215/running-shell-command-and-capturing-the-output

import os
import re
import numpy as np
import subprocess
from glob import glob
from itertools import product
from collections import OrderedDict
from decimal import Decimal
import matplotlib.pyplot as plt
from intersection import intersection
import pickle

all_datacards = glob('datacards/datacard*hnl*m_12*.txt')
all_datacards = [dd for dd in all_datacards if 'coarse' not in dd]
all_datacards.sort()
#print(all_datacards)


signal_type = 'majorana'
method = 'asymptotic' # 'toys'
variable = 'hnl_m_12'
# categories_to_combine = OrderedDict(zip(['lxy_lt_0p5', 'lxy_0p5_to_2p0', 'lxy_mt_2p0'], ['disp1', 'disp2', 'disp3']))
categories_to_combine = OrderedDict(zip(['lxy_lt_0p5', 'lxy_0p5_to_1p5', 'lxy_1p5_to_4p0', 'lxy_mt_4p0'], ['disp1', 'disp2', 'disp3', 'disp4']))
run_blind = False
flavour = r'$|V|^2_{e}$'

# nested dictionary with mass and coupling as keys
digested_datacards = OrderedDict()

# store results for 2D limits
limits2D = OrderedDict()

for idc in all_datacards:
    if signal_type not in idc:
        continue

    # string mangling
    name = idc.split('/')[1].split('.')[0]
    signal_name = re.findall(r'hnl_m_\d+_v2_\d+p\d+Em\d+', name)[0]
    signal_mass = float(re.findall(r'\d+', re.findall(r'hnl_m_\d+_', signal_name)[0])[0])
    signal_coupling_raw = re.findall(r'\d+', re.findall(r'_\d+p\d+Em\d+', signal_name)[0])
    signal_coupling = float('%s.%se-%s' %(signal_coupling_raw[0], signal_coupling_raw[1], signal_coupling_raw[2]))

    if signal_mass not in digested_datacards.keys():
        digested_datacards[signal_mass] = OrderedDict()
    
    if signal_coupling not in digested_datacards[signal_mass].keys():
        digested_datacards[signal_mass][signal_coupling] = []
    
    if any([v in idc for v in categories_to_combine.keys()]):
        digested_datacards[signal_mass][signal_coupling].append(idc) 
    
for mass, couplings in digested_datacards.iteritems():
    
    if mass in [1,10,11,12,20]: continue
    
    #print('mass =', mass)
    
    v2s       = []
    obs       = []
    minus_two = []
    minus_one = []
    central   = []
    plus_one  = []
    plus_two  = []
    
    for coupling in couplings.keys():
        #print('\tcoupling =', coupling)
        datacards_to_combine = couplings[coupling]
        # gonna combine the cards    
        command = 'combineCards.py'
        for cat, idc in product(categories_to_combine, datacards_to_combine):
            if cat in idc:
                command += ' %s=%s ' %(categories_to_combine[cat],idc)
        command += ' > datacard_combined_tmp.txt'
        
        #print('\t\t',command)
        os.system(command)
        
        command = 'combine -M AsymptoticLimits datacard_combined_tmp.txt'
        if run_blind:
            command += ' --run blind'
        
        #print('\t\t',command)
        results = subprocess.check_output(command.split())
        
        result_file_name = ('result_m_%d_v2_%.1E.txt' %(mass, Decimal(coupling))).replace('-', 'm')
        with open(result_file_name, 'w') as ff:
            print >> ff, results

        new_obs       = None
        new_minus_two = None
        new_minus_one = None
        new_central   = None
        new_plus_one  = None
        new_plus_two  = None

        lines = results.split('\n')
        for line in lines:
            if 'Observed'        in line: new_obs       = float(re.findall(r'\d+\.\d+', line.split(':')[-1])[-1])
            if 'Expected  2.5%:' in line: new_minus_two = float(re.findall(r'\d+\.\d+', line.split(':')[-1])[-1])
            if 'Expected 16.0%:' in line: new_minus_one = float(re.findall(r'\d+\.\d+', line.split(':')[-1])[-1])
            if 'Expected 50.0%:' in line: new_central   = float(re.findall(r'\d+\.\d+', line.split(':')[-1])[-1])
            if 'Expected 84.0%:' in line: new_plus_one  = float(re.findall(r'\d+\.\d+', line.split(':')[-1])[-1])
            if 'Expected 97.5%:' in line: new_plus_two  = float(re.findall(r'\d+\.\d+', line.split(':')[-1])[-1])

        if any([jj is None for jj in [new_minus_two, new_minus_one, new_central, new_plus_one, new_plus_two] ]):
            continue
        if not run_blind:
            if new_obs is None:
                continue    
        
        v2s      .append(coupling     )
        obs      .append(new_obs      )
        minus_two.append(new_minus_two)
        minus_one.append(new_minus_one)
        central  .append(new_central  )
        plus_one .append(new_plus_one )
        plus_two .append(new_plus_two )
        
    
    if not run_blind:
        graph = zip(v2s, minus_two, minus_one, central, plus_one, plus_two, obs)
    else:
        graph = zip(v2s, minus_two, minus_one, central, plus_one, plus_two)    
    graph.sort(key = lambda x : x[0]) # sort by coupling
    
    v2s       = [jj[0] for jj in graph]
    minus_two = [jj[1] for jj in graph]
    minus_one = [jj[2] for jj in graph]
    central   = [jj[3] for jj in graph]
    plus_one  = [jj[4] for jj in graph]
    plus_two  = [jj[5] for jj in graph]
    if not run_blind:
        obs = [jj[6] for jj in graph]
    
    plt.clf()
    plt.fill_between(v2s, minus_two, plus_two, color='gold', label=r'$\pm 2 \sigma$')
    plt.fill_between(v2s, minus_one, plus_one, color='forestgreen' , label=r'$\pm 1 \sigma$')
    plt.plot(v2s, central, color='red', label='central expected', linewidth=2)
    if not run_blind:
        plt.plot(v2s, obs, color='black', label='observed')    
    
    plt.axhline(y=1, color='black', linestyle='-')
    plt.xlabel(flavour)
    plt.ylabel('exclusion limit 95% CL')
    plt.title('HNL m = %d GeV %s' %(mass, signal_type))
    # plt.legend(loc='upper right')
    plt.legend()
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    #plt.tight_layout()
    plt.yscale('linear')
    plt.savefig('limit_m_%s_lin.pdf' %mass)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('limit_m_%s_log.pdf' %mass)

    # save the crossing for 2D limits
    limits2D[mass] = OrderedDict()

    # find the intersections    
    x_minus_two, y = intersection(np.array(v2s), np.array(minus_two), np.array(v2s), np.ones(len(v2s)))    
    x_minus_one, y = intersection(np.array(v2s), np.array(minus_one), np.array(v2s), np.ones(len(v2s)))    
    x_central  , y = intersection(np.array(v2s), np.array(central)  , np.array(v2s), np.ones(len(v2s)))    
    x_plus_one , y = intersection(np.array(v2s), np.array(plus_one) , np.array(v2s), np.ones(len(v2s)))    
    x_plus_two , y = intersection(np.array(v2s), np.array(plus_two) , np.array(v2s), np.ones(len(v2s)))    
    if not run_blind:
        x_obs , y = intersection(np.array(v2s), np.array(obs) , np.array(v2s), np.ones(len(v2s)))    

    limits2D[mass]['exp_minus_two'] = x_minus_two
    limits2D[mass]['exp_minus_one'] = x_minus_one
    limits2D[mass]['exp_central'  ] = x_central  
    limits2D[mass]['exp_plus_one' ] = x_plus_one 
    limits2D[mass]['exp_plus_two' ] = x_plus_two 
    if not run_blind:
        limits2D[mass]['obs'] = x_obs 

#     import sys ; sys.exit(0)


##########################################################################################
##########################################################################################

# import pickle
# with open('results.pck', 'r') as ff:
#     limits2D = pickle.load(ff)

import pickle
with open('results.pck', 'w') as ff:
    pickle.dump(limits2D, ff)
    
masses_obs       = []
masses_central   = []
masses_one_sigma = []
masses_two_sigma = []

minus_two = []
minus_one = []
central   = []
plus_one  = []
plus_two  = []
obs       = []

# go through the different mass points first left to right to catch the lower exclusion bound
# then right to left to catch the upper exclusion bound
for mass in sorted(limits2D.keys()):

    if len(limits2D[mass]['obs'])>0: 
        obs.append( min(limits2D[mass]['obs']) )
        masses_obs.append(mass)

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

    if len(limits2D[mass]['obs'])>1: 
        obs.append( max(limits2D[mass]['obs']) )
        masses_obs.append(mass)

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

## NORMAL
plt.fill_between(masses_two_sigma, minus_two, plus_two, color='gold'       , label=r'$\pm 2 \sigma$')
plt.fill_between(masses_one_sigma, minus_one, plus_one, color='forestgreen', label=r'$\pm 1 \sigma$')
plt.plot        (masses_central  , central            , color='red'        , label='central expected', linewidth=2)
plt.plot        (masses_obs      , obs                , color='black'      , label='observed', linewidth=2)


## SPLINE - 1
# from scipy.interpolate import make_interp_spline, BSpline
# xnew = np.linspace(min(masses_central), max(masses_central), 300) 
# 
# plus_two_spl  = make_interp_spline(masses_two_sigma, plus_two , k=3)
# plus_one_spl  = make_interp_spline(masses_one_sigma, plus_one , k=3)
# minus_two_spl = make_interp_spline(masses_two_sigma, minus_two, k=3)
# minus_one_spl = make_interp_spline(masses_one_sigma, minus_one, k=3)
# central_spl   = make_interp_spline(masses_central  , central  , k=3)
# obs_spl       = make_interp_spline(masses_obs      , obs      , k=3)

# plt.fill_between(masses_two_sigma, minus_two_spl, plus_two_spl, color='gold'       , label=r'$\pm 2 \sigma$')
# plt.fill_between(masses_one_sigma, minus_one_spl, plus_one_spl, color='forestgreen', label=r'$\pm 1 \sigma$')
# plt.plot        (masses_central  , central_spl                , color='red'        , label='central expected', linewidth=2)
# plt.plot        (masses_obs      , obs_spl                    , color='black'      , label='observed', linewidth=2)

## SPLINE - 2
# https://stackoverflow.com/questions/33962717/interpolating-a-closed-curve-using-scipy
# from scipy import interpolate

# append the starting x,y coordinates
# x = np.r_[masses_central, masses_central[0]]
# y = np.r_[central       , central[0]]
# x = masses_central
# y = central       

# fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
# is needed in order to force the spline fit to pass through all the input points.
# plt.clf()
# tck, u = interpolate.splprep([x, y], s=0, k=1, t=100, per=True)
# 
# # evaluate the spline fits for 1000 evenly spaced distance values
# xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)
# 
# plt.plot(xi, yi, color='red', label='central expected', linewidth=2)
# plt.plot(x, y, 'or', color='blue')


## SPLINE-3
# plt.clf()
# from scipy.interpolate import interp1d # Different interface to the same function
# 
# i = np.arange(len(x))
# 
# # 5x the original number of points
# interp_i = np.linspace(0, i.max(), 5 * i.max())
# 
# xi = interp1d(i, x, kind='cubic')(interp_i)
# yi = interp1d(i, y, kind='cubic')(interp_i)
# 
# plt.plot(xi, yi, color='red', label='central expected', linewidth=2)
# plt.plot(x, y, 'or', color='blue')



## SAVE 
plt.ylabel(flavour)
plt.xlabel('mass (GeV)')
plt.legend()
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
#plt.tight_layout()
plt.yscale('log')
plt.xscale('linear')
plt.grid(True)
plt.savefig('2d_hnl_limit.pdf')


# splines !
# https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot


    
