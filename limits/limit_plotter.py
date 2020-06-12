import sys
import os
import glob
import re
import pickle
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from intersection import intersection
from utils import getMassList


def getOptions():
  from argparse import ArgumentParser
  parser = ArgumentParser(description='Run limit on a single mass/coupling point', add_help=True)
  parser.add_argument('--version', type=str, dest='version', help='version label', default='L1')
  parser.add_argument('--signal', type=str, dest='signal', help='signal under consideration', default='majorana', choices=['majorana', 'dirac'])
  parser.add_argument('--channels', type=str, dest='channels', help='channels to combine', default='mmm,mem_os,mem_ss')
  parser.add_argument('--run_blind', dest='run_blind', help='run blinded or unblinded', action='store_true', default=False)
  return parser.parse_args()


if __name__ == "__main__":

  opt=getOptions()

  # get the parsed info
  if 'mmm' in opt.channels or 'mem' in opt.channels: flavour = r'$|V|^2_{u}$'
  elif 'eee' in opt.channels or 'eem' in opt.channels: flavour = r'$|V|^2_{e}$'

  signal_type = opt.signal 

  # get the files 
  pathToResults = './results/limits/{}/'.format(opt.version)
  fileName = 'result*.txt'

  files = [f for f in glob.glob(pathToResults+fileName)]
 
  # get the list of the masses from the fileNames
  masses = getMassList(files)
 
  # needed for the 2D limit plot
  limits2D = OrderedDict()

  # create directory
  plotDir = './results/plots/{}'.format(opt.version) 
  os.system('mkdir -p {d}'.format(d=plotDir)) 
   
  for mass in masses:
    print '\nmass {}'.format(mass)

    v2s       = []
    obs       = []
    minus_two = []
    minus_one = []
    central   = []
    plus_one  = []
    plus_two  = []

    for limitFile in files:
      if 'm_{}_'.format(mass) not in limitFile: continue
   
      # for each mass, get the list of the couplings from the file name
      signal_coupling = re.findall(r'\d+', limitFile.split('/')[len(limitFile.split('/'))-1])
      coupling = '{}.{}Em{}'.format(signal_coupling[3], signal_coupling[4], signal_coupling[5])
      val_coupling = float('{}.{}e-{}'.format(signal_coupling[3], signal_coupling[4], signal_coupling[5]))
      
      try:
        thefile = open('{}/result_m_{}_v2_{}.txt'.format(pathToResults, mass, coupling), 'r')
        
        v2s.append(val_coupling)

        # get the necessary information from the result files
        content = thefile.readlines()
        for line in content:
          if 'Observed' in line:
            values = re.findall(r'\d+', line)
            val_obs = values[0] + '.' + values[1]
            obs.append(float(val_obs))
          if 'Expected  2.5' in line: 
            values = re.findall(r'\d+', line)
            val_minus_two = values[2] + '.' + values[3]
            minus_two.append(float(val_minus_two))
          elif 'Expected 16' in line: 
            values = re.findall(r'\d+', line)
            val_minus_one = values[2] + '.' + values[3]
            minus_one.append(float(val_minus_one))
          elif 'Expected 50' in line: 
            values = re.findall(r'\d+', line)
            val_central = values[2] + '.' + values[3]
            central.append(float(val_central))
          elif 'Expected 84' in line: 
            values = re.findall(r'\d+', line)
            val_plus_one = values[2] + '.' + values[3]
            plus_one.append(float(val_plus_one))
          elif 'Expected 97.5' in line: 
            values = re.findall(r'\d+', line)
            val_plus_two = values[2] + '.' + values[3]
            plus_two.append(float(val_plus_two))

      except:
        print 'Cannot open {}result_m_{}_v2_{}.txt'.format(pathToResults, mass, coupling)

    # check that the result files have all the necessary information
    if (len(minus_two) != len(minus_one)) or (len(minus_two) != len(central)) or (len(minus_two) != len(plus_one)) or (len(minus_two) != len(plus_two) or len(minus_two)==0): continue
    if not opt.run_blind:
      if len(obs)==0: 
        print 'WARNING: cannot plot unblinded if limits were produced blinded'
        print '--> Aborting'
        continue
   

    print '-> will plot 1D limit for mass {}'.format(mass)
    
    if not opt.run_blind:
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
    if not opt.run_blind:
        obs = [jj[6] for jj in graph]
    
    plt.clf()
    print '   couplings: {}'.format(v2s)
    plt.fill_between(v2s, minus_two, plus_two, color='gold', label=r'$\pm 2 \sigma$')
    plt.fill_between(v2s, minus_one, plus_one, color='forestgreen' , label=r'$\pm 1 \sigma$')
    plt.plot(v2s, central, color='red', label='central expected', linewidth=2)
    if not opt.run_blind:
        plt.plot(v2s, obs, color='black', label='observed')    
    
    plt.axhline(y=1, color='black', linestyle='-')
    plt.xlabel(flavour)
    plt.ylabel('exclusion limit 95% CL')
    plt.title('HNL m = %s GeV %s' %(mass, signal_type))
    plt.legend()
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.yscale('linear')
    plt.savefig('%s/limit_m_%s_lin.pdf' %(plotDir, mass))
    plt.savefig('%s/limit_m_%s_lin.png' %(plotDir, mass))
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('%s/limit_m_%s_log.pdf' %(plotDir, mass))
    plt.savefig('%s/limit_m_%s_log.png' %(plotDir, mass))
  
    
    # save the crossing for 2D limits
    limits2D[mass] = OrderedDict()

    # find the intersections    
    x_minus_two, y = intersection(np.array(v2s), np.array(minus_two), np.array(v2s), np.ones(len(v2s)))    
    x_minus_one, y = intersection(np.array(v2s), np.array(minus_one), np.array(v2s), np.ones(len(v2s)))    
    x_central  , y = intersection(np.array(v2s), np.array(central)  , np.array(v2s), np.ones(len(v2s)))    
    x_plus_one , y = intersection(np.array(v2s), np.array(plus_one) , np.array(v2s), np.ones(len(v2s)))    
    x_plus_two , y = intersection(np.array(v2s), np.array(plus_two) , np.array(v2s), np.ones(len(v2s)))    
    if not opt.run_blind:
        x_obs , y = intersection(np.array(v2s), np.array(obs) , np.array(v2s), np.ones(len(v2s)))    

    limits2D[mass]['exp_minus_two'] = x_minus_two
    limits2D[mass]['exp_minus_one'] = x_minus_one
    limits2D[mass]['exp_central'  ] = x_central  
    limits2D[mass]['exp_plus_one' ] = x_plus_one 
    limits2D[mass]['exp_plus_two' ] = x_plus_two 
    if not opt.run_blind:
        limits2D[mass]['obs'] = x_obs 
   

  print 'will plot 2D limits' 
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
      
      if not opt.run_blind:
        if len(limits2D[mass]['obs'])>0: 
            obs.append( min(limits2D[mass]['obs']) )
            masses_obs.append(mass)
      
      if len(limits2D[mass]['exp_central'])>0: 
          central.append( min(limits2D[mass]['exp_central']) )
          masses_central.append(mass)

      if len(limits2D[mass]['exp_minus_one'])>0 and len(limits2D[mass]['exp_plus_one' ])>0: 
          minus_one.append( min(limits2D[mass]['exp_minus_one']) )
          plus_one .append( min(limits2D[mass]['exp_plus_one' ]) )
          masses_one_sigma.append(float(mass))

      if len(limits2D[mass]['exp_minus_two'])>0 and len(limits2D[mass]['exp_plus_two' ])>0: 
          minus_two.append( min(limits2D[mass]['exp_minus_two']) )
          plus_two .append( min(limits2D[mass]['exp_plus_two' ]) )
          masses_two_sigma.append(float(mass))
      
  for mass in sorted(limits2D.keys(), reverse=True):

      if not opt.run_blind:
        if len(limits2D[mass]['obs'])>1: 
            obs.append( max(limits2D[mass]['obs']) )
            masses_obs.append(mass)
      
      if len(limits2D[mass]['exp_central'])>1: 
          central.append( max(limits2D[mass]['exp_central'  ]) )
          masses_central.append(mass)

      if len(limits2D[mass]['exp_minus_one'])>1 and len(limits2D[mass]['exp_plus_one' ])>1: 
          minus_one       .append( max(limits2D[mass]['exp_minus_one']) )
          plus_one        .append( max(limits2D[mass]['exp_plus_one' ]) )
          masses_one_sigma.append(float(mass))

      if len(limits2D[mass]['exp_minus_two'])>1 and len(limits2D[mass]['exp_plus_two' ])>1: 
          minus_two       .append( max(limits2D[mass]['exp_minus_two']) )
          plus_two        .append( max(limits2D[mass]['exp_plus_two' ]) )
          masses_two_sigma.append(float(mass))
  
  # plot the 2D limits
  plt.clf()
  plt.fill_between(masses_two_sigma, minus_two, plus_two, color='gold'       , label=r'$\pm 2 \sigma$')
  plt.fill_between(masses_one_sigma, minus_one, plus_one, color='forestgreen', label=r'$\pm 1 \sigma$')
  plt.plot(masses_central, central, color='red', label='central expected', linewidth=2)
  if not opt.run_blind:
    plt.plot(masses_obs, obs, color='black', label='observed', linewidth=2)

  plt.ylabel(flavour)
  plt.xlabel('mass (GeV)')
  plt.legend()
  plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
  plt.yscale('log')
  plt.xscale('linear')
  plt.grid(True)
  plt.savefig('{}/2d_hnl_limit.pdf'.format(plotDir))
  plt.savefig('{}/2d_hnl_limit.png'.format(plotDir))


