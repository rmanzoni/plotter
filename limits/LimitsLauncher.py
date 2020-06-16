import sys
import os
import glob
import re
from utils import getMassList, getCoupling

'''

This script allows to launch the different steps in the limit production
  -> combination of the datacards between displacement bins/years/flavour channels
  -> computation of the limit using combine
  -> plotting the limits

Usage:
  1. pdate the User's decision board to the studied case  
  2. python LimitsLauncher.py


'''


#"----------------User's decision board-----------------"

# choose which scripts to run
do_combineDatacards = False
do_produceLimits = False
do_producePlots = True   #note that this flag can be turned to true only when the limit results have been produced


version_label = 'test3'

# path to datacards # Note that it will get /<year>/<channel> appended
path_to_datacards = './datacards'

# which years to process
years = ['2016', '2017', '2018'] 

# which channels
channels = ['mmm', 'mem_os', 'mem_ss']

# run blind
run_blind = True

# submit limits on the batch
submit_batch = True

datacard_wildcard = 'datacard*hnl_m_12*.txt' 

# choose between 'dirac' and 'majorana'
signal_type = 'majorana'

# you may want to select the mass values you would like to run on
mass_whiteList = ['10.0', '6.0'] #['10.0']

# you may want to select the mass values you would like to ignore
mass_blackList = [] 

# you may want to select the coupling values you would like to run on
coupling_whiteList = []

# you may want to select the coupling values you would like to ignore
coupling_blackList = [] #['1e-07'] 

#'------------------------------------------------------'


# forming strings in parsing format
def getStringParser(input):
  idx = 0
  string_tmp = ''
  while idx < len(input):
    string_tmp += '{},'.format(input[idx])
    idx = idx + 1
  string = string_tmp[0:len(string_tmp)-1]
  if len(input)!= 0:
    return string
  else:
    return None

#.-.-.-.-.-.-.-.-.-.-.-.
if do_combineDatacards:
  print 'Will create the datacards combined between the years {} and channels {}'.format(years, channels)

  command_datacard = 'python combine_datacards.py --version {ver} --signal {sig} --years {ys} --channels {ch} --pathDC {pdc} --wildcard {wc} --mass_whitelist {mwl} --mass_blacklist {mbl} --coupling_whitelist {cwl} --coupling_blacklist {cbl}'.format(ver=version_label, sig=signal_type, ys=getStringParser(years), ch=getStringParser(channels), pdc=path_to_datacards, wc=datacard_wildcard, mwl=getStringParser(mass_whiteList), mbl=getStringParser(mass_blackList), cwl=getStringParser(coupling_whiteList), cbl=getStringParser(coupling_blackList))
  if submit_batch:
    command_datacard += ' --submit_batch'
  if run_blind:
    command_datacard += ' --run_blind'
  

  print command_datacard

  os.system(command_datacard)

  print '\nDone with combining the datacards'


#.-.-.-.-.-.-.-.-.-.-.-.
if do_produceLimits:
  print 'Will run the limits production tool'
  
  os.system('mkdir -p results/limits/{}'.format(version_label)) 
  
  if run_blind: flag_blind = 1
  else: flag_blind = 0
  
  # get the files 
  pathToResults = './datacards_combined/{}/'.format(version_label)
  fileName = 'datacard_combined*txt'

  files = [f for f in glob.glob(pathToResults+fileName)]

  masses = getMassList(files)
  
  for mass in masses:
    print '\nmass {}'.format(mass)
    
    for limitFile in files:
      
      if '_{}_'.format(mass) not in limitFile: continue
      coupling = getCoupling(limitFile)
      
      print '\n will produce limit for mass {} and coupling {}'.format(mass, coupling)

      command_sh = 'sh submitter.sh {m} {c} {fb} {v}'.format(m=str(mass), c=str(coupling), fb=flag_blind, v=version_label)
      command_sh_batch = 'sbatch -p wn --account=t3 -o logs/{v}/limits_{m}_{c}.log -e logs/{v}/limits_{m}_{c}.log --job-name=limits_{m}_{c} --time=0-01:00 submitter.sh {m} {c} {fb} {v}'.format(v=version_label, m=str(mass), c=str(coupling), fb=flag_blind)

      if submit_batch:
        os.system(command_sh_batch) 
        print '\t\t-> limit job submitted'
      else:
        print '\t\t-> running limit'
        os.system(command_sh) 

  print '\nDone with the limit submission'


#.-.-.-.-.-.-.-.-.-.-.-.
if do_producePlots:
 
  print 'will run the limit plotter'

  command_plotter = 'python limit_plotter.py --version {ver} --signal {sig} --channels {ch}  --mass_whitelist {mwl} --mass_blacklist {mbl} --coupling_whitelist {cwl} --coupling_blacklist {cbl}'.format(ver=version_label, sig=signal_type, ch=getStringParser(channels), mwl=getStringParser(mass_whiteList), mbl=getStringParser(mass_blackList), cwl=getStringParser(coupling_whiteList), cbl=getStringParser(coupling_blackList))
  if run_blind:
    command_plotter += ' --run_blind'

  os.system(command_plotter)

  print '\nDone with the plotting'


