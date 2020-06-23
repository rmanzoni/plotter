import re

def getPointGrid(files):
  return re.findall(r'\d+', files.split('/')[len(files.split('/'))-1])
  

def getMassList(files): 
  masses = []

  for limitFile in files:
    signal_mass = getPointGrid(limitFile) 
    if '{}.{}'.format(signal_mass[0], signal_mass[1]) not in masses:
      masses.append('{}.{}'.format(signal_mass[0], signal_mass[1])) 

  masses.sort()
  return masses


def getCoupling(files):
  signal_coupling = getPointGrid(files) 
  coupling = '{}.{}Em{}'.format(signal_coupling[2], signal_coupling[3], signal_coupling[4])
  return coupling




