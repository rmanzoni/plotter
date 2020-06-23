import sys 
import os
import re
import subprocess
from decimal import Decimal


def getOptions():
     from argparse import ArgumentParser
     parser = ArgumentParser(description='Run limit on a single mass/coupling point', add_help=True)
     parser.add_argument('--mass', type=str, dest='mass', help='mass', default='1.0')
     parser.add_argument('--coupling', type=str, dest='coupling', help='coupling', default='1e-07')
     parser.add_argument('--run_blind', type=int, dest='run_blind', help='run blind?', default='1')
     parser.add_argument('--version', type=str, dest='version', help='version label', default='L1')
     return parser.parse_args()


if __name__ == "__main__":

        opt = getOptions()
        mass=opt.mass
        coupling=opt.coupling

        command = 'combine -M AsymptoticLimits datacards_combined/{v}/datacard_combined_{m}_{c}.txt'.format(v=opt.version, m=mass, c=coupling)
        if opt.run_blind:
          command += ' --run blind'
        
        print '\t\t',command
         
        results = subprocess.check_output(command.split())
         

        #result_file_name = ('results/limits/%s/result_m_%s_v2_%.1E.txt' %(opt.version, str(mass), Decimal(coupling))).replace('-', 'm') 
        result_file_name = 'results/limits/{}/result_m_{}_v2_{}.txt'.format(opt.version, str(mass), str(coupling)) 
        with open(result_file_name, 'w') as ff:
            print >> ff, results



