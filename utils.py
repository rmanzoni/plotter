from datetime import datetime
from os import environ as env
from os.path import exists as ensure_path
from os import mkdir
from getpass import getuser as user

def set_paths(channel):
    assert channel in ['mmm', 'mem', 'eem', 'eee'], 'ERROR: Channel not valid.'
    if user() == 'manzoni': 
        env['BASE_DIR'] = '/Users/manzoni/Documents/efficiencyNN/HNL/%s/ntuples/' %channel
        env['PLOT_DIR'] = '/Users/manzoni/Documents/efficiencyNN/HNL/%s/plots/'   %channel
        env['NN_DIR']   = '/Users/manzoni/Documents/efficiencyNN/HNL/plotter/NN/'

    if user() == 'cesareborgia': 
        env['BASE_DIR'] = '/Users/cesareborgia/cernbox/ntuples/2018/'
        env['PLOT_DIR'] = '/Users/cesareborgia/cernbox/plots/plotter/%s/' %channel
        env['NN_DIR']   = '/Users/cesareborgia/HNL/plotter/NN/%s/'        %channel

def get_time_str():
    today   = datetime.now()
    date    = today.strftime('%y%m%d')
    hour    = str(today.hour)
    minit   = str(today.minute)
    time_str = date + '_' + hour + 'h_' + minit + 'm/'
    return time_str

def plot_dir():
    plot_dir = env['PLOT_DIR'] + get_time_str()
    if not ensure_path(plot_dir): mkdir(plot_dir)
    return  plot_dir
