from datetime import datetime
from os import environ as env
from os.path import exists as ensure_path
from os import makedirs
from getpass import getuser as user

def set_paths(channel, year):
    assert channel in ['mmm', 'mem', 'eem', 'eee'], 'ERROR: Channel not valid.'
    assert year in [2017, 2018], 'ERROR: Year not valid.'    
    if user() == 'manzoni': 
        env['BASE_DIR'] = '/Users/manzoni/Documents/HNL/ntuples/%d/%s' %(year, channel)
        env['PLOT_DIR'] = '/Users/manzoni/Documents/HNL/plotter/plots_%d_%s/' %(year, channel)
        env['NN_DIR']   = '/Users/manzoni/Documents/HNL/plotter/nn/'

    if user() == 'cesareborgia': 
        env['BASE_DIR'] = '/Users/cesareborgia/cernbox/ntuples/%d/'       %year
        env['PLOT_DIR'] = '/Users/cesareborgia/cernbox/plots/plotter/%s/' %channel
        env['NN_DIR']   = '/Users/cesareborgia/HNL/plotter/nn/%s/'        %channel

def get_time_str():
    today   = datetime.now()
    date    = today.strftime('%y%m%d')
    hour    = str(today.hour)
    minit   = str(today.minute)
    time_str = date + '_' + hour + 'h_' + minit + 'm/'
    return time_str

def plot_dir():
    plot_dir = env['PLOT_DIR'] + get_time_str()
    if not ensure_path(plot_dir): makedirs(plot_dir)       #mkdir(plot_dir)
    return  plot_dir

def nn_dir():
    nn_dir = env['NN_DIR'] + get_time_str()
    if not ensure_path(nn_dir): mkdir(nn_dir)
    return  nn_dir
