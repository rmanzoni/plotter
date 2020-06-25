from datetime import datetime
from os import environ as env
from os.path import exists as ensure_path
from os import makedirs
import pickle

def get_time_str():
    today   = datetime.now()
    date    = today.strftime('%y%m%d')
    hour    = str(today.hour)
    minit   = str(today.minute)
    time_str = date + '_' + hour + 'h_' + minit + 'm'
    return time_str

def save_plotter_and_selections(plotter, sel_data, sel_mc, sel_tight, training_name=''):

    with open('/'.join([plotter.plt_dir, 'plotter.pck']), 'wb') as plt_file:
        pickle.dump(plotter, plt_file)

    with open('/'.join([plotter.plt_dir, 'selections.py']), 'a') as selection_file:
        
        print('selection_data = [', file=selection_file)
        for isel in sel_data:
            print("\t'%s'," %isel , file=selection_file)
        print(']', file=selection_file)

        print('\n'*2+'#'*80+'\n'*2, file=selection_file)

        print('selection_mc = ['  , file=selection_file)
        for isel in sel_mc:
            print("\t'%s'," %isel , file=selection_file)
        print(']', file=selection_file)

        print('\n'*2+'#'*80+'\n'*2, file=selection_file)

        print("'selection_tight = '%s'" %sel_tight, file=selection_file)

    with open('/'.join([plotter.plt_dir, 'training.txt']), 'a') as training_file:
        print('NN training: ', training_name, file=training_file)

