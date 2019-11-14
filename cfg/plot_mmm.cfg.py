from plotter import Plotter
from selections import Selections
from os.getpass import getuser as user
from os import environ as env

# TODO DO I PUT THiS HERE? OR RATHER SOMEWHERE ELSE? LIKE SELECTIONS? HAS TO BE AMONG THE FIRST INITS...!

if user == 'manzoni': 
     env['BASE_DIR'] = '...'

if user == 'cesareborgia': 
    env['BASE_DIR'] = '...'

cuts = Selections('mmm')

plotter = Plotter (channel        = 'mmm',
                   base_dir        = '/Users/cesareborgia/cernbox/ntuples/2018/',
                   post_fix        = 'HNLTreeProducer_mmm/tree.root',
                   # baseline_selection = cuts.selections['baseline_low_pT'],
                   # specific selection = cuts.selections['resonance_vetoes'],
                   lumi           = 59700.,
                   model          = 'NN/mmm/12Nov19_v0/net_model_weighted.h5', 
                   transformation = 'NN/mmm/12Nov19_v0/input_tranformation_weighted.pck',
                   features       = 'NN/mmm/12Nov19_v0/input_features.pck',
                   )

plotter.plot()
