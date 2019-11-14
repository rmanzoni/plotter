from plotter import Plotter
from selections import Selections
from os import environ as env

cuts = Selections('mmm')

plotter = Plotter (channel        = 'mmm',
                   base_dir        = ,
                   post_fix        = 'HNLTreeProducer_mmm/tree.root',
                   # baseline_selection = cuts.selections['baseline_low_pT'],
                   # specific selection = cuts.selections['resonance_vetoes'],
                   lumi           = 59700.,
                   model          = 'NN/mmm/12Nov19_v0/net_model_weighted.h5', 
                   transformation = 'NN/mmm/12Nov19_v0/input_tranformation_weighted.pck',
                   features       = 'NN/mmm/12Nov19_v0/input_features.pck',
                   plot_signals   = True,
                   blinded        = True,
                   )

plotter.plot()
