from plotter import Plotter
from selections import Selections
from utils import set_paths
from os import environ as env

ch = 'eee'

set_paths(ch)
cuts = Selections(ch)

plotter = Plotter (channel         = ch,
                   base_dir        = env['BASE_DIR'],
                   post_fix        = 'HNLTreeProducer_%s/tree.root' %ch,

                   selection_data  = ' & '.join([ cuts.selections['pt_iso'], cuts.selections['baseline'], cuts.selections['vetoes_12_OS'], cuts.selections['vetoes_01_OS'], 
                                                  cuts.selections['vetoes_02_OS'], ]),

                   selection_mc    = ' & '.join([ cuts.selections['pt_iso'], cuts.selections['baseline'], cuts.selections['vetoes_12_OS'], cuts.selections['vetoes_01_OS'], 
                                                  cuts.selections['vetoes_02_OS'], cuts.selections['is_prompt_lepton'] ]),

                   selection_tight = cuts.selections_pd['tight'],

                   lumi            = 59700.,
                   model           = env['NN_DIR'] + 'net_model_weighted.h5', 
                   transformation  = env['NN_DIR'] + 'input_tranformation_weighted.pck',
                   features        = env['NN_DIR'] + 'input_features.pck',
                   plot_signals    = True,
                   blinded         = True,
                   )
from pdb import set_trace; set_trace()
plotter.plot()
