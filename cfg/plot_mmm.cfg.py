from plotter import Plotter
from selections import Selections
from os import environ as env

cuts = Selections('mmm')

plotter = Plotter (channel         = 'mmm',
                   base_dir        = env['BASE_DIR'],
                   post_fix        = 'HNLTreeProducer_mmm/tree.root',

                   selection_data  = ' & '.join([ cuts.selections['pt_iso'], cuts.selections['baseline'], cuts.selections['vetoes_12_OS'], cuts.selections['vetoes_01_OS'], 
                                                 cuts.selections['vetoes_02_OS'], cuts.selections['zmm'] ]),

                   selection_mc    = ' & '.join([ selections_data, cuts.selections['is_prompt_lepton'] ]),

                   selection_tight = cuts.selections_pd['tight'],

                   lumi            = 59700.,
                   model           = env['NN_DIR'] + '/12Nov19_v0/net_model_weighted.h5', 
                   transformation  = env['NN_DIR'] + '/12Nov19_v0/input_tranformation_weighted.pck',
                   features        = env['NN_DIR'] + '/12Nov19_v0/input_features.pck',
                   plot_signals    = True,
                   blinded         = True,
                   )

plotter.plot()
