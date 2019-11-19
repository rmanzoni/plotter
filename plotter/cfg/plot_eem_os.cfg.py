from os import environ as env
from plotter.plotter import Plotter
from plotter.selections import Selections
from plotter.utils import set_paths

ch = 'eem'

set_paths(ch, 2018)
cuts = Selections(ch)

selection = [ 
    cuts.selections['pt_iso'], 
    cuts.selections['baseline'], 
    cuts.selections['vetoes_01_OS'],
    'l0_q!=l1_q', 
    cuts.selections['sideband'], 
#     cuts.selections['signal_region'], 
]

plotter = Plotter (channel         = ch+'_os',
                   base_dir        = env['NTUPLE_DIR'],
                   post_fix        = 'HNLTreeProducer/tree.root', # 'HNLTreeProducer_%s/tree.root' %ch,
                   selection_data  = selection,
                   selection_mc    = selection + [cuts.selections['is_prompt_lepton']],
                   selection_tight = cuts.selections_pd['tight'],
                   lumi            = 59700.,
#                    model           = env['NN_DIR'] + '/eem_191119_19h_46m/net_model_weighted.h5', 
#                    transformation  = env['NN_DIR'] + '/eem_191119_19h_46m/input_tranformation_weighted.pck',
#                    features        = env['NN_DIR'] + '/eem_191119_19h_46m/input_features.pck',
#                    model           = env['NN_DIR'] + '/eem_191119_20h_14m/net_model_weighted.h5', 
#                    transformation  = env['NN_DIR'] + '/eem_191119_20h_14m/input_tranformation_weighted.pck',
#                    features        = env['NN_DIR'] + '/eem_191119_20h_14m/input_features.pck',
#                    model           = env['NN_DIR'] + '/eem_191119_20h_20m/net_model_weighted.h5', 
#                    transformation  = env['NN_DIR'] + '/eem_191119_20h_20m/input_tranformation_weighted.pck',
#                    features        = env['NN_DIR'] + '/eem_191119_20h_20m/input_features.pck',
                   model           = env['NN_DIR'] + '/eem_191119_22h_30m/net_model_weighted.h5', 
                   transformation  = env['NN_DIR'] + '/eem_191119_22h_30m/input_tranformation_weighted.pck',
                   features        = env['NN_DIR'] + '/eem_191119_22h_30m/input_features.pck',
                   process_signals = False,
                   plot_signals    = False,
                   blinded         = False,
                   )

if __name__ == '__main__':
    plotter.plot()
    pass
    