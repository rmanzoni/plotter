from os import environ as env
from plotter.plotter import Plotter
from plotter.selections import Selections
from plotter.utils import set_paths

ch = 'mmm'

set_paths(ch, 2018)
cuts = Selections(ch)

selection = [ 
    cuts.selections['pt_iso'], 
    cuts.selections['baseline'], 
    cuts.selections['vetoes_12_OS'], 
    cuts.selections['vetoes_01_OS'], 
    cuts.selections['vetoes_02_OS'],
]

plotter = Plotter (channel         = ch,
                   base_dir        = env['NTUPLE_DIR'],
                   post_fix        = 'HNLTreeProducer/tree.root', # 'HNLTreeProducer_%s/tree.root' %ch,
                   selection_data  = selection,
                   selection_mc    = selection + [cuts.selections['is_prompt_lepton']],
                   selection_tight = cuts.selections_pd['tight'],
                   lumi            = 59700.,
                   model           = env['NN_DIR'] + '/mmm/12Nov19_v0/net_model_weighted.h5', 
                   transformation  = env['NN_DIR'] + '/mmm/12Nov19_v0/input_tranformation_weighted.pck',
                   features        = env['NN_DIR'] + '/mmm/12Nov19_v0/input_features.pck',
                   process_signals = True,
                   plot_signals    = True,
                   blinded         = False,
                   )

if __name__ == '__main__':
    plotter.plot()
