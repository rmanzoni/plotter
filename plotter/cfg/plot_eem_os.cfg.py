from os import environ as env
from plotter.plotter import Plotter
from plotter.selections import Selections
from plotter.utils import set_paths, save_plotter_and_selections

ch = 'eem'

set_paths(ch, 2018)
cuts = Selections(ch)

selection = [ 
    cuts.selections['pt_iso'], 
    cuts.selections['baseline'], 
    cuts.selections['vetoes_01_OS'],
    'l0_q!=l1_q', 
#     cuts.selections['sideband'], 
    cuts.selections['signal_region'], 
#     'l1_pt>7',
    'l0_pt>32',
]

# extra selection to be applied on variables that don't exist
# in the root tree but they're created for the pandas dataset
# pandas_selection = 'hnl_2d_disp_sig_alt>20'
pandas_selection = ''

selection_mc = selection + [cuts.selections['is_prompt_lepton']]
selection_tight = cuts.selections_pd['tight']

training = 'all_channels_200526_12h_46m'
# training = 'all_channels_200525_19h_38m'
# training = 'all_channels_200525_18h_55m'
# training = 'all_channels_200523_22h_39m' #<==== GOOD
# training = 'all_channels_200523_15h_3m'
# training = 'all_channels_200523_15h_16m'

plotter = Plotter (channel          = ch+'_os',
                   base_dir         = '/Users/manzoni/Documents/HNL/ntuples/20may20', #env['NTUPLE_DIR'],
                   post_fix         = 'HNLTreeProducer_%s/tree.root' %ch,
                   selection_data   = selection,
                   selection_mc     = selection_mc,
                   selection_tight  = selection_tight,
                   pandas_selection = pandas_selection,
                   lumi             = 59700.,

                   model            = '/'.join([env['NN_DIR'], 'trainings', training, 'net_model_weighted.h5'           ]), 
                   transformation   = '/'.join([env['NN_DIR'], 'trainings', training, 'input_tranformation_weighted.pck']),
                   features         = '/'.join([env['NN_DIR'], 'trainings', training, 'input_features.pck'              ]),

                   process_signals  = True, # switch off for control regions
                   mini_signals     = True, # process only the signals that you'll plot
                   plot_signals     = True, 
                   blinded          = False,

                   datacards        = ['hnl_m_12_lxy_lt_0p5', 'hnl_m_12_lxy_0p5_to_1p5', 'hnl_m_12_lxy_1p5_to_4p0', 'hnl_m_12_lxy_mt_4p0'], # FIXME! improve this to accept wildcards / regex

                   mc_subtraction   = True,
                   
                   dir_suffix       = 'signal',
                   )

if __name__ == '__main__':
    plotter.plot()
    # save the plotter and all
    save_plotter_and_selections(plotter, selection, selection_mc, selection_tight)
    pass
