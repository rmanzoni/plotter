from os import environ as env
from plotter.plotter import Plotter
from plotter.selections import Selections
from plotter.utils import set_paths, save_plotter_and_selections

ch = 'mmm'

set_paths(ch, 2018)
cuts = Selections(ch)

selection = [ 
    cuts.selections['pt_iso'], 
    cuts.selections['baseline'], 
    cuts.selections['vetoes_12_OS'], 
    cuts.selections['vetoes_01_OS'], 
    cuts.selections['vetoes_02_OS'],
    cuts.selections['signal_region'], 
#     cuts.selections['sideband'], 

#     'hnl_2d_disp_sig>20',
    'hnl_pt_12>15',
    'sv_cos>0.99',
    'sv_prob>0.001',
    'l0_reliso_rho_03<0.1',
    'l0_pt>25',
    'abs(l1_dz)<10',
    'abs(l2_dz)<10',
]

# extra selection to be applied on variables that don't exist
# in the root tree but they're created for the pandas dataset
pandas_selection = 'hnl_2d_disp_sig_alt>20'

selection_mc = selection + [cuts.selections['is_prompt_lepton']]
selection_tight = cuts.selections_pd['tight']

plotter = Plotter (channel          = ch,
                   base_dir         = env['NTUPLE_DIR'],
                   post_fix         = 'HNLTreeProducer/tree.root', # 'HNLTreeProducer_%s/tree.root' %ch,
                   selection_data   = selection,
                   selection_mc     = selection_mc,
                   selection_tight  = selection_tight,
                   pandas_selection = pandas_selection,
                   lumi             = 59700.,
#                    model            = env['NN_DIR'] + '/trainings/mmm/12Nov19_v0/net_model_weighted.h5', 
#                    transformation   = env['NN_DIR'] + '/trainings/mmm/12Nov19_v0/input_tranformation_weighted.pck',
#                    features         = env['NN_DIR'] + '/trainings/mmm/12Nov19_v0/input_features.pck',
                   model            = env['NN_DIR'] + '/all_channels_191126_9h_45m/net_model_weighted.h5', 
                   transformation   = env['NN_DIR'] + '/all_channels_191126_9h_45m/input_tranformation_weighted.pck',
                   features         = env['NN_DIR'] + '/all_channels_191126_9h_45m/input_features.pck',
                   process_signals  = True, # switch off for control regions
                   mini_signals     = True, # process only the signals that you'll plot
                   plot_signals     = True, 
                   blinded          = True,
#                    datacards        = ['hnl_m_12_lxy_0p5_to_2p0', 'hnl_m_12_lxy_lt_0p5', 'hnl_m_12_lxy_mt_2p0'], # FIXME! improve this to accept wildcards / regex
                   datacards        = ['hnl_m_12_lxy_lt_0p5', 'hnl_m_12_lxy_0p5_to_1p5', 'hnl_m_12_lxy_1p5_to_4p0', 'hnl_m_12_lxy_mt_4p0'], # FIXME! improve this to accept wildcards / regex
                   )

if __name__ == '__main__':
    plotter.plot()
    # save the plotter and all
    save_plotter_and_selections(plotter, selection, selection_mc, selection_tight)
    pass
