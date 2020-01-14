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

    'l1_pt>7',
#     'hnl_2d_disp_sig>20',
    'hnl_pt_12>15',
    'sv_cos>0.99',
    'sv_prob>0.001',
    'l0_reliso_rho_03<0.1',
    'l0_pt>32',
    'abs(l1_dz)<10',
    'abs(l2_dz)<10',
]

# extra selection to be applied on variables that don't exist
# in the root tree but they're created for the pandas dataset
pandas_selection = 'hnl_2d_disp_sig_alt>20'

selection_mc = selection + [cuts.selections['is_prompt_lepton']]
selection_tight = cuts.selections_pd['tight']

plotter = Plotter (channel          = ch+'_os',
                   base_dir         = env['NTUPLE_DIR'],
                   post_fix         = 'HNLTreeProducer/tree.root', # 'HNLTreeProducer_%s/tree.root' %ch,
                   selection_data   = selection,
                   selection_mc     = selection_mc,
                   selection_tight  = selection_tight,
                   pandas_selection = pandas_selection,
                   lumi             = 59700.,
#                    model            = env['NN_DIR'] + '/eem_191119_19h_46m/net_model_weighted.h5', 
#                    transformation   = env['NN_DIR'] + '/eem_191119_19h_46m/input_tranformation_weighted.pck',
#                    features         = env['NN_DIR'] + '/eem_191119_19h_46m/input_features.pck',
#                    model            = env['NN_DIR'] + '/eem_191119_20h_14m/net_model_weighted.h5', 
#                    transformation   = env['NN_DIR'] + '/eem_191119_20h_14m/input_tranformation_weighted.pck',
#                    features         = env['NN_DIR'] + '/eem_191119_20h_14m/input_features.pck',
#                    model            = env['NN_DIR'] + '/eem_191119_20h_20m/net_model_weighted.h5', 
#                    transformation   = env['NN_DIR'] + '/eem_191119_20h_20m/input_tranformation_weighted.pck',
#                    features         = env['NN_DIR'] + '/eem_191119_20h_20m/input_features.pck',
                   model            = env['NN_DIR'] + 'trainings/eem_191119_22h_30m/net_model_weighted.h5', 
                   transformation   = env['NN_DIR'] + 'trainings/eem_191119_22h_30m/input_tranformation_weighted.pck',
                   features         = env['NN_DIR'] + 'trainings/eem_191119_22h_30m/input_features.pck',
                   process_signals  = True,
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
    