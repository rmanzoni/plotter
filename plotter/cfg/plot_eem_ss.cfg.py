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
    'l0_q==l1_q', 
    cuts.selections['sideband'], 
#     cuts.selections['signal_region'], 

    'hnl_2d_disp_sig>20',
    'hnl_pt_12>15',
    'sv_cos>0.99',
    'sv_prob>0.005',
]

selection_mc = selection + [cuts.selections['is_prompt_lepton']]
selection_tight = cuts.selections_pd['tight']

plotter = Plotter (channel         = ch+'_ss',
                   base_dir        = env['NTUPLE_DIR'],
                   post_fix        = 'HNLTreeProducer/tree.root', # 'HNLTreeProducer_%s/tree.root' %ch,
                   selection_data  = selection,
                   selection_mc    = selection_mc,
                   selection_tight = selection_tight,
                   lumi            = 59700.,
#                    model           = env['NN_DIR'] + '/eem_191119_19h_46m/net_model_weighted.h5', 
#                    transformation  = env['NN_DIR'] + '/eem_191119_19h_46m/input_tranformation_weighted.pck',
#                    features        = env['NN_DIR'] + '/eem_191119_19h_46m/input_features.pck',
                   model           = env['NN_DIR'] + '/eem_191119_22h_30m/net_model_weighted.h5', 
                   transformation  = env['NN_DIR'] + '/eem_191119_22h_30m/input_tranformation_weighted.pck',
                   features        = env['NN_DIR'] + '/eem_191119_22h_30m/input_features.pck',
                   process_signals = True,
                   mini_signals    = True, # process only the signals that you'll plot
                   plot_signals    = True,
                   blinded         = True,
                   datacards       = ['hnl_m_12_lxy_0p5_to_2p0', 'hnl_m_12_lxy_lt_0p5', 'hnl_m_12_lxy_mt_2p0'], # FIXME! improve this to accept wildcards / regex
                   )

if __name__ == '__main__':

    plotter.plot()

    # save the plotter and all
    save_plotter_and_selections(plotter, selection, selection_mc, selection_tight)

    pass
    