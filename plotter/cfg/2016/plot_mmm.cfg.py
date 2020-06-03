from os import environ as env
from plotter.objects.plotter import Plotter
from plotter.objects.selections import Selections
from plotter.objects.utils import save_plotter_and_selections

ch = 'mmm'

cuts = Selections(ch)

selection = [ 
    cuts.selections['pt_iso'], 
    cuts.selections['baseline'], 
    cuts.selections['vetoes_12_OS'], 
    cuts.selections['vetoes_01_OS'], 
    cuts.selections['vetoes_02_OS'],
#     cuts.selections['signal_region'], 
    cuts.selections['sideband'], 
    '(hlt_IsoMu24 | hlt_IsoTkMu24)',
]

# extra selection to be applied on variables that don't exist
# in the root tree but they're created for the pandas dataset
# pandas_selection = 'hnl_2d_disp_sig_alt>20'
pandas_selection = ''

selection_mc = selection + [cuts.selections['is_prompt_lepton']]
selection_tight = cuts.selections_pd['tight']

training = 'run2/all_channels__200601_18h_20m'
# training = 'run2/all_channels__200601_17h_19m'
# training = '2016/all_channels__200601_15h_57m'

plotter = Plotter (
    channel          = ch,
    year             = 2016,
    plot_dir         = '/'.join([env['BASE_DIR'], 'plotter', 'plots', '2016']), 
    base_dir         = '/'.join([env['BASE_DIR'], 'ntuples', 'may20', '2016']),
    post_fix         = 'HNLTreeProducer_%s/tree.root' %ch,
#     dir_suffix       = 'signal_dd_datacards', #'signal',
    dir_suffix       = 'sideband', 

    selection_data   = selection,
    selection_mc     = selection_mc,
    selection_tight  = selection_tight,
    pandas_selection = pandas_selection,

    lumi             = 35900.,

    model            = '/'.join([env['BASE_DIR'], 'nn', 'trainings', training, 'net_model_weighted.h5'           ]), 
    transformation   = '/'.join([env['BASE_DIR'], 'nn', 'trainings', training, 'input_tranformation_weighted.pck']),
    features         = '/'.join([env['BASE_DIR'], 'nn', 'trainings', training, 'input_features.pck'              ]),

    process_signals  = True, # switch off for control regions
    mini_signals     = True, # process only the signals that you'll plot
    plot_signals     = True, 
    blinded          = False,

    datacards        = ['log_hnl_2d_disp'        , 
                        'hnl_m_12'               ,  
                        'hnl_m_12_lxy_lt_0p5'    , 
                        'hnl_m_12_lxy_0p5_to_1p5', 
                        'hnl_m_12_lxy_1p5_to_4p0', 
                        'hnl_m_12_lxy_mt_4p0'], # FIXME! improve this to accept wildcards / regex
    
    mc_subtraction   = True,
    
    data_driven      = True,
)

if __name__ == '__main__':
    plotter.plot()
    # save the plotter and all
    save_plotter_and_selections(plotter, selection, selection_mc, selection_tight)
    pass
    
