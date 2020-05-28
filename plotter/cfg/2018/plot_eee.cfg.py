from os import environ as env
from plotter.objects.plotter import Plotter
from plotter.objects.selections import Selections
from plotter.objects.utils import save_plotter_and_selections

ch = 'eee'

cuts = Selections(ch)

selection = [ 
    cuts.selections['pt_iso'], 
    cuts.selections['baseline'], 
    cuts.selections['vetoes_12_OS'], 
    cuts.selections['vetoes_01_OS'], 
    cuts.selections['vetoes_02_OS'],
    cuts.selections['signal_region'], 
#     cuts.selections['sideband'], 
]

# extra selection to be applied on variables that don't exist
# in the root tree but they're created for the pandas dataset
# pandas_selection = 'hnl_2d_disp_sig_alt>20'
pandas_selection = ''

selection_mc = selection + [cuts.selections['is_prompt_lepton']]
selection_tight = cuts.selections_pd['tight']

training = '2018/all_channels__200528_23h_35m'
# training = 'all_channels_200526_12h_46m'
# training = 'all_channels_200525_19h_38m'
# training = 'all_channels_200525_18h_55m'
# training = 'all_channels_200523_22h_39m' #<==== GOOD
# training = 'all_channels_200523_15h_53m'
# training = 'all_channels_200523_15h_3m'
# training = 'all_channels_200523_15h_16m'

plotter = Plotter (
    channel          = ch,
    year             = 2018,
    plot_dir         = '/'.join([env['BASE_DIR'], 'plotter', 'plots', '2018']), 
    base_dir         = '/'.join([env['BASE_DIR'], 'ntuples', 'may20', '2018']),
    post_fix         = 'HNLTreeProducer_%s/tree.root' %ch,
    dir_suffix       = 'signal_dd_datacards', #'signal',

    selection_data   = selection,
    selection_mc     = selection_mc,
    selection_tight  = selection_tight,
    pandas_selection = pandas_selection,

    lumi             = 59700.,

    model            = '/'.join([env['BASE_DIR'], 'nn', 'trainings', training, 'net_model_weighted.h5'           ]), 
    transformation   = '/'.join([env['BASE_DIR'], 'nn', 'trainings', training, 'input_tranformation_weighted.pck']),
    features         = '/'.join([env['BASE_DIR'], 'nn', 'trainings', training, 'input_features.pck'              ]),

    process_signals  = True, # switch off for control regions
    mini_signals     = False, # process only the signals that you'll plot
    plot_signals     = True, 
    blinded          = False,

    datacards        = ['hnl_m_12_lxy_lt_0p5', 'hnl_m_12_lxy_0p5_to_1p5', 'hnl_m_12_lxy_1p5_to_4p0', 'hnl_m_12_lxy_mt_4p0'], # FIXME! improve this to accept wildcards / regex

    mc_subtraction   = True,
    
    data_driven      = True,
)

if __name__ == '__main__':
    plotter.plot()
    # save the plotter and all
    save_plotter_and_selections(plotter, selection, selection_mc, selection_tight)
    pass
    
