import numpy as np
from NN.nn_parametric_trainer import Trainer
from plotter.selections import Selections
from plotter.utils import set_paths
from collections import OrderedDict
from os import environ as env

ch = 'mmm'
set_paths(ch, 2018)

extra_selections = [
    'hnl_2d_disp_sig>20',
    'hnl_pt_12>15',
    'sv_cos>0.99',
    'sv_prob>0.001',
]

cuts_mmm = Selections('mmm')
selection_mmm = [ 
    cuts_mmm.selections['pt_iso'], 
    cuts_mmm.selections['baseline'], 
    cuts_mmm.selections['vetoes_12_OS'], 
    cuts_mmm.selections['vetoes_01_OS'], 
    cuts_mmm.selections['vetoes_02_OS'],
    cuts_mmm.selections['sideband'], 
] + extra_selections

cuts_mem = Selections('mem')
selection_mem = [ 
    cuts_mem.selections['pt_iso'], 
    cuts_mem.selections['baseline'],
    cuts_mem.selections['sideband'], 
    cuts_mem.selections['vetoes_02_OS'],
] + extra_selections

cuts_eee = Selections('eee')
selection_eee = [ 
    cuts_eee.selections['pt_iso'], 
    cuts_eee.selections['baseline'], 
    cuts_eee.selections['vetoes_12_OS'], 
    cuts_eee.selections['vetoes_01_OS'], 
    cuts_eee.selections['vetoes_02_OS'],
    cuts_eee.selections['sideband'], 
] + extra_selections

cuts_eem = Selections('eem')
selection_eem = [ 
    cuts_eem.selections['pt_iso'], 
    cuts_eem.selections['baseline'],
    cuts_eem.selections['sideband'], 
    cuts_eem.selections['vetoes_01_OS'],
] + extra_selections

composed_features = OrderedDict()

composed_features['abs_l0_eta'    ] = lambda df : np.abs(df.l0_eta)
composed_features['abs_l1_eta'    ] = lambda df : np.abs(df.l1_eta)
composed_features['abs_l2_eta'    ] = lambda df : np.abs(df.l2_eta)
composed_features['log_abs_l0_dxy'] = lambda df : np.log10(np.abs(df.l0_dxy))
composed_features['log_abs_l0_dz' ] = lambda df : np.log10(np.abs(df.l0_dz ))
composed_features['log_abs_l1_dxy'] = lambda df : np.log10(np.abs(df.l1_dxy))
composed_features['log_abs_l1_dz' ] = lambda df : np.log10(np.abs(df.l1_dz ))
composed_features['log_abs_l2_dxy'] = lambda df : np.log10(np.abs(df.l2_dxy))
composed_features['log_abs_l2_dz' ] = lambda df : np.log10(np.abs(df.l2_dz ))
composed_features['abs_q_02'      ] = lambda df : np.abs(df.hnl_q_02)
composed_features['abs_q_01'      ] = lambda df : np.abs(df.hnl_q_01)

# https://stackoverflow.com/questions/20528328/numpy-logical-or-for-more-than-two-arguments
# save a label to distinguish different channels
# 1 = mmm
# 2 = mem_os
# 3 = mem_ss
# 4 = eee
# 5 = eem_os
# 6 = eem_ss
# composed_features['channel' ] = lambda df : 1 * (np.abs(df.l0_pdgid)==13 and np.abs(df.l1_pdgid)==13 and np.abs(df.l2_pdgid)==13) + 2 * (np.abs(df.l0_pdgid)==13 and np.abs(df.l1_pdgid)==11 and np.abs(df.l2_pdgid)==13 and df.hnl_q_02!=0) + 3 * (np.abs(df.l0_pdgid)==13 and np.abs(df.l1_pdgid)==11 and np.abs(df.l2_pdgid)==13 and df.hnl_q_02==0) + 4 * (np.abs(df.l0_pdgid)==11 and np.abs(df.l1_pdgid)==11 and np.abs(df.l2_pdgid)==11) + 5 * (np.abs(df.l0_pdgid)==11 and np.abs(df.l1_pdgid)==11 and np.abs(df.l2_pdgid)==13 and df.hnl_q_02!=0) + 6 * (np.abs(df.l0_pdgid)==11 and np.abs(df.l1_pdgid)==11 and np.abs(df.l2_pdgid)==13 and df.hnl_q_02==0)

trainer = Trainer (channel         = 'all_channels',
                   base_dir        = env['NTUPLE_DIR'],
                   #post_fix        = 'HNLTreeProducer_%s/tree.root' %ch,
                   post_fix        = 'HNLTreeProducer/tree.root',

                   features        = ['l0_pt'      ,
                                      'l1_pt'      ,
                                      'l2_pt'      ,
                                      'hnl_dr_12'  ,
                                      'hnl_m_12'   ,
                                      'sv_prob'    ,
                                      'hnl_2d_disp',
                                      'channel'    ,],
                   
                   composed_features = composed_features,
                   
                   selection_data_mmm  = selection_mmm,
                   selection_mc_mmm    = selection_mmm + [cuts_mmm.selections['is_prompt_lepton']],

                   selection_data_mem  = selection_mem,
                   selection_mc_mem    = selection_mem + [cuts_mem.selections['is_prompt_lepton']],

                   selection_data_eee  = selection_eee,
                   selection_mc_eee    = selection_eee + [cuts_eee.selections['is_prompt_lepton']],

                   selection_data_eem  = selection_eem,
                   selection_mc_eem    = selection_eem + [cuts_eem.selections['is_prompt_lepton']],

                   selection_tight = cuts_mmm.selections_pd['tight'],
                   lumi = 59700.,
                   
                   epochs = 40,
                   )

if __name__ == '__main__':
    trainer.train()
