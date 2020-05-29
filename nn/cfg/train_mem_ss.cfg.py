import numpy as np
from NN.nn_trainer import Trainer
from plotter.selections import Selections
from plotter.utils import set_paths
from collections import OrderedDict
from os import environ as env

ch = 'mem'
set_paths(ch, 2018)
cuts = Selections(ch)

selection = [ 
    cuts.selections['pt_iso'], 
    cuts.selections['baseline'],
    'l0_q==l2_q', 
    cuts.selections['sideband'], 
]

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
composed_features['is_ss'         ] = lambda df : df.hnl_q_02!=0

trainer = Trainer (channel           = ch+'_ss',
                   base_dir          = env['NTUPLE_DIR'],
                   #post_fix          = 'HNLTreeProducer_%s/tree.root' %ch,
                   post_fix          = 'HNLTreeProducer/tree.root',
   
                   features          = ['l0_pt'              ,
                                        'l1_pt'              ,
                                        'l2_pt'              ,
                                        'hnl_dr_12'          ,
                                        'hnl_m_12'           ,
                                        'sv_prob'            ,
                                        'hnl_2d_disp'        ,],

                   composed_features = composed_features,

                   selection_data    = selection,
                   selection_mc      = selection + [cuts.selections['is_prompt_lepton']],
   
                   selection_tight   = cuts.selections_pd['tight'],
                   lumi              = 59700.
                   )

if __name__ == '__main__':
    trainer.train()
    pass
