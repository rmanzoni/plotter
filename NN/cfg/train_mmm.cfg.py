from NN.nn_trainer import Trainer
from selections import Selections
from utils import set_paths
from os import environ as env

ch = 'mmm'

set_paths(ch)
cuts = Selections(ch)

trainer = Trainer (channel         = ch,
                   base_dir        = env['NTUPLE_DIR'],
                   post_fix        = 'HNLTreeProducer_%s/tree.root' %ch,

                   features        = ['l0_pt'              ,
                                      'l1_pt'              ,
                                      'l2_pt'              ,
                                      'hnl_dr_12'          ,
                                      'hnl_m_12'           ,
                                      'sv_prob'            ,
                                      'hnl_2d_disp'        ,
                                      ],

                   selection_data  = ' & '.join([ cuts.selections['pt_iso'], cuts.selections['baseline'], cuts.selections['vetoes_12_OS'], cuts.selections['vetoes_01_OS'], 
                                                  cuts.selections['vetoes_02_OS'], ]),

                   selection_mc    = ' & '.join([ cuts.selections['pt_iso'], cuts.selections['baseline'], cuts.selections['vetoes_12_OS'], cuts.selections['vetoes_01_OS'], 
                                                  cuts.selections['vetoes_02_OS'], cuts.selections['is_prompt_lepton'] ]),

                   selection_tight = cuts.selections_pd['tight'],
                   lumi = 59700.
                   )

trainer.train()
