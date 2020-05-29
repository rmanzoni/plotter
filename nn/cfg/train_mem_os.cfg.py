from NN.nn_trainer import Trainer
from plotter.selections import Selections
from plotter.utils import set_paths
from os import environ as env

ch = 'mem'
set_paths(ch, 2018)
cuts = Selections(ch)

selection = [ 
    cuts.selections['pt_iso'], 
    cuts.selections['baseline'],
    cuts.selections['vetoes_02_OS'],
    cuts.selections['sideband'], 
]

trainer = Trainer (channel         = ch+'_os',
                   base_dir        = env['NTUPLE_DIR'],
                   #post_fix        = 'HNLTreeProducer_%s/tree.root' %ch,
                   post_fix        = 'HNLTreeProducer/tree.root',

                   features        = ['l0_pt'              ,
                                      'l1_pt'              ,
                                      'l2_pt'              ,
                                      'hnl_dr_12'          ,
                                      'hnl_m_12'           ,
                                      'sv_prob'            ,
                                      'hnl_2d_disp'        ,],
                                      
                   selection_data  = selection,
                   selection_mc    = selection + [cuts.selections['is_prompt_lepton']],

                   selection_tight = cuts.selections_pd['tight'],
                   lumi = 59700.
                   )

if __name__ == '__main__':
    trainer.train()
    pass
