import numpy as np

# variables
variables = [
    ('hnl_m_01'       , np.linspace(0.,120., 30 + 1), r'$m_{01}$ (GeV)'      , 'events'),
    ('hnl_m_12'       , np.linspace(0., 12., 12 + 1), r'$m_{12}$ (GeV)'      , 'events'),
    ('hnl_m_02'       , np.linspace(0.,120., 30 + 1), r'$m_{02}$ (GeV)'      , 'events'),

    ('hnl_2d_disp'    , np.linspace( 0, 30, 25 + 1) , r'$L_{xy}$ (cm)'       , 'events'),
    ('hnl_2d_disp_sig', np.linspace( 0,200, 25 + 1) , r'$L_{xy}/\sigma_{xy}$', 'events'),
    ('nbj'            , np.linspace( 0,  5,  5 + 1) , '#b-jet'               , 'events'),
    ('hnl_w_vis_m'    , np.linspace( 0,150, 40 + 1) , r'$m_{3l}$'            , 'events'),
    ('hnl_q_01'       , np.linspace(-3,  3,  3 + 1) , r'$q_{01}$'            , 'events'),
    ('sv_cos'         , np.linspace( 0,  1, 30 + 1) , r'$\cos\alpha$'        , 'events'),
    ('sv_prob'        , np.linspace( 0,  1, 30 + 1) , 'SV probability'       , 'events'),
]

