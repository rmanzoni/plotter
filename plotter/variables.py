import numpy as np

class Variable(object):
    def __init__(self, var, bins, xlabel, ylabel, label=None, extra_label=None, extra_selection=None):
        self.var = var
        self.bins = bins
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.extra_label = extra_label
        self.extra_selection = extra_selection
        self.label = self.var if label is None else self.label
        if self.extra_label is not None:
            self.label = '_'.join([self.label, self.extra_label])
    
    
m12_bins_displaced_1 = np.linspace(0., 10., 10 + 1)    
m12_bins_displaced_2 = np.array([0., 1., 2., 3., 4., 5., 10]) 
m12_bins_displaced_3 = np.array([0., 1., 2., 3., 10])     

# m12_bins_displaced_1_alt = np.array([0., 1., 2., 3., 4., 5., 6., 8., 10.])    
# m12_bins_displaced_2_alt = np.array([0., 1., 2., 3., 4., 10]) 
# m12_bins_displaced_3_alt = np.array([0., 1., 2., 3., 10])     

m12_bins_displaced_1_alt = np.array([0., 1., 2., 3., 4., 10.])    
m12_bins_displaced_2_alt = np.array([0., 1., 2., 3., 4., 10.]) 
m12_bins_displaced_3_alt = np.array([0., 1., 2., 3., 10.])     

m12_bins_displaced_1_coarse = np.linspace(0., 10., 5 + 1)    
m12_bins_displaced_2_coarse = np.array([0., 2., 4., 10]) 
m12_bins_displaced_3_coarse = np.array([0., 2., 4., 10])     
    
# variables
variables = [
#     Variable('_norm_', np.linspace(0.,  1., 2 + 1), 'normalisation', 'events'),
    
    Variable('l0_pt', np.linspace(25.,100.,12 + 1), 'l_{1} p_{T} (GeV)', 'events'),
    Variable('l1_pt', np.linspace( 5., 50.,12 + 1), 'l_{2} p_{T} (GeV)', 'events'),
    Variable('l2_pt', np.linspace( 5., 30.,12 + 1), 'l_{3} p_{T} (GeV)', 'events'),

    Variable('l0_eta', np.linspace(-2.5, 2.5, 10 + 1), 'l_{1} \eta', 'events'),
    Variable('l1_eta', np.linspace(-2.5, 2.5, 10 + 1), 'l_{2} \eta', 'events'),
    Variable('l2_eta', np.linspace(-2.5, 2.5, 10 + 1), 'l_{3} \eta', 'events'),

    Variable('l0_phi', np.linspace(-3.15, 3.15, 10 + 1), 'l_{1} \phi', 'events'),
    Variable('l1_phi', np.linspace(-3.15, 3.15, 10 + 1), 'l_{2} \phi', 'events'),
    Variable('l2_phi', np.linspace(-3.15, 3.15, 10 + 1), 'l_{3} \phi', 'events'),

    Variable('l0_pt', np.linspace(25.,100.,12 + 1), 'l_{1} p_{T} (GeV)', 'events', extra_selection='hnl_2d_disp<=0.5'                  , extra_label='lxy_lt_0p5'    ),
    Variable('l0_pt', np.linspace(25.,100.,12 + 1), 'l_{1} p_{T} (GeV)', 'events', extra_selection='hnl_2d_disp>0.5 & hnl_2d_disp<=2.0', extra_label='lxy_0p5_to_2p0'),
    Variable('l0_pt', np.linspace(25.,100.,10 + 1), 'l_{1} p_{T} (GeV)', 'events', extra_selection='hnl_2d_disp>2.0'                   , extra_label='lxy_mt_2p0'    ),

    Variable('l1_pt', np.linspace( 5., 50.,12 + 1), 'l_{2} p_{T} (GeV)', 'events', extra_selection='hnl_2d_disp<=0.5'                  , extra_label='lxy_lt_0p5'    ),
    Variable('l1_pt', np.linspace( 5., 50.,12 + 1), 'l_{2} p_{T} (GeV)', 'events', extra_selection='hnl_2d_disp>0.5 & hnl_2d_disp<=2.0', extra_label='lxy_0p5_to_2p0'),
    Variable('l1_pt', np.linspace( 5., 50.,10 + 1), 'l_{2} p_{T} (GeV)', 'events', extra_selection='hnl_2d_disp>2.0'                   , extra_label='lxy_mt_2p0'    ),

    Variable('l2_pt', np.linspace( 5., 30.,12 + 1), 'l_{3} p_{T} (GeV)', 'events', extra_selection='hnl_2d_disp<=0.5'                  , extra_label='lxy_lt_0p5'    ),
    Variable('l2_pt', np.linspace( 5., 30.,12 + 1), 'l_{3} p_{T} (GeV)', 'events', extra_selection='hnl_2d_disp>0.5 & hnl_2d_disp<=2.0', extra_label='lxy_0p5_to_2p0'),
    Variable('l2_pt', np.linspace( 5., 30.,10 + 1), 'l_{3} p_{T} (GeV)', 'events', extra_selection='hnl_2d_disp>2.0'                   , extra_label='lxy_mt_2p0'    ),

    Variable('hnl_m_12', m12_bins_displaced_1, 'm_{23} (GeV)', 'events'),

#     Variable('hnl_m_12', m12_bins_displaced_1, 'm_{23} (GeV)', 'events', extra_selection='hnl_2d_disp<=0.5'                  , extra_label='lxy_lt_0p5'    ),
#     Variable('hnl_m_12', m12_bins_displaced_2, 'm_{23} (GeV)', 'events', extra_selection='hnl_2d_disp>0.5 & hnl_2d_disp<=2.0', extra_label='lxy_0p5_to_2p0'),
#     Variable('hnl_m_12', m12_bins_displaced_3, 'm_{23} (GeV)', 'events', extra_selection='hnl_2d_disp>2.0'                   , extra_label='lxy_mt_2p0'    ),

    Variable('hnl_m_12', m12_bins_displaced_1_alt, 'm_{23} (GeV)', 'events', extra_selection='hnl_2d_disp<=0.5'                  , extra_label='lxy_lt_0p5'    ),
    Variable('hnl_m_12', m12_bins_displaced_2_alt, 'm_{23} (GeV)', 'events', extra_selection='hnl_2d_disp>0.5 & hnl_2d_disp<=2.0', extra_label='lxy_0p5_to_2p0'),
    Variable('hnl_m_12', m12_bins_displaced_3_alt, 'm_{23} (GeV)', 'events', extra_selection='hnl_2d_disp>2.0'                   , extra_label='lxy_mt_2p0'    ),

    Variable('hnl_m_12', m12_bins_displaced_1_coarse, 'm_{23} (GeV)', 'events', extra_selection='hnl_2d_disp<=0.5'                  , extra_label='lxy_lt_0p5_coarse'    ),
    Variable('hnl_m_12', m12_bins_displaced_2_coarse, 'm_{23} (GeV)', 'events', extra_selection='hnl_2d_disp>0.5 & hnl_2d_disp<=2.0', extra_label='lxy_0p5_to_2p0_coarse'),
    Variable('hnl_m_12', m12_bins_displaced_3_coarse, 'm_{23} (GeV)', 'events', extra_selection='hnl_2d_disp>2.0'                   , extra_label='lxy_mt_2p0_coarse'    ),

#     Variable('hnl_m_12', np.linspace(0., 12., 12 + 1), 'm_{23} (GeV)', 'events', extra_selection='hnl_2d_disp<=0.5'                  , extra_label='lxy_lt_0p5'    ),
#     Variable('hnl_m_12', np.linspace(0., 12., 12 + 1), 'm_{23} (GeV)', 'events', extra_selection='hnl_2d_disp>0.5 & hnl_2d_disp<=2.0', extra_label='lxy_0p5_to_2p0'),
#     Variable('hnl_m_12', np.linspace(0., 12., 12 + 1), 'm_{23} (GeV)', 'events', extra_selection='hnl_2d_disp>2.0'                   , extra_label='lxy_mt_2p0'    ),

#     Variable('hnl_m_12', np.linspace(0., 12., 12 + 1), 'm_{23} (GeV)', 'events', extra_selection='hnl_2d_disp<=2.0'                  , extra_label='lxy_lt_2p0'    ),
#     Variable('hnl_m_12', np.linspace(0., 12., 12 + 1), 'm_{23} (GeV)', 'events', extra_selection='hnl_2d_disp>2.0 & hnl_2d_disp<=5.0', extra_label='lxy_0p5_to_2p0'),
#     Variable('hnl_m_12', np.linspace(0., 12., 12 + 1), 'm_{23} (GeV)', 'events', extra_selection='hnl_2d_disp>5.0'                   , extra_label='lxy_mt_5p0'    ),

    Variable('hnl_2d_disp'    , np.linspace( 0  ,  30, 25 + 1) , 'L_{xy} (cm)'       , 'events'),
    Variable('hnl_2d_disp_sig', np.linspace( 0  , 200, 25 + 1) , 'L_{xy}/\sigma_{xy}', 'events'),
    Variable('hnl_2d_disp_sig', np.linspace( 0  ,1000, 25 + 1) , 'L_{xy}/\sigma_{xy}', 'events', extra_label='hnl_2d_disp_sig_extended'),
    Variable('nbj'            , np.linspace( 0  ,   5,  5 + 1) , '#b-jet'            , 'events'),
    Variable('hnl_w_vis_m'    , np.linspace( 0  , 150, 40 + 1) , 'm_{3l}'            , 'events'),
    Variable('sv_cos'         , np.linspace( 0.9,   1, 30 + 1) , '\cos\alpha'        , 'events'),
    Variable('sv_prob'        , np.linspace( 0  ,   1, 30 + 1) , 'SV probability'    , 'events'),
    Variable('sv_prob'        , np.linspace( 0  , 0.1, 50 + 1) , 'SV probability'    , 'events', extra_label='sv_prob_fine'),
    Variable('sv_prob'        , np.linspace( 0  ,   1, 20 + 1) , 'SV probability'    , 'events', extra_label='sv_prob_coarse'),

    Variable('hnl_q_01'       , np.linspace(-3  ,  3,  3 + 1) , 'q_{12}'            , 'events'),
    Variable('hnl_q_02'       , np.linspace(-3  ,  3,  3 + 1) , 'q_{13}'            , 'events'),
    Variable('hnl_q_12'       , np.linspace(-3  ,  3,  3 + 1) , 'q_{23}'            , 'events'),

    Variable('hnl_dr_01', np.linspace( 0  , 6, 25 + 1), '\DeltaR_{12}', 'events'),
    Variable('hnl_dr_02', np.linspace( 0  , 6, 25 + 1), '\DeltaR_{13}', 'events'),
    Variable('hnl_dr_12', np.linspace( 0  , 1, 25 + 1), '\DeltaR_{23}', 'events'),

    Variable('n_vtx' , np.linspace( 0  , 70, 35 + 1), '#PV', 'events'),
    Variable('pfmet_pt', np.linspace( 0  ,100, 12 + 1), 'PF E_{T}^{miss}', 'events'),

    Variable('hnl_pt_12', np.linspace( 10, 60, 20 + 1), 'p_{T}^{23} (GeV)', 'events'),

    Variable('log_abs_l0_dxy', np.linspace( -4, 2, 12 + 1), 'log_{10}( l_{1} |d_{xy}|) (cm)', 'events'),
    Variable('log_abs_l0_dz' , np.linspace( -4, 2, 12 + 1), 'log_{10}( l_{1} |d_{z}|) (cm)', 'events'),
    Variable('log_abs_l1_dxy', np.linspace( -4, 2, 12 + 1), 'log_{10}( l_{2} |d_{xy}|) (cm)', 'events'),
    Variable('log_abs_l1_dz' , np.linspace( -4, 2, 12 + 1), 'log_{10}( l_{2} |d_{z}|) (cm)', 'events'),
    Variable('log_abs_l2_dxy', np.linspace( -4, 2, 12 + 1), 'log_{10}( l_{3} |d_{xy}|) (cm)', 'events'),
    Variable('log_abs_l2_dz' , np.linspace( -4, 2, 12 + 1), 'log_{10}( l_{3} |d_{z}|) (cm)', 'events'),

    Variable('log_l0_dxy_sig', np.linspace( -4, 2, 12 + 1), 'log_{10}( l_{1} d_{xy}^{sig})', 'events'),
    Variable('log_l0_dz_sig' , np.linspace( -4, 2, 12 + 1), 'log_{10}( l_{1} d_{z}^{sig})' , 'events'),
    Variable('log_l1_dxy_sig', np.linspace( -4, 2, 12 + 1), 'log_{10}( l_{2} d_{xy}^{sig})', 'events'),
    Variable('log_l1_dz_sig' , np.linspace( -4, 2, 12 + 1), 'log_{10}( l_{2} d_{z}^{sig})' , 'events'),
    Variable('log_l2_dxy_sig', np.linspace( -4, 2, 12 + 1), 'log_{10}( l_{3} d_{xy}^{sig})', 'events'),
    Variable('log_l2_dz_sig' , np.linspace( -4, 2, 12 + 1), 'log_{10}( l_{3} d_{z}^{sig})' , 'events'),

#     Variable('fr'             , np.linspace( 0  ,  1, 30 + 1) , 'fake rate'         , 'events'),
]


