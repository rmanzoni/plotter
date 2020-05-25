from re import findall
import numpy as np
import pandas as pd
from root_pandas import read_root
from collections import OrderedDict

global channel_dict
channel_dict = OrderedDict()
channel_dict['mmm'] = 1
channel_dict['mem'] = 2
channel_dict['eee'] = 3
channel_dict['eem'] = 4

global signal_weights
signal_weights = [
    '1.0em01',
    '2.0em01',
    '3.0em01',
    '4.0em01',
    '5.0em01',
    '6.0em01',
    '7.0em01',
    '8.0em01',
    '9.0em01',
    '1.0em02',
    '2.0em02',
    '3.0em02',
    '4.0em02',
    '5.0em02',
    '6.0em02',
    '7.0em02',
    '8.0em02',
    '9.0em02',
    '1.0em03',
    '2.0em03',
    '3.0em03',
    '4.0em03',
    '5.0em03',
    '6.0em03',
    '7.0em03',
    '8.0em03',
    '9.0em03',
    '1.0em04',
    '2.0em04',
    '3.0em04',
    '4.0em04',
    '5.0em04',
    '6.0em04',
    '7.0em04',
    '8.0em04',
    '9.0em04',
    '1.0em05',
    '2.0em05',
    '3.0em05',
    '4.0em05',
    '5.0em05',
    '6.0em05',
    '7.0em05',
    '8.0em05',
    '9.0em05',
    '1.0em06',
    '2.0em06',
    '3.0em06',
    '4.0em06',
    '5.0em06',
    '6.0em06',
    '7.0em06',
    '8.0em06',
    '9.0em06',
    '1.0em07',
    '2.0em07',
    '3.0em07',
    '4.0em07',
    '5.0em07',
    '6.0em07',
    '7.0em07',
    '8.0em07',
    '9.0em07',
    '1.0em08',
    '2.0em08',
    '3.0em08',
    '4.0em08',
    '5.0em08',
    '6.0em08',
    '7.0em08',
    '8.0em08',
    '9.0em08',
    '1.0em09',
    '2.0em09',
    '3.0em09',
    '4.0em09',
    '5.0em09',
    '6.0em09',
    '7.0em09',
    '8.0em09',
    '9.0em09',
    '1.0em10',
    '2.0em10',
    '3.0em10',
    '4.0em10',
    '5.0em10',
    '6.0em10',
    '7.0em10',
    '8.0em10',
    '9.0em10',
]

@np.vectorize
def ptcone(pt, iso, iso_cut):
    if iso < iso_cut:
        return pt
    else:
        return (1.+iso-iso_cut)*pt

class Sample(object):
    def __init__(self, 
                 name, 
                 channel,
                 label,
                 selection,
                 datacard_name,
                 colour,
                 position_in_stack, 
                 basedir, 
                 postfix, 
                 isdata, 
                 ismc, 
                 issignal, 
                 weight,
                 xs,
                 toplot=True,
                 extra_signal_weights=[],
                 is_generator=False):
        self.name                 = name
        self.channel              = channel
        self.label                = label   
        self.selection            = selection         
        self.datacard_name        = datacard_name ; print('loading', self.name, '\t', self.datacard_name, end = '')        
        self.colour               = colour           
        self.position_in_stack    = position_in_stack
        self.basedir              = basedir          
        self.postfix              = postfix          
        self.isdata               = isdata           
        self.ismc                 = ismc             
        self.issignal             = issignal         
        self.weight               = weight           
        self.xs                   = xs        
        self.nevents              = 1.
        self.file                 = '/'.join([basedir, self.name, postfix])       
        self.toplot               = toplot 
        self.extra_signal_weights = extra_signal_weights
        self.is_generator         = is_generator  
        
        if not self.isdata:
            nevents_file = '/'.join([basedir, self.name, 'SkimAnalyzerCount/SkimReport.txt'])
            with open(nevents_file) as ff:
                lines = ff.readlines()
                for line in lines:
                    if 'Sum Norm Weights' in line:
                        self.nevents = float(findall(r'\d+', lines[2])[0])
                        break
        tree_file = '/'.join([self.basedir, self.name, self.postfix])

#         print('\n\n=========> tree file\n', tree_file)
#         print('\n\n=========> selection\n', self.selection, '\n\n')
                
        self.df = read_root( tree_file, 'tree', where=self.selection, warn_missing_tree=True ); print('\tselected events', len(self.df))
        
        # self awareness...
        self.df['channel'] = channel_dict[self.channel]
        # FIXME! extra features should be computed here once for all not in trainer or plotter
        self.df['abs_l0_eta'         ] = np.abs(self.df.l0_eta)
        self.df['abs_l1_eta'         ] = np.abs(self.df.l1_eta)
        self.df['abs_l2_eta'         ] = np.abs(self.df.l2_eta)
        self.df['abs_l0_pdgid'       ] = np.abs(self.df.l0_pdgid)
        self.df['abs_l1_pdgid'       ] = np.abs(self.df.l1_pdgid)
        self.df['abs_l2_pdgid'       ] = np.abs(self.df.l2_pdgid)
        self.df['log_abs_l0_dxy'     ] = np.log10(np.abs(self.df.l0_dxy))
        self.df['log_abs_l0_dz'      ] = np.log10(np.abs(self.df.l0_dz ))
        self.df['log_abs_l1_dxy'     ] = np.log10(np.abs(self.df.l1_dxy))
        self.df['log_abs_l1_dz'      ] = np.log10(np.abs(self.df.l1_dz ))
        self.df['log_abs_l2_dxy'     ] = np.log10(np.abs(self.df.l2_dxy))
        self.df['log_abs_l2_dz'      ] = np.log10(np.abs(self.df.l2_dz ))
        self.df['abs_q_02'           ] = np.abs(self.df.hnl_q_02)
        self.df['abs_q_01'           ] = np.abs(self.df.hnl_q_01)
        self.df['log_l0_dxy_sig'     ] = np.log10(self.df.l0_dxy_error / np.abs(self.df.l0_dxy ))
        self.df['log_l1_dxy_sig'     ] = np.log10(self.df.l1_dxy_error / np.abs(self.df.l1_dxy ))
        self.df['log_l2_dxy_sig'     ] = np.log10(self.df.l2_dxy_error / np.abs(self.df.l2_dxy ))
        self.df['log_l0_dz_sig'      ] = np.log10(self.df.l0_dz_error / np.abs(self.df.l0_dz ))
        self.df['log_l1_dz_sig'      ] = np.log10(self.df.l1_dz_error / np.abs(self.df.l1_dz ))
        self.df['log_l2_dz_sig'      ] = np.log10(self.df.l2_dz_error / np.abs(self.df.l2_dz ))

        self.df['log_hnl_2d_disp'    ] = np.log10(self.df.hnl_2d_disp)
        self.df['log_hnl_2d_disp_sig'] = np.log10(self.df.hnl_2d_disp_sig)
        
        self.df['l0_ptcone'      ] = ptcone(self.df.l0_pt, self.df.l0_reliso_rho_03, 0.1) if len(self.df) else np.nan
        self.df['l1_ptcone'      ] = ptcone(self.df.l1_pt, self.df.l1_reliso_rho_03, 0.2) if len(self.df) else np.nan
        self.df['l2_ptcone'      ] = ptcone(self.df.l2_pt, self.df.l2_reliso_rho_03, 0.2) if len(self.df) else np.nan
        
        # defined à la Martina
        self.df['hnl_2d_disp_sig_alt'] = self.df.hnl_2d_disp**2 / np.sqrt(self.df.sv_covxx * self.df.sv_x**2 + self.df.sv_covyy * self.df.sv_y**2)
        
        self.df['_norm_'        ] = 0.
                
        # scale to 1/pb 
        self.lumi_scaling = 1. if self.isdata else (self.xs / self.nevents)
 


def get_data_samples(channel, basedir, postfix, selection):
    if   channel [0] == 'm': lep = 'mu'
    elif channel [0] == 'e': lep = 'ele'
    assert lep == 'ele' or lep == 'mu', 'Lepton flavor error'
    data = [
        Sample('Single_{lep}_2018A'.format(lep=lep), channel, '2018A', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2018B'.format(lep=lep), channel, '2018B', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2018C'.format(lep=lep), channel, '2018C', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2018D'.format(lep=lep), channel, '2018D', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
    ]
    return data

def get_mc_samples(channel, basedir, postfix, selection):
    mc = [
        Sample('DYJetsToLL_M50_ext', channel,  r'DY$\to\ell\ell$', selection, 'DY'   , 'gold'     ,10, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,  6077.22),
        Sample('TTJets_ext'        , channel,  r'$t\bar{t}$'     , selection, 'TT'   , 'slateblue', 0, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,   831.76),
        Sample('WW'                , channel,  'WW'              , selection, 'WW'   , 'blue'     , 5, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    75.88),
        Sample('WZ'                , channel,  'WZ'              , selection, 'WZ'   , 'blue'     , 5, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    27.6 ),
        Sample('ZZ'                , channel,  'ZZ'              , selection, 'ZZ'   , 'blue'     , 5, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    12.14),
        Sample('DYJetsToLL_M5to50' , channel,  r'DY$\to\ell\ell$', selection, 'DYLM' , 'gold'     ,10, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1., 81880.0 ),
        Sample('ST_tW_inc'         , channel,  r'single$-t$'     , selection, 'TtW'  , 'slateblue', 1, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    35.85),
        Sample('ST_tch_inc'        , channel,  r'single$-t$'     , selection, 'Ttch' , 'slateblue', 1, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,   136.02),
        Sample('STbar_tW_inc'      , channel,  r'single$-t$'     , selection, 'TbtW' , 'slateblue', 1, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    35.85),
        Sample('STbar_tch_inc'     , channel,  r'single$-t$'     , selection, 'Tbtch', 'slateblue', 1, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    80.95),
    ]   
    return mc         

def get_signal_samples(channel, basedir, postfix, selection, mini=False):
    assert channel[0] == 'e' or channel[0] == 'm', 'Lepton flavor error'
    if channel [0] == 'm':
        if mini:
            signal = [ 
                ########## M = 2
                Sample('HN3L_M_2_V_0p0110905365064_mu_massiveAndCKM_LO'   , channel, '#splitline{m=2 GeV |V|^{2}=1.2 10^{-4}}{Majorana}' , selection, 'hnl_m_2_v2_1p2Em04_majorana' , 'forestgreen',10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  0.5278   , toplot=True ),
                ########## M = 5
                Sample('HN3L_M_5_V_0p00145602197786_mu_massiveAndCKM_LO'  , channel, '#splitline{m=5 GeV |V|^{2}=2.1 10^{-6}}{Majorana}' , selection, 'hnl_m_5_v2_2p1Em06_majorana' , 'chocolate'  ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  0.008434 , toplot=True ),
                ########## M = 10
                Sample('HN3L_M_10_V_0p001_mu_massiveAndCKM_LO'            , channel, '#splitline{m=10 GeV |V|^{2}=1.0 10^{-6}}{Majorana}', selection, 'hnl_m_10_v2_1p0Em06_majorana', 'teal'       ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  0.004121 , toplot=True ),
            ]
        else:
            signal = [ 
                Sample('HN3L_M_1_V_0p0949736805647_mu_massiveAndCKM_LO'   , channel, '#splitline{m=1 GeV, |V|^{2}=9.0 10^{-3}}{Majorana}' , selection, 'hnl_m_1_v2_9p0Em03_majorana' , 'darkorange' ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  38.67    , toplot=False, is_generator=True),
                Sample('HN3L_M_2_V_0p0110905365064_mu_massiveAndCKM_LO'   , channel, '#splitline{m=2 GeV, |V|^{2}=1.2 10^{-4}}{Majorana}' , selection, 'hnl_m_2_v2_1p2Em04_majorana' , 'forestgreen',10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  0.5278   , toplot=True , is_generator=True),
                Sample('HN3L_M_2_V_0p0248394846967_mu_massiveAndCKM_LO'   , channel, '#splitline{m=2 GeV, |V|^{2}=6.2 10^{-4}}{Majorana}' , selection, 'hnl_m_2_v2_6p2Em04_majorana' , 'forestgreen',10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  2.647    , toplot=False),
                Sample('HN3L_M_3_V_0p00707813534767_mu_massiveAndCKM_LO'  , channel, '#splitline{m=3 GeV, |V|^{2}=5.0 10^{-5}}{Majorana}' , selection, 'hnl_m_3_v2_5p0Em05_majorana' , 'firebrick'  ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  0.2014   , toplot=False, is_generator=True),
                Sample('HN3L_M_4_V_0p00290516780927_mu_massiveAndCKM_LO'  , channel, '#splitline{m=4 GeV, |V|^{2}=8.4 10^{-6}}{Majorana}' , selection, 'hnl_m_4_v2_8p4Em06_majorana' , 'indigo'     ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  0.0335   , toplot=False, is_generator=True),
                Sample('HN3L_M_5_V_0p000316227766017_mu_massiveAndCKM_LO' , channel, '#splitline{m=5 GeV, |V|^{2}=1.0 10^{-7}}{Majorana}' , selection, 'hnl_m_5_v2_1p0Em07_majorana' , 'chocolate'  ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  0.0003981, toplot=False),
                Sample('HN3L_M_5_V_0p000547722557505_mu_massiveAndCKM_LO' , channel, '#splitline{m=5 GeV, |V|^{2}=3.0 10^{-7}}{Majorana}' , selection, 'hnl_m_5_v2_3p0Em07_majorana' , 'chocolate'  ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  0.001194 , toplot=False),
                Sample('HN3L_M_5_V_0p00145602197786_mu_massiveAndCKM_LO'  , channel, '#splitline{m=5 GeV, |V|^{2}=2.1 10^{-6}}{Majorana}' , selection, 'hnl_m_5_v2_2p1Em06_majorana' , 'chocolate'  ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  0.008434 , toplot=True , is_generator=True),
                Sample('HN3L_M_6_V_0p00202484567313_mu_massiveAndCKM_LO'  , channel, '#splitline{m=6 GeV, |V|^{2}=4.1 10^{-6}}{Majorana}' , selection, 'hnl_m_6_v2_4p1Em06_majorana' , 'olive'      ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  0.01655  , toplot=False, is_generator=True),
                Sample('HN3L_M_8_V_0p000547722557505_mu_massiveAndCKM_LO' , channel, '#splitline{m=8 GeV, |V|^{2}=3.0 10^{-7}}{Majorana}' , selection, 'hnl_m_8_v2_3p0Em07_majorana' , 'darkgray'   ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  0.00123  , toplot=False, is_generator=True),
                Sample('HN3L_M_10_V_0p000547722557505_mu_massiveAndCKM_LO', channel, '#splitline{m=10 GeV, |V|^{2}=3.0 10^{-7}}{Majorana}', selection, 'hnl_m_10_v2_3p0Em07_majorana', 'teal'       ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  0.001237 , toplot=False),
                Sample('HN3L_M_10_V_0p000756967634711_mu_massiveAndCKM_LO', channel, '#splitline{m=10 GeV, |V|^{2}=5.7 10^{-7}}{Majorana}', selection, 'hnl_m_10_v2_5p7Em07_majorana', 'teal'       ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  0.002362 , toplot=False),
                Sample('HN3L_M_10_V_0p001_mu_massiveAndCKM_LO'            , channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-6}}{Majorana}', selection, 'hnl_m_10_v2_1p0Em06_majorana', 'teal'       ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  0.004121 , toplot=True , is_generator=True),
            ]
                
    elif channel [0] == 'e': 
        if mini:
            signal = [
                Sample('HN3L_M_2_V_0p0248394846967_e_massiveAndCKM_LO'   , channel, '#splitline{m=2 GeV, |V|^{2}=6.2 10^{-4}}{Majorana}' , selection, 'hnl_m_2_v2_6p2Em04_majorana' , 'forestgreen',10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  2.648    , toplot=True , is_generator=False),
                Sample('HN3L_M_8_V_0p00151327459504_e_massiveAndCKM_LO'  , channel, '#splitline{m=8 GeV, |V|^{2}=2.3 10^{-6}}{Majorana}' , selection, 'hnl_m_8_v2_2p3Em06_majorana' , 'darkgray'   ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  9.383e-03, toplot=True , is_generator=False),
             ]
        else:
            signal = [
                Sample('HN3L_M_1_V_0p212367605816_e_massiveAndCKM_LO'    , channel, '#splitline{m=1 GeV, |V|^{2}=4.5 10^{-2}}{Majorana}' , selection, 'hnl_m_1_v2_4p5Em02_majorana' , 'darkorange' ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  191.1    , toplot=False, is_generator=True),
                Sample('HN3L_M_2_V_0p0248394846967_e_massiveAndCKM_LO'   , channel, '#splitline{m=2 GeV, |V|^{2}=6.2 10^{-4}}{Majorana}' , selection, 'hnl_m_2_v2_6p2Em04_majorana' , 'forestgreen',10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  2.648    , toplot=True , is_generator=True),
                Sample('HN3L_M_3_V_0p00707813534767_e_massiveAndCKM_LO'  , channel, '#splitline{m=3 GeV, |V|^{2}=5.1 10^{-5}}{Majorana}' , selection, 'hnl_m_3_v2_5p1Em05_majorana' , 'firebrick'  ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  0.2022   , toplot=False, is_generator=True),
                Sample('HN3L_M_4_V_0p00290516780927_e_massiveAndCKM_LO'  , channel, '#splitline{m=4 GeV, |V|^{2}=8.4 10^{-6}}{Majorana}' , selection, 'hnl_m_4_v2_8p4Em06_majorana' , 'indigo'     ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  3.365e-02, toplot=False, is_generator=True),
                Sample('HN3L_M_5_V_0p00145602197786_e_massiveAndCKM_LO'  , channel, '#splitline{m=5 GeV, |V|^{2}=2.1 10^{-6}}{Majorana}' , selection, 'hnl_m_5_v2_2p1Em06_majorana' , 'chocolate'  ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  8.479e-03, toplot=False, is_generator=True),
                Sample('HN3L_M_6_V_0p00202484567313_e_massiveAndCKM_LO'  , channel, '#splitline{m=6 GeV, |V|^{2}=4.1 10^{-6}}{Majorana}' , selection, 'hnl_m_6_v2_4p1Em06_majorana' , 'olive'      ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  1.655e-02, toplot=False, is_generator=True),
#                 Sample('HN3L_M_7_V_0p0316227766017_e_massiveAndCKM_LO'   , channel, '#splitline{m=7 GeV, |V|^{2}=1.0 10^{-4}}{Majorana}' , selection, 'hnl_m_7_v2_1p0Em04_majorana' , 'darkgray'   ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  4.088    , toplot=False, is_generator=True),
                Sample('HN3L_M_8_V_0p00151327459504_e_massiveAndCKM_LO'  , channel, '#splitline{m=8 GeV, |V|^{2}=2.3 10^{-6}}{Majorana}' , selection, 'hnl_m_8_v2_2p3Em06_majorana' , 'darkgray'   ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  9.383e-03, toplot=True , is_generator=True),
                Sample('HN3L_M_10_V_0p000756967634711_e_massiveAndCKM_LO', channel, '#splitline{m=10 GeV, |V|^{2}=5.7 10^{-7}}{Majorana}', selection, 'hnl_m_10_v2_5p7Em07_majorana', 'teal'       ,10, '/'.join([basedir, 'sig']), postfix, False, True, False, 1.,  2.366e-03, toplot=False, is_generator=True),
             ]
        
    # generate reweighted samples
    generators = [isample for isample in signal if isample.is_generator]
    for igenerator in generators:
        for iw in signal_weights:
            v2_val = iw.split('em')[0]
            v2_exp = iw.split('em')[1]
            mass = igenerator.name.split('_')[2]
            label = '#splitline{m=%s GeV, |V|^{2}=%s 10^{ -%d}}{Majorana}' %(mass, v2_val, int(v2_exp))
            # don't reweigh existing signals
            if label == igenerator.label:
                continue
            new_sample = Sample(
                name                 = igenerator.name                                                          , 
                channel              = igenerator.channel                                                       , 
                label                = label                                                                    , 
                selection            = igenerator.selection                                                     , 
                datacard_name        = 'hnl_m_%s_v2_%s_majorana'%(mass, iw.replace('.','p').replace('em', 'Em')), 
                colour               = igenerator.colour                                                        ,
                position_in_stack    = igenerator.position_in_stack                                             , 
                basedir              = igenerator.basedir                                                       , 
                postfix              = igenerator.postfix                                                       , 
                isdata               = igenerator.isdata                                                        , 
                ismc                 = igenerator.ismc                                                          , 
                issignal             = igenerator.issignal                                                      , 
                weight               = igenerator.weight                                                        ,  
                xs                   = igenerator.xs                                                            , 
                toplot               = False                                                                    , 
                extra_signal_weights = ['ctau_w_v2_%s'%iw, 'xs_w_v2_%s'%iw]                                     ,
            )
            signal.append(new_sample)


    return signal

