from re import findall
import numpy as np
import pandas as pd
from root_pandas import read_root
from collections import OrderedDict
from plotter.objects.sample import Sample, signal_weights

def get_data_samples(channel, basedir, postfix, selection):
    if   channel [0] == 'm': lep = 'mu'
    elif channel [0] == 'e': lep = 'ele'
    assert lep == 'ele' or lep == 'mu', 'Lepton flavor error'
    data = [
        Sample('Single_{lep}_2016C'.format(lep=lep), ['Single_{lep}_2016C'.format(lep=lep)], channel, '2016C', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2016D'.format(lep=lep), ['Single_{lep}_2016D'.format(lep=lep)], channel, '2016D', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2016E'.format(lep=lep), ['Single_{lep}_2016E'.format(lep=lep)], channel, '2016E', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2016F'.format(lep=lep), ['Single_{lep}_2016F'.format(lep=lep)], channel, '2016F', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2016G'.format(lep=lep), ['Single_{lep}_2016G'.format(lep=lep)], channel, '2016G', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2016H'.format(lep=lep), ['Single_{lep}_2016H'.format(lep=lep)], channel, '2016H', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
    ]
    return data

def get_mc_samples(channel, basedir, postfix, selection):
    mc = [
        Sample('DYJetsToLL_M50'    , ['DYJetsToLL_M50'                             ], channel,  r'DY$\to\ell\ell$', selection, 'DY_lo'    , 'gold'     ,10, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,  6077.22),
        Sample('TTJets'            , ['TTJets'                                     ], channel,  r'$t\bar{t}$'     , selection, 'TT'       , 'slateblue', 0, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,   831.76),
        Sample('WW'                , ['WW'                                         ], channel,  'WW'              , selection, 'WW'       , 'blue'     , 5, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    75.88),
        Sample('WZ'                , ['WZ'                                         ], channel,  'WZ'              , selection, 'WZ'       , 'blue'     , 5, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    27.6 ),
        Sample('ZZ'                , ['ZZ'                                         ], channel,  'ZZ'              , selection, 'ZZ'       , 'blue'     , 5, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    12.14),
        Sample('DYJetsToLL_M10to50', ['DYJetsToLL_M10to50','DYJetsToLL_M10to50_ext'], channel,  r'DY$\to\ell\ell$', selection, 'DY_lo_low', 'gold'     ,10, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1., 18610.0 ),
        Sample('WJetsToLNu'        , ['WJetsToLNu','WJetsToLNu_ext'                ], channel,  r'W$\to\ell\nu$'  , selection, 'W'        , 'brown'    , 2, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1., 59850.0 ),
        Sample('ST_tW_inc'         , ['ST_tW_inc'                                  ], channel,  r'single$-t$'     , selection, 'TtW'      , 'slateblue', 1, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    35.85),
        Sample('ST_tch_inc'        , ['ST_tch_inc'                                 ], channel,  r'single$-t$'     , selection, 'Ttch'     , 'slateblue', 1, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,   136.02),
        Sample('STbar_tW_inc'      , ['STbar_tW_inc'                               ], channel,  r'single$-t$'     , selection, 'TbtW'     , 'slateblue', 1, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    35.85),
        Sample('STbar_tch_inc'     , ['STbar_tch_inc'                              ], channel,  r'single$-t$'     , selection, 'Tbtch'    , 'slateblue', 1, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    80.95),
    ]   
    return mc         


## RM CHANGE SIGNALS!!

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
    existing_datacard_names= [isample.datacard_name for isample in signal]
    for igenerator in generators:
        for iw in signal_weights:
            v2_val = iw.split('em')[0]
            v2_exp = iw.split('em')[1]
            mass = igenerator.name.split('_')[2]
            label = '#splitline{m=%s GeV, |V|^{2}=%s 10^{-%d}}{Majorana}' %(mass, v2_val, int(v2_exp))
            datacard_name = 'hnl_m_%s_v2_%s_majorana'%(mass, iw.replace('.','p').replace('em', 'Em'))
            # don't reweigh existing signals
            if datacard_name in existing_datacard_names:
                continue
                
            new_sample = dc(igenerator)
            new_sample.label = label
            new_sample.datacard_name        = datacard_name
            new_sample.toplot               = False
            new_sample.extra_signal_weights = ['ctau_w_v2_%s'%iw, 'xs_w_v2_%s'%iw]
 
            print('generated reweighed signal', label, 'from', igenerator.name)           
            signal.append(new_sample)

    return signal

