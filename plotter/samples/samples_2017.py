from re import findall
import numpy as np
import pandas as pd
import gc # free mem up
from os import environ as env
from root_pandas import read_root
from collections import OrderedDict
from plotter.objects.sample import Sample, signal_weights_dict, signal_weights, ranges
from copy import copy

def get_data_samples(channel, basedir, postfix, selection):
    if   channel [0] == 'm': lep = 'mu'
    elif channel [0] == 'e': lep = 'ele'
    assert lep == 'ele' or lep == 'mu', 'Lepton flavor error'
    data = [
        Sample('Single_{lep}_2017B'.format(lep=lep), ['Single_{lep}_2017B'.format(lep=lep)], channel, '2017B', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2017C'.format(lep=lep), ['Single_{lep}_2017C'.format(lep=lep)], channel, '2017C', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2017D'.format(lep=lep), ['Single_{lep}_2017D'.format(lep=lep)], channel, '2017D', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2017E'.format(lep=lep), ['Single_{lep}_2017E'.format(lep=lep)], channel, '2017E', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2017F'.format(lep=lep), ['Single_{lep}_2017F'.format(lep=lep)], channel, '2017F', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
    ]
    return data

def get_mc_samples(channel, basedir, postfix, selection):
    mc = [
#         Sample('DYJetsToLL_M50'         , ['DYJetsToLL_M50'], channel,  r'DY$\to\ell\ell$', selection, 'DY_lo'     , 'gold'     ,10, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,  6077.22),
#         Sample('DYJetsToLL_M50'         , ['DYJetsToLL_M50', 'DYJetsToLL_M50_ext'], channel,  r'DY$\to\ell\ell$', selection, 'DY_lo'     , 'gold'     ,10, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,  6077.22),
#         Sample('DYJetsToLL_M50'         , ['DYJetsToLL_M50_fxfx', 'DYJetsToLL_M50_fxfx_ext'], channel,  r'DY$\to\ell\ell$', selection, 'DY_lo'     , 'gold'     ,10, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,  6077.22),
        Sample('DYJetsToLL_M50'         , ['DYJetsToLL_M50', 'DYJetsToLL_M50_ext', 'DYJetsToLL_M50_fxfx', 'DYJetsToLL_M50_fxfx_ext'], channel,  r'DY$\to\ell\ell$', selection, 'DY_lo'     , 'gold'     ,10, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,  6077.22),
        Sample('TTJets'                 , ['TTJets_fxfx'                                                                           ], channel,  r'$t\bar{t}$'     , selection, 'TT'        , 'slateblue', 0, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,   831.76),
        Sample('WW'                     , ['WW'                                                                                    ], channel,  'WW'              , selection, 'WW'        , 'blue'     , 5, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    75.88),
        Sample('WZ'                     , ['WZ'                                                                                    ], channel,  'WZ'              , selection, 'WZ'        , 'blue'     , 5, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    27.6 ),
        Sample('ZZ'                     , ['ZZ'                                                                                    ], channel,  'ZZ'              , selection, 'ZZ'        , 'blue'     , 5, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    12.14),
        Sample('WJetsToLNu'             , ['WJetsToLNu'                                                                            ], channel,  r'W$\to\ell\nu$'  , selection, 'W'         , 'brown'    , 2, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1., 59850.0 ),
        Sample('DYJetsToLL_M10to50'     , ['DYJetsToLL_M10to50'                                                                    ], channel,  r'DY$\to\ell\ell$', selection, 'DY_lo_low' , 'gold'     ,10, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1., 15810.0 ),
#         Sample('ST_tW_inc'              , ['ST_tW_inc'                                                                             ], channel,  r'single$-t$'     , selection, 'TtW'       , 'slateblue', 1, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    35.85),
#         Sample('ST_tch_inc'             , ['ST_tch_inc'                                                                            ], channel,  r'single$-t$'     , selection, 'Ttch'      , 'slateblue', 1, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,   136.02),
#         Sample('STbar_tW_inc'           , ['STbar_tW_inc'                                                                          ], channel,  r'single$-t$'     , selection, 'TbtW'      , 'slateblue', 1, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    35.85),
#         Sample('STbar_tch_inc'          , ['STbar_tch_inc'                                                                         ], channel,  r'single$-t$'     , selection, 'Tbtch'     , 'slateblue', 1, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    80.95),
    ]   
    return mc         

def get_signal_samples(channel, basedir, postfix, selection, mini=False):
    assert channel[0] == 'e' or channel[0] == 'm', 'Lepton flavor error'
    if channel [0] == 'm':
        if mini:
            signal = [ 
                Sample('HN3L_M_2_V_0p0110905365064_mu_massiveAndCKM_LO' , ['HN3L_M_2_V_0p0110905365064_mu_massiveAndCKM_LO' ], channel, '#splitline{m=2 GeV, |V|^{2}=1.2 10^{-4}}{Majorana}' , selection, 'hnl_m_2_v2_1p2Em04_majorana' , 'forestgreen',10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,  0.5278   , toplot=True ),
                Sample('HN3L_M_5_V_0p00145602197786_mu_massiveAndCKM_LO', ['HN3L_M_5_V_0p00145602197786_mu_massiveAndCKM_LO'], channel, '#splitline{m=5 GeV, |V|^{2}=2.1 10^{-6}}{Majorana}' , selection, 'hnl_m_5_v2_2p1Em06_majorana' , 'chocolate'  ,10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,  0.008434 , toplot=True ),
                Sample('HN3L_M_10_V_0p001_mu_massiveAndCKM_LO'          , ['HN3L_M_10_V_0p001_mu_massiveAndCKM_LO'          ], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-6}}{Majorana}', selection, 'hnl_m_10_v2_1p0Em06_majorana', 'teal'       ,10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,  0.004121 , toplot=True ),
            ]
        else:
            signal = [ 
                Sample('HN3L_M_1_V_0p022360679775_mu_massiveAndCKM_LO'    , ['HN3L_M_1_V_0p022360679775_mu_massiveAndCKM_LO'    ], channel, '#splitline{m=1 GeV, |V|^{2}=5.0 10^{-4}}{Majorana}' , selection, 'hnl_m_1_v2_5p0Em04_majorana' ,  'darkorange'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       2.107, toplot=False), 
                Sample('HN3L_M_1_V_0p0949736805647_mu_massiveAndCKM_LO'   , ['HN3L_M_1_V_0p0949736805647_mu_massiveAndCKM_LO'   ], channel, '#splitline{m=1 GeV, |V|^{2}=9.0 10^{-3}}{Majorana}' , selection, 'hnl_m_1_v2_9p0Em03_majorana' ,  'darkorange'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,        38.7, toplot=False), 
                Sample('HN3L_M_1_V_0p212367605816_mu_massiveAndCKM_LO'    , ['HN3L_M_1_V_0p212367605816_mu_massiveAndCKM_LO'    ], channel, '#splitline{m=1 GeV, |V|^{2}=4.5 10^{-2}}{Majorana}' , selection, 'hnl_m_1_v2_4p5Em02_majorana' ,  'darkorange'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       193.3, toplot=False), 
                Sample('HN3L_M_1_V_0p707106781187_mu_massiveAndCKM_LO'    , ['HN3L_M_1_V_0p707106781187_mu_massiveAndCKM_LO'    ], channel, '#splitline{m=1 GeV, |V|^{2}=5.0 10^{-1}}{Majorana}' , selection, 'hnl_m_1_v2_5p0Em01_majorana' ,  'darkorange'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      2113.0, toplot=False), 
                Sample('HN3L_M_2_V_0p0110905365064_mu_massiveAndCKM_LO'   , ['HN3L_M_2_V_0p0110905365064_mu_massiveAndCKM_LO'   ], channel, '#splitline{m=2 GeV, |V|^{2}=1.2 10^{-4}}{Majorana}' , selection, 'hnl_m_2_v2_1p2Em04_majorana' ,  'forestgreen'  , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       0.526, toplot=True ), 
                Sample('HN3L_M_2_V_0p0248394846967_mu_massiveAndCKM_LO'   , ['HN3L_M_2_V_0p0248394846967_mu_massiveAndCKM_LO'   ], channel, '#splitline{m=2 GeV, |V|^{2}=6.2 10^{-4}}{Majorana}' , selection, 'hnl_m_2_v2_6p2Em04_majorana' ,  'forestgreen'  , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       2.645, toplot=False), 
                Sample('HN3L_M_2_V_0p22360679775_mu_massiveAndCKM_LO'     , ['HN3L_M_2_V_0p22360679775_mu_massiveAndCKM_LO'     ], channel, '#splitline{m=2 GeV, |V|^{2}=5.0 10^{-2}}{Majorana}' , selection, 'hnl_m_2_v2_5p0Em02_majorana' ,  'forestgreen'  , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       211.1, toplot=False), 
                Sample('HN3L_M_2_V_0p707106781187_mu_massiveAndCKM_LO'    , ['HN3L_M_2_V_0p707106781187_mu_massiveAndCKM_LO'    ], channel, '#splitline{m=2 GeV, |V|^{2}=5.0 10^{-1}}{Majorana}' , selection, 'hnl_m_2_v2_5p0Em01_majorana' ,  'forestgreen'  , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      2108.0, toplot=False), 
                Sample('HN3L_M_3_V_0p00707813534767_mu_massiveAndCKM_LO'  , ['HN3L_M_3_V_0p00707813534767_mu_massiveAndCKM_LO'  ], channel, '#splitline{m=3 GeV, |V|^{2}=5.0 10^{-5}}{Majorana}' , selection, 'hnl_m_3_v2_5p0Em05_majorana' ,  'firebrick'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.2009, toplot=False), 
                Sample('HN3L_M_3_V_0p22360679775_mu_massiveAndCKM_LO'     , ['HN3L_M_3_V_0p22360679775_mu_massiveAndCKM_LO'     ], channel, '#splitline{m=3 GeV, |V|^{2}=5.0 10^{-2}}{Majorana}' , selection, 'hnl_m_3_v2_5p0Em02_majorana' ,  'firebrick'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       197.4, toplot=False), 
                Sample('HN3L_M_3_V_0p707106781187_mu_massiveAndCKM_LO'    , ['HN3L_M_3_V_0p707106781187_mu_massiveAndCKM_LO'    ], channel, '#splitline{m=3 GeV, |V|^{2}=5.0 10^{-1}}{Majorana}' , selection, 'hnl_m_3_v2_5p0Em01_majorana' ,  'firebrick'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      1971.0, toplot=False), 
                Sample('HN3L_M_4_V_0p00290516780927_mu_massiveAndCKM_LO'  , ['HN3L_M_4_V_0p00290516780927_mu_massiveAndCKM_LO'  ], channel, '#splitline{m=4 GeV, |V|^{2}=8.4 10^{-6}}{Majorana}' , selection, 'hnl_m_4_v2_8p4Em06_majorana' ,  'indigo'       , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.03348, toplot=False), 
                Sample('HN3L_M_4_V_0p0707106781187_mu_massiveAndCKM_LO'   , ['HN3L_M_4_V_0p0707106781187_mu_massiveAndCKM_LO'   ], channel, '#splitline{m=4 GeV, |V|^{2}=5.0 10^{-3}}{Majorana}' , selection, 'hnl_m_4_v2_5p0Em03_majorana' ,  'indigo'       , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       19.57, toplot=False), 
                Sample('HN3L_M_4_V_0p22360679775_mu_massiveAndCKM_LO'     , ['HN3L_M_4_V_0p22360679775_mu_massiveAndCKM_LO'     ], channel, '#splitline{m=4 GeV, |V|^{2}=5.0 10^{-2}}{Majorana}' , selection, 'hnl_m_4_v2_5p0Em02_majorana' ,  'indigo'       , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       195.7, toplot=False), 
                Sample('HN3L_M_5_V_0p000316227766017_mu_massiveAndCKM_LO' , ['HN3L_M_5_V_0p000316227766017_mu_massiveAndCKM_LO' ], channel, '#splitline{m=5 GeV, |V|^{2}=1.0 10^{-7}}{Majorana}' , selection, 'hnl_m_5_v2_1p0Em07_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,   0.0003976, toplot=False), 
                Sample('HN3L_M_5_V_0p000547722557505_mu_massiveAndCKM_LO' , ['HN3L_M_5_V_0p000547722557505_mu_massiveAndCKM_LO' ], channel, '#splitline{m=5 GeV, |V|^{2}=3.0 10^{-7}}{Majorana}' , selection, 'hnl_m_5_v2_3p0Em07_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.001196, toplot=False), 
                Sample('HN3L_M_5_V_0p001_mu_massiveAndCKM_LO'             , ['HN3L_M_5_V_0p001_mu_massiveAndCKM_LO'             ], channel, '#splitline{m=5 GeV, |V|^{2}=1.0 10^{-6}}{Majorana}' , selection, 'hnl_m_5_v2_1p0Em06_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.003983, toplot=False), 
                Sample('HN3L_M_5_V_0p00145602197786_mu_massiveAndCKM_LO'  , ['HN3L_M_5_V_0p00145602197786_mu_massiveAndCKM_LO'  ], channel, '#splitline{m=5 GeV, |V|^{2}=2.1 10^{-6}}{Majorana}' , selection, 'hnl_m_5_v2_2p1Em06_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.00842, toplot=True ), 
                Sample('HN3L_M_5_V_0p0707106781187_mu_massiveAndCKM_LO'   , ['HN3L_M_5_V_0p0707106781187_mu_massiveAndCKM_LO'   ], channel, '#splitline{m=5 GeV, |V|^{2}=5.0 10^{-3}}{Majorana}' , selection, 'hnl_m_5_v2_5p0Em03_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       19.57, toplot=False), 
                Sample('HN3L_M_5_V_0p22360679775_mu_massiveAndCKM_LO'     , ['HN3L_M_5_V_0p22360679775_mu_massiveAndCKM_LO'     ], channel, '#splitline{m=5 GeV, |V|^{2}=5.0 10^{-2}}{Majorana}' , selection, 'hnl_m_5_v2_5p0Em02_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       195.8, toplot=False), 
                Sample('HN3L_M_6_V_0p00202484567313_mu_massiveAndCKM_LO'  , ['HN3L_M_6_V_0p00202484567313_mu_massiveAndCKM_LO'  ], channel, '#splitline{m=6 GeV, |V|^{2}=4.1 10^{-6}}{Majorana}' , selection, 'hnl_m_6_v2_4p1Em06_majorana' ,  'olive'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.01656, toplot=False), 
                Sample('HN3L_M_6_V_0p0316227766017_mu_massiveAndCKM_LO'   , ['HN3L_M_6_V_0p0316227766017_mu_massiveAndCKM_LO'   ], channel, '#splitline{m=6 GeV, |V|^{2}=1.0 10^{-3}}{Majorana}' , selection, 'hnl_m_6_v2_1p0Em03_majorana' ,  'olive'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       3.977, toplot=False), 
                Sample('HN3L_M_6_V_0p1_mu_massiveAndCKM_LO'               , ['HN3L_M_6_V_0p1_mu_massiveAndCKM_LO'               ], channel, '#splitline{m=6 GeV, |V|^{2}=1.0 10^{-2}}{Majorana}' , selection, 'hnl_m_6_v2_1p0Em02_majorana' ,  'olive'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       39.68, toplot=False), 
                Sample('HN3L_M_7_V_0p0022361_mu_massiveAndCKM_LO'         , ['HN3L_M_7_V_0p0022361_mu_massiveAndCKM_LO'         ], channel, '#splitline{m=7 GeV, |V|^{2}=5.0 10^{-6}}{Majorana}' , selection, 'hnl_m_7_v2_5p0Em06_majorana' ,  'peru'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.02013, toplot=False), 
                Sample('HN3L_M_7_V_0p0316227766017_mu_massiveAndCKM_LO'   , ['HN3L_M_7_V_0p0316227766017_mu_massiveAndCKM_LO'   ], channel, '#splitline{m=7 GeV, |V|^{2}=1.0 10^{-3}}{Majorana}' , selection, 'hnl_m_7_v2_1p0Em03_majorana' ,  'peru'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       4.026, toplot=False), 
#                 Sample('HN3L_M_7_V_0p1_mu_massiveAndCKM_LO'               , ['HN3L_M_7_V_0p1_mu_massiveAndCKM_LO'               ], channel, '#splitline{m=7 GeV, |V|^{2}=1.0 10^{-2}}{Majorana}' , selection, 'hnl_m_7_v2_1p0Em02_majorana' ,  'peru'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       40.18, toplot=False), 
                Sample('HN3L_M_8_V_0p000547722557505_mu_massiveAndCKM_LO' , ['HN3L_M_8_V_0p000547722557505_mu_massiveAndCKM_LO' ], channel, '#splitline{m=8 GeV, |V|^{2}=3.0 10^{-7}}{Majorana}' , selection, 'hnl_m_8_v2_3p0Em07_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.001231, toplot=False), 
                Sample('HN3L_M_8_V_0p001_mu_massiveAndCKM_LO'             , ['HN3L_M_8_V_0p001_mu_massiveAndCKM_LO'             ], channel, '#splitline{m=8 GeV, |V|^{2}=1.0 10^{-6}}{Majorana}' , selection, 'hnl_m_8_v2_1p0Em06_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.004103, toplot=False), 
                Sample('HN3L_M_8_V_0p00151327459504_mu_massiveAndCKM_LO'  , ['HN3L_M_8_V_0p00151327459504_mu_massiveAndCKM_LO'  ], channel, '#splitline{m=8 GeV, |V|^{2}=2.3 10^{-6}}{Majorana}' , selection, 'hnl_m_8_v2_2p3Em06_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.00939, toplot=False), 
                Sample('HN3L_M_8_V_0p0022360679775_mu_massiveAndCKM_LO'   , ['HN3L_M_8_V_0p0022360679775_mu_massiveAndCKM_LO'   ], channel, '#splitline{m=8 GeV, |V|^{2}=5.0 10^{-6}}{Majorana}' , selection, 'hnl_m_8_v2_5p0Em06_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.02025, toplot=False), 
                Sample('HN3L_M_8_V_0p0316227766017_mu_massiveAndCKM_LO'   , ['HN3L_M_8_V_0p0316227766017_mu_massiveAndCKM_LO'   ], channel, '#splitline{m=8 GeV, |V|^{2}=1.0 10^{-3}}{Majorana}' , selection, 'hnl_m_8_v2_1p0Em03_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       4.044, toplot=False), 
                Sample('HN3L_M_8_V_0p1_mu_massiveAndCKM_LO'               , ['HN3L_M_8_V_0p1_mu_massiveAndCKM_LO'               ], channel, '#splitline{m=8 GeV, |V|^{2}=1.0 10^{-2}}{Majorana}' , selection, 'hnl_m_8_v2_1p0Em02_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       40.55, toplot=False), 
                Sample('HN3L_M_9_V_0p00316227766017_mu_massiveAndCKM_LO'  , ['HN3L_M_9_V_0p00316227766017_mu_massiveAndCKM_LO'  ], channel, '#splitline{m=9 GeV, |V|^{2}=1.0 10^{-5}}{Majorana}' , selection, 'hnl_m_9_v2_1p0Em05_majorana' ,  'plum'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.04053, toplot=False), 
                Sample('HN3L_M_9_V_0p0316227766017_mu_massiveAndCKM_LO'   , ['HN3L_M_9_V_0p0316227766017_mu_massiveAndCKM_LO'   ], channel, '#splitline{m=9 GeV, |V|^{2}=1.0 10^{-3}}{Majorana}' , selection, 'hnl_m_9_v2_1p0Em03_majorana' ,  'plum'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       4.044, toplot=False), 
                Sample('HN3L_M_10_V_0p000316227766017_mu_massiveAndCKM_LO', ['HN3L_M_10_V_0p000316227766017_mu_massiveAndCKM_LO'], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-7}}{Majorana}', selection, 'hnl_m_10_v2_1p0Em07_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,   0.0004131, toplot=False), 
                Sample('HN3L_M_10_V_0p000547722557505_mu_massiveAndCKM_LO', ['HN3L_M_10_V_0p000547722557505_mu_massiveAndCKM_LO'], channel, '#splitline{m=10 GeV, |V|^{2}=3.0 10^{-7}}{Majorana}', selection, 'hnl_m_10_v2_3p0Em07_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.001238, toplot=False), 
                Sample('HN3L_M_10_V_0p000756967634711_mu_massiveAndCKM_LO', ['HN3L_M_10_V_0p000756967634711_mu_massiveAndCKM_LO'], channel, '#splitline{m=10 GeV, |V|^{2}=5.7 10^{-7}}{Majorana}', selection, 'hnl_m_10_v2_5p7Em07_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.002363, toplot=False), 
                Sample('HN3L_M_10_V_0p001_mu_massiveAndCKM_LO'            , ['HN3L_M_10_V_0p001_mu_massiveAndCKM_LO'            ], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-6}}{Majorana}', selection, 'hnl_m_10_v2_1p0Em06_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.004131, toplot=True ), 
                Sample('HN3L_M_10_V_0p00316227766017_mu_massiveAndCKM_LO' , ['HN3L_M_10_V_0p00316227766017_mu_massiveAndCKM_LO' ], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-5}}{Majorana}', selection, 'hnl_m_10_v2_1p0Em05_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.04067, toplot=False), 
                Sample('HN3L_M_10_V_0p01_mu_massiveAndCKM_LO'             , ['HN3L_M_10_V_0p01_mu_massiveAndCKM_LO'             ], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-4}}{Majorana}', selection, 'hnl_m_10_v2_1p0Em04_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.4074, toplot=False), 
                Sample('HN3L_M_11_V_0p00316227766017_mu_massiveAndCKM_LO' , ['HN3L_M_11_V_0p00316227766017_mu_massiveAndCKM_LO' ], channel, '#splitline{m=11 GeV, |V|^{2}=1.0 10^{-5}}{Majorana}', selection, 'hnl_m_11_v2_1p0Em05_majorana',  'seagreen'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.04079, toplot=False), 
                Sample('HN3L_M_11_V_0p01_mu_massiveAndCKM_LO'             , ['HN3L_M_11_V_0p01_mu_massiveAndCKM_LO'             ], channel, '#splitline{m=11 GeV, |V|^{2}=1.0 10^{-4}}{Majorana}', selection, 'hnl_m_11_v2_1p0Em04_majorana',  'seagreen'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.4061, toplot=False), 
                Sample('HN3L_M_12_V_0p00316227766017_mu_massiveAndCKM_LO' , ['HN3L_M_12_V_0p00316227766017_mu_massiveAndCKM_LO' ], channel, '#splitline{m=12 GeV, |V|^{2}=1.0 10^{-5}}{Majorana}', selection, 'hnl_m_12_v2_1p0Em05_majorana',  'coral'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.04035, toplot=False), 
                Sample('HN3L_M_12_V_0p01_mu_massiveAndCKM_LO'             , ['HN3L_M_12_V_0p01_mu_massiveAndCKM_LO'             ], channel, '#splitline{m=12 GeV, |V|^{2}=1.0 10^{-4}}{Majorana}', selection, 'hnl_m_12_v2_1p0Em04_majorana',  'coral'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.4053, toplot=False), 
                Sample('HN3L_M_20_V_0p001_mu_massiveAndCKM_LO'            , ['HN3L_M_20_V_0p001_mu_massiveAndCKM_LO'            ], channel, '#splitline{m=20 GeV, |V|^{2}=1.0 10^{-6}}{Majorana}', selection, 'hnl_m_20_v2_1p0Em06_majorana',  'crimson'      , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.003857, toplot=False), 
                Sample('HN3L_M_20_V_0p00316227766017_mu_massiveAndCKM_LO' , ['HN3L_M_20_V_0p00316227766017_mu_massiveAndCKM_LO' ], channel, '#splitline{m=20 GeV, |V|^{2}=1.0 10^{-5}}{Majorana}', selection, 'hnl_m_20_v2_1p0Em05_majorana',  'crimson'      , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.03855, toplot=False), 
                Sample('HN3L_M_20_V_0p01_mu_massiveAndCKM_LO'             , ['HN3L_M_20_V_0p01_mu_massiveAndCKM_LO'             ], channel, '#splitline{m=20 GeV, |V|^{2}=1.0 10^{-4}}{Majorana}', selection, 'hnl_m_20_v2_1p0Em04_majorana',  'crimson'      , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.3795, toplot=False), 
            ]
                
    elif channel [0] == 'e': 
        if mini:
            signal = [
                Sample('HN3L_M_2_V_0p0110905365064_e_massiveAndCKM_LO'   , ['HN3L_M_2_V_0p0110905365064_e_massiveAndCKM_LO' ], channel, '#splitline{m=2 GeV, |V|^{2}=1.2 10^{-4}}{Majorana}' , selection, 'hnl_m_2_v2_1p2Em04_majorana' , 'forestgreen', 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.5273, toplot=True), 
                Sample('HN3L_M_8_V_0p0022360679775_e_massiveAndCKM_LO'   , ['HN3L_M_8_V_0p0022360679775_e_massiveAndCKM_LO' ], channel, '#splitline{m=8 GeV, |V|^{2}=5.0 10^{-6}}{Majorana}' , selection, 'hnl_m_8_v2_5p0Em06_majorana' , 'darkgray'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.02046, toplot=True), 
                Sample('HN3L_M_12_V_0p01_e_massiveAndCKM_LO'             , ['HN3L_M_12_V_0p01_e_massiveAndCKM_LO'           ], channel, '#splitline{m=12 GeV, |V|^{2}=1.0 10^{-4}}{Majorana}', selection, 'hnl_m_12_v2_1p0Em04_majorana', 'coral'      , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.4113, toplot=True), 
             ]
        else:
            signal = [
                Sample('HN3L_M_1_V_0p022360679775_e_massiveAndCKM_LO'    , ['HN3L_M_1_V_0p022360679775_e_massiveAndCKM_LO'    ], channel, '#splitline{m=1 GeV, |V|^{2}=5.0 10^{-04}}{Majorana}'  , selection, 'hnl_m_1_v2_5p0Em04_majorana' ,  'darkorange'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       2.087, toplot=False), 
                Sample('HN3L_M_1_V_0p0949736805647_e_massiveAndCKM_LO'   , ['HN3L_M_1_V_0p0949736805647_e_massiveAndCKM_LO'   ], channel, '#splitline{m=1 GeV, |V|^{2}=9.0 10^{-03}}{Majorana}'  , selection, 'hnl_m_1_v2_9p0Em03_majorana' ,  'darkorange'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       38.24, toplot=False), 
                Sample('HN3L_M_1_V_0p212367605816_e_massiveAndCKM_LO'    , ['HN3L_M_1_V_0p212367605816_e_massiveAndCKM_LO'    ], channel, '#splitline{m=1 GeV, |V|^{2}=4.5 10^{-02}}{Majorana}'  , selection, 'hnl_m_1_v2_4p5Em02_majorana' ,  'darkorange'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       190.7, toplot=False), 
                Sample('HN3L_M_1_V_0p707106781187_e_massiveAndCKM_LO'    , ['HN3L_M_1_V_0p707106781187_e_massiveAndCKM_LO'    ], channel, '#splitline{m=1 GeV, |V|^{2}=5.0 10^{-01}}{Majorana}'  , selection, 'hnl_m_1_v2_5p0Em01_majorana' ,  'darkorange'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      2090.0, toplot=False), 
                Sample('HN3L_M_2_V_0p0110905365064_e_massiveAndCKM_LO'   , ['HN3L_M_2_V_0p0110905365064_e_massiveAndCKM_LO'   ], channel, '#splitline{m=2 GeV, |V|^{2}=1.2 10^{-04}}{Majorana}'  , selection, 'hnl_m_2_v2_1p2Em04_majorana' ,  'forestgreen'  , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.5264, toplot=True ), 
                Sample('HN3L_M_2_V_0p0248394846967_e_massiveAndCKM_LO'   , ['HN3L_M_2_V_0p0248394846967_e_massiveAndCKM_LO'   ], channel, '#splitline{m=2 GeV, |V|^{2}=6.2 10^{-04}}{Majorana}'  , selection, 'hnl_m_2_v2_6p2Em04_majorana' ,  'forestgreen'  , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,        2.64, toplot=False), 
                Sample('HN3L_M_2_V_0p22360679775_e_massiveAndCKM_LO'     , ['HN3L_M_2_V_0p22360679775_e_massiveAndCKM_LO'     ], channel, '#splitline{m=2 GeV, |V|^{2}=5.0 10^{-02}}{Majorana}'  , selection, 'hnl_m_2_v2_5p0Em02_majorana' ,  'forestgreen'  , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       212.3, toplot=False), 
                Sample('HN3L_M_2_V_0p707106781187_e_massiveAndCKM_LO'    , ['HN3L_M_2_V_0p707106781187_e_massiveAndCKM_LO'    ], channel, '#splitline{m=2 GeV, |V|^{2}=5.0 10^{-01}}{Majorana}'  , selection, 'hnl_m_2_v2_5p0Em01_majorana' ,  'forestgreen'  , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      2120.0, toplot=False), 
                Sample('HN3L_M_3_V_0p00707813534767_e_massiveAndCKM_LO'  , ['HN3L_M_3_V_0p00707813534767_e_massiveAndCKM_LO'  ], channel, '#splitline{m=3 GeV, |V|^{2}=5.0 10^{-05}}{Majorana}'  , selection, 'hnl_m_3_v2_5p0Em05_majorana' ,  'firebrick'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.2022, toplot=False), 
                Sample('HN3L_M_3_V_0p22360679775_e_massiveAndCKM_LO'     , ['HN3L_M_3_V_0p22360679775_e_massiveAndCKM_LO'     ], channel, '#splitline{m=3 GeV, |V|^{2}=5.0 10^{-02}}{Majorana}'  , selection, 'hnl_m_3_v2_5p0Em02_majorana' ,  'firebrick'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       200.4, toplot=False), 
                Sample('HN3L_M_3_V_0p707106781187_e_massiveAndCKM_LO'    , ['HN3L_M_3_V_0p707106781187_e_massiveAndCKM_LO'    ], channel, '#splitline{m=3 GeV, |V|^{2}=5.0 10^{-01}}{Majorana}'  , selection, 'hnl_m_3_v2_5p0Em01_majorana' ,  'firebrick'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      2000.0, toplot=False), 
                Sample('HN3L_M_4_V_0p00290516780927_e_massiveAndCKM_LO'  , ['HN3L_M_4_V_0p00290516780927_e_massiveAndCKM_LO'  ], channel, '#splitline{m=4 GeV, |V|^{2}=8.4 10^{-06}}{Majorana}'  , selection, 'hnl_m_4_v2_8p4Em06_majorana' ,  'indigo'       , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.03366, toplot=False), 
                Sample('HN3L_M_4_V_0p0707106781187_e_massiveAndCKM_LO'   , ['HN3L_M_4_V_0p0707106781187_e_massiveAndCKM_LO'   ], channel, '#splitline{m=4 GeV, |V|^{2}=5.0 10^{-03}}{Majorana}'  , selection, 'hnl_m_4_v2_5p0Em03_majorana' ,  'indigo'       , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       19.74, toplot=False), 
                Sample('HN3L_M_4_V_0p22360679775_e_massiveAndCKM_LO'     , ['HN3L_M_4_V_0p22360679775_e_massiveAndCKM_LO'     ], channel, '#splitline{m=4 GeV, |V|^{2}=5.0 10^{-02}}{Majorana}'  , selection, 'hnl_m_4_v2_5p0Em02_majorana' ,  'indigo'       , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       196.6, toplot=False), 
                Sample('HN3L_M_5_V_0p000316227766017_e_massiveAndCKM_LO' , ['HN3L_M_5_V_0p000316227766017_e_massiveAndCKM_LO' ], channel, '#splitline{m=5 GeV, |V|^{2}=1.0 10^{-07}}{Majorana}'  , selection, 'hnl_m_5_v2_1p0Em07_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,   0.0003985, toplot=False), 
                Sample('HN3L_M_5_V_0p000547722557505_e_massiveAndCKM_LO' , ['HN3L_M_5_V_0p000547722557505_e_massiveAndCKM_LO' ], channel, '#splitline{m=5 GeV, |V|^{2}=3.0 10^{-07}}{Majorana}'  , selection, 'hnl_m_5_v2_3p0Em07_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.001193, toplot=False), 
                Sample('HN3L_M_5_V_0p001_e_massiveAndCKM_LO'             , ['HN3L_M_5_V_0p001_e_massiveAndCKM_LO'             ], channel, '#splitline{m=5 GeV, |V|^{2}=1.0 10^{-06}}{Majorana}'  , selection, 'hnl_m_5_v2_1p0Em06_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.003985, toplot=False), 
                Sample('HN3L_M_5_V_0p00145602197786_e_massiveAndCKM_LO'  , ['HN3L_M_5_V_0p00145602197786_e_massiveAndCKM_LO'  ], channel, '#splitline{m=5 GeV, |V|^{2}=2.1 10^{-06}}{Majorana}'  , selection, 'hnl_m_5_v2_2p1Em06_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.008468, toplot=False), 
                Sample('HN3L_M_5_V_0p0707106781187_e_massiveAndCKM_LO'   , ['HN3L_M_5_V_0p0707106781187_e_massiveAndCKM_LO'   ], channel, '#splitline{m=5 GeV, |V|^{2}=5.0 10^{-03}}{Majorana}'  , selection, 'hnl_m_5_v2_5p0Em03_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       19.86, toplot=False), 
                Sample('HN3L_M_5_V_0p22360679775_e_massiveAndCKM_LO'     , ['HN3L_M_5_V_0p22360679775_e_massiveAndCKM_LO'     ], channel, '#splitline{m=5 GeV, |V|^{2}=5.0 10^{-02}}{Majorana}'  , selection, 'hnl_m_5_v2_5p0Em02_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       198.0, toplot=False), 
                Sample('HN3L_M_6_V_0p00202484567313_e_massiveAndCKM_LO'  , ['HN3L_M_6_V_0p00202484567313_e_massiveAndCKM_LO'  ], channel, '#splitline{m=6 GeV, |V|^{2}=4.1 10^{-06}}{Majorana}'  , selection, 'hnl_m_6_v2_4p1Em06_majorana' ,  'olive'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.01655, toplot=False), 
                Sample('HN3L_M_6_V_0p0316227766017_e_massiveAndCKM_LO'   , ['HN3L_M_6_V_0p0316227766017_e_massiveAndCKM_LO'   ], channel, '#splitline{m=6 GeV, |V|^{2}=1.0 10^{-03}}{Majorana}'  , selection, 'hnl_m_6_v2_1p0Em03_majorana' ,  'olive'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       4.003, toplot=False), 
                Sample('HN3L_M_6_V_0p1_e_massiveAndCKM_LO'               , ['HN3L_M_6_V_0p1_e_massiveAndCKM_LO'               ], channel, '#splitline{m=6 GeV, |V|^{2}=1.0 10^{-02}}{Majorana}'  , selection, 'hnl_m_6_v2_1p0Em02_majorana' ,  'olive'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       39.96, toplot=False), 
                Sample('HN3L_M_7_V_0p0022361_e_massiveAndCKM_LO'         , ['HN3L_M_7_V_0p0022361_e_massiveAndCKM_LO'         ], channel, '#splitline{m=7 GeV, |V|^{2}=5.0 10^{-06}}{Majorana}'  , selection, 'hnl_m_7_v2_5p0Em06_majorana' ,  'peru'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.02018, toplot=False), 
                Sample('HN3L_M_7_V_0p0316227766017_e_massiveAndCKM_LO'   , ['HN3L_M_7_V_0p0316227766017_e_massiveAndCKM_LO'   ], channel, '#splitline{m=7 GeV, |V|^{2}=1.0 10^{-03}}{Majorana}'  , selection, 'hnl_m_7_v2_1p0Em03_majorana' ,  'peru'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       4.031, toplot=False), 
                Sample('HN3L_M_7_V_0p1_e_massiveAndCKM_LO'               , ['HN3L_M_7_V_0p1_e_massiveAndCKM_LO'               ], channel, '#splitline{m=7 GeV, |V|^{2}=1.0 10^{-02}}{Majorana}'  , selection, 'hnl_m_7_v2_1p0Em02_majorana' ,  'peru'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       40.49, toplot=False), 
                Sample('HN3L_M_8_V_0p000316227766017_e_massiveAndCKM_LO' , ['HN3L_M_8_V_0p000316227766017_e_massiveAndCKM_LO' ], channel, '#splitline{m=8 GeV, |V|^{2}=1.0 10^{-07}}{Majorana}'  , selection, 'hnl_m_8_v2_1p0Em07_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,   0.0004101, toplot=False), 
                Sample('HN3L_M_8_V_0p000547722557505_e_massiveAndCKM_LO' , ['HN3L_M_8_V_0p000547722557505_e_massiveAndCKM_LO' ], channel, '#splitline{m=8 GeV, |V|^{2}=3.0 10^{-07}}{Majorana}'  , selection, 'hnl_m_8_v2_3p0Em07_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.00123, toplot=False), 
                Sample('HN3L_M_8_V_0p001_e_massiveAndCKM_LO'             , ['HN3L_M_8_V_0p001_e_massiveAndCKM_LO'             ], channel, '#splitline{m=8 GeV, |V|^{2}=1.0 10^{-06}}{Majorana}'  , selection, 'hnl_m_8_v2_1p0Em06_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.004101, toplot=False), 
                Sample('HN3L_M_8_V_0p00151327459504_e_massiveAndCKM_LO'  , ['HN3L_M_8_V_0p00151327459504_e_massiveAndCKM_LO'  ], channel, '#splitline{m=8 GeV, |V|^{2}=2.3 10^{-06}}{Majorana}'  , selection, 'hnl_m_8_v2_2p3Em06_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.009365, toplot=False), 
                Sample('HN3L_M_8_V_0p0022360679775_e_massiveAndCKM_LO'   , ['HN3L_M_8_V_0p0022360679775_e_massiveAndCKM_LO'   ], channel, '#splitline{m=8 GeV, |V|^{2}=5.0 10^{-06}}{Majorana}'  , selection, 'hnl_m_8_v2_5p0Em06_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.02026, toplot=True ), 
                Sample('HN3L_M_8_V_0p0316227766017_e_massiveAndCKM_LO'   , ['HN3L_M_8_V_0p0316227766017_e_massiveAndCKM_LO'   ], channel, '#splitline{m=8 GeV, |V|^{2}=1.0 10^{-03}}{Majorana}'  , selection, 'hnl_m_8_v2_1p0Em03_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       4.071, toplot=False), 
                Sample('HN3L_M_8_V_0p1_e_massiveAndCKM_LO'               , ['HN3L_M_8_V_0p1_e_massiveAndCKM_LO'               ], channel, '#splitline{m=8 GeV, |V|^{2}=1.0 10^{-02}}{Majorana}'  , selection, 'hnl_m_8_v2_1p0Em02_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       40.68, toplot=False), 
                Sample('HN3L_M_9_V_0p00316227766017_e_massiveAndCKM_LO'  , ['HN3L_M_9_V_0p00316227766017_e_massiveAndCKM_LO'  ], channel, '#splitline{m=9 GeV, |V|^{2}=1.0 10^{-05}}{Majorana}'  , selection, 'hnl_m_9_v2_1p0Em05_majorana' ,  'plum'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.04064, toplot=False), 
                Sample('HN3L_M_9_V_0p0316227766017_e_massiveAndCKM_LO'   , ['HN3L_M_9_V_0p0316227766017_e_massiveAndCKM_LO'   ], channel, '#splitline{m=9 GeV, |V|^{2}=1.0 10^{-03}}{Majorana}'  , selection, 'hnl_m_9_v2_1p0Em03_majorana' ,  'plum'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       4.073, toplot=False), 
                Sample('HN3L_M_10_V_0p000316227766017_e_massiveAndCKM_LO', ['HN3L_M_10_V_0p000316227766017_e_massiveAndCKM_LO'], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-07}}{Majorana}' , selection, 'hnl_m_10_v2_1p0Em07_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,   0.0004118, toplot=False), 
                Sample('HN3L_M_10_V_0p000547722557505_e_massiveAndCKM_LO', ['HN3L_M_10_V_0p000547722557505_e_massiveAndCKM_LO'], channel, '#splitline{m=10 GeV, |V|^{2}=3.0 10^{-07}}{Majorana}' , selection, 'hnl_m_10_v2_3p0Em07_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.001238, toplot=False), 
                Sample('HN3L_M_10_V_0p000756967634711_e_massiveAndCKM_LO', ['HN3L_M_10_V_0p000756967634711_e_massiveAndCKM_LO'], channel, '#splitline{m=10 GeV, |V|^{2}=5.7 10^{-07}}{Majorana}' , selection, 'hnl_m_10_v2_5p7Em07_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.00236, toplot=False), 
                Sample('HN3L_M_10_V_0p001_e_massiveAndCKM_LO'            , ['HN3L_M_10_V_0p001_e_massiveAndCKM_LO'            ], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-06}}{Majorana}' , selection, 'hnl_m_10_v2_1p0Em06_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.004113, toplot=False), 
                Sample('HN3L_M_10_V_0p00316227766017_e_massiveAndCKM_LO' , ['HN3L_M_10_V_0p00316227766017_e_massiveAndCKM_LO' ], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-05}}{Majorana}' , selection, 'hnl_m_10_v2_1p0Em05_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.04081, toplot=False), 
                Sample('HN3L_M_10_V_0p01_e_massiveAndCKM_LO'             , ['HN3L_M_10_V_0p01_e_massiveAndCKM_LO'             ], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-04}}{Majorana}' , selection, 'hnl_m_10_v2_1p0Em04_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.4071, toplot=False), 
                Sample('HN3L_M_11_V_0p00316227766017_e_massiveAndCKM_LO' , ['HN3L_M_11_V_0p00316227766017_e_massiveAndCKM_LO' ], channel, '#splitline{m=11 GeV, |V|^{2}=1.0 10^{-05}}{Majorana}' , selection, 'hnl_m_11_v2_1p0Em05_majorana',  'seagreen'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.04076, toplot=False), 
                Sample('HN3L_M_11_V_0p01_e_massiveAndCKM_LO'             , ['HN3L_M_11_V_0p01_e_massiveAndCKM_LO'             ], channel, '#splitline{m=11 GeV, |V|^{2}=1.0 10^{-04}}{Majorana}' , selection, 'hnl_m_11_v2_1p0Em04_majorana',  'seagreen'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.4067, toplot=False), 
                Sample('HN3L_M_12_V_0p00316227766017_e_massiveAndCKM_LO' , ['HN3L_M_12_V_0p00316227766017_e_massiveAndCKM_LO' ], channel, '#splitline{m=12 GeV, |V|^{2}=1.0 10^{-05}}{Majorana}' , selection, 'hnl_m_12_v2_1p0Em05_majorana',  'coral'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.04059, toplot=False), 
                Sample('HN3L_M_12_V_0p01_e_massiveAndCKM_LO'             , ['HN3L_M_12_V_0p01_e_massiveAndCKM_LO'             ], channel, '#splitline{m=12 GeV, |V|^{2}=1.0 10^{-04}}{Majorana}' , selection, 'hnl_m_12_v2_1p0Em04_majorana',  'coral'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.4056, toplot=True ), 
                Sample('HN3L_M_20_V_0p001_e_massiveAndCKM_LO'            , ['HN3L_M_20_V_0p001_e_massiveAndCKM_LO'            ], channel, '#splitline{m=20 GeV, |V|^{2}=1.0 10^{-06}}{Majorana}' , selection, 'hnl_m_20_v2_1p0Em06_majorana',  'crimson'      , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.003858, toplot=False), 
                Sample('HN3L_M_20_V_0p00316227766017_e_massiveAndCKM_LO' , ['HN3L_M_20_V_0p00316227766017_e_massiveAndCKM_LO' ], channel, '#splitline{m=20 GeV, |V|^{2}=1.0 10^{-05}}{Majorana}' , selection, 'hnl_m_20_v2_1p0Em05_majorana',  'crimson'      , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.03866, toplot=False), 
                Sample('HN3L_M_20_V_0p01_e_massiveAndCKM_LO'             , ['HN3L_M_20_V_0p01_e_massiveAndCKM_LO'             ], channel, '#splitline{m=20 GeV, |V|^{2}=1.0 10^{-04}}{Majorana}' , selection, 'hnl_m_20_v2_1p0Em04_majorana',  'crimson'      , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.3804, toplot=False), 
             ]
        

    if not mini:
        # generate reweighted samples
        # use all non-empty samples for reweighing
        generators = [isample for isample in signal if isample.df.size>0]
        existing_datacard_names= [isample.datacard_name for isample in signal]
        for igenerator in generators:
            for iw in signal_weights:
                v2_val = iw.split('em')[0]
                v2_exp = iw.split('em')[1]
                mass = igenerator.name.split('_')[2]
                # don't produce *everything*
                if signal_weights_dict[iw] < ranges[int(mass)][0]: continue
                if signal_weights_dict[iw] > ranges[int(mass)][1]: continue  
                label = '#splitline{m=%s GeV, |V|^{2}=%s 10^{-%d}}{Majorana}' %(mass, v2_val, int(v2_exp))
                datacard_name = 'hnl_m_%s_v2_%s_majorana'%(mass, iw.replace('.','p').replace('em', 'Em'))
                # don't reweigh existing signals
                if datacard_name in existing_datacard_names:
                    continue
                
                new_sample = copy(igenerator)
            
                new_sample.label                = label
                new_sample.datacard_name        = datacard_name
                new_sample.toplot               = False
                new_sample.extra_signal_weights = ['ctau_w_v2_%s'%iw]
                new_sample.xs                   = igenerator.xs * igenerator.df['xs_w_v2_%s'%iw][0]
 
                print('generated reweighed signal', label, 'from', igenerator.name)           
                signal.append(new_sample)

        # now let's group the samples by their datacard name
        datacard_names = set([isample.datacard_name for isample in signal])
    
        merged_signals = []
        for ii, iname in enumerate(datacard_names):
    
            print ('\t', iname, '\t%d/%d'%(ii+1, len(datacard_names)))
            weighed_samples = [isample for isample in signal if isample.datacard_name==iname]
        
            merged_sample               = copy(weighed_samples[0])
            merged_sample.datacard_name = iname
            merged_sample.df            = pd.concat([isample.df for isample in weighed_samples])
            # take the average reweighed xs, because
            # the coupling^2 in Tom's page does not have all significant digits
            # and may be off by up to a few %
            # NB: the sample's name bear the _exact_ coupling (not squared!)
            merged_sample.xs            = np.mean([isample.xs for isample in weighed_samples])
            merged_sample.nevents       = np.sum([isample.nevents for isample in weighed_samples])
            merged_sample.lumi_scaling  = merged_sample.xs / merged_sample.nevents
        
            merged_signals.append(merged_sample)

            print(merged_sample.xs, '\t', [isample.xs for isample in weighed_samples])

            # free up some memory
            for already_merged in weighed_samples:
                signal.remove(already_merged)
                del already_merged
                gc.collect()
        
        signal = merged_signals
            
    return signal

if __name__ == '__main__':

    # Study using all signal for signal reweighing
    plot_dir = '/'.join([env['BASE_DIR'], 'plotter', 'plots', '2017']) 
    base_dir = '/'.join([env['BASE_DIR'], 'ntuples', 'may20', '2017'])
    bkgs = get_mc_samples('mmm', base_dir, 'HNLTreeProducer_mmm/tree.root', '1')
#     signals = get_signal_samples('mmm', base_dir, 'HNLTreeProducer_mmm/tree.root', '1', mini=False)
