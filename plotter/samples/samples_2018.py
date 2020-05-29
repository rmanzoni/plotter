from re import findall
import numpy as np
import pandas as pd
import gc # free mem up
from os import environ as env
from root_pandas import read_root
from collections import OrderedDict
from plotter.objects.sample import Sample, signal_weights_dict, signal_weights, ranges
from copy import deepcopy, copy
from itertools import groupby

def get_data_samples(channel, basedir, postfix, selection):
    if   channel [0] == 'm': lep = 'mu'
    elif channel [0] == 'e': lep = 'ele'
    assert lep == 'ele' or lep == 'mu', 'Lepton flavor error'
    data = [
        Sample('Single_{lep}_2018A'.format(lep=lep), ['Single_{lep}_2018A'.format(lep=lep)], channel, '2018A', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2018B'.format(lep=lep), ['Single_{lep}_2018B'.format(lep=lep)], channel, '2018B', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2018C'.format(lep=lep), ['Single_{lep}_2018C'.format(lep=lep)], channel, '2018C', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2018D'.format(lep=lep), ['Single_{lep}_2018D'.format(lep=lep)], channel, '2018D', selection, 'data_obs', 'black', 9999, '/'.join([basedir, 'data']), postfix, True, False, False, 1., 1.),
    ]
    return data

def get_mc_samples(channel, basedir, postfix, selection):
    mc = [
#         Sample('DYJetsToLL_M50_fxfx'    , ['DYJetsToLL_M50_fxfx', 'DYJetsToLL_M50_fxfx_ext'], channel,  r'DY$\to\ell\ell$', selection, 'DY_nlo'    , 'gold'     ,10, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,  6077.22),
#         Sample('DYJetsToLL_M50'         , ['DYJetsToLL_M50'     ], channel,  r'DY$\to\ell\ell$', selection, 'DY_lo'     , 'gold'     ,10, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,  6077.22),
        Sample('DYJetsToLL_M50'         , ['DYJetsToLL_M50', 'DYJetsToLL_M50_fxfx', 'DYJetsToLL_M50_fxfx_ext'], channel,  r'DY$\to\ell\ell$', selection, 'DY_lo'     , 'gold'     ,10, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,  6077.22),
        Sample('TTJets'                 , ['TTJets','TTJets_ext'                                             ], channel,  r'$t\bar{t}$'     , selection, 'TT'        , 'slateblue', 0, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,   831.76),
        Sample('WW'                     , ['WW'                                                              ], channel,  'WW'              , selection, 'WW'        , 'blue'     , 5, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    75.88),
        Sample('WZ'                     , ['WZ'                                                              ], channel,  'WZ'              , selection, 'WZ'        , 'blue'     , 5, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    27.6 ),
        Sample('ZZ'                     , ['ZZ'                                                              ], channel,  'ZZ'              , selection, 'ZZ'        , 'blue'     , 5, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    12.14),
        Sample('WJetsToLNu'             , ['WJetsToLNu'                                                      ], channel,  r'W$\to\ell\nu$'  , selection, 'W'         , 'brown'    , 2, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1., 59850.0 ),
        Sample('DYJetsToLL_M5to50'      , ['DYJetsToLL_M5to50'                                               ], channel,  r'DY$\to\ell\ell$', selection, 'DY_lo_low' , 'gold'     ,10, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1., 81880.0 ),
        Sample('ST_tW_inc'              , ['ST_tW_inc'                                                       ], channel,  r'single$-t$'     , selection, 'TtW'       , 'slateblue', 1, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    35.85),
        Sample('ST_tch_inc'             , ['ST_tch_inc'                                                      ], channel,  r'single$-t$'     , selection, 'Ttch'      , 'slateblue', 1, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,   136.02),
        Sample('STbar_tW_inc'           , ['STbar_tW_inc'                                                    ], channel,  r'single$-t$'     , selection, 'TbtW'      , 'slateblue', 1, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    35.85),
        Sample('STbar_tch_inc'          , ['STbar_tch_inc'                                                   ], channel,  r'single$-t$'     , selection, 'Tbtch'     , 'slateblue', 1, '/'.join([basedir, 'bkg']), postfix, False, True, False, 1.,    80.95),
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
                Sample('HN3L_M_1_V_0p022360679775_mu_massiveAndCKM_LO'             , ['HN3L_M_1_V_0p022360679775_mu_massiveAndCKM_LO'             ], channel, '#splitline{m=1 GeV, |V|^{2}=5.0 10^{-4}}{Majorana}'  , selection, 'hnl_m_1_v2_5p0Em04_majorana' ,  'darkorange'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       2.144, toplot=False), 
                Sample('HN3L_M_1_V_0p0949736805647_mu_massiveAndCKM_LO'            , ['HN3L_M_1_V_0p0949736805647_mu_massiveAndCKM_LO'            ], channel, '#splitline{m=1 GeV, |V|^{2}=9.0 10^{-3}}{Majorana}'  , selection, 'hnl_m_1_v2_9p0Em03_majorana' ,  'darkorange'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       38.67, toplot=False, is_generator=True), 
                Sample('HN3L_M_1_V_0p212367605816_mu_massiveAndCKM_LO'             , ['HN3L_M_1_V_0p212367605816_mu_massiveAndCKM_LO'             ], channel, '#splitline{m=1 GeV, |V|^{2}=4.5 10^{-2}}{Majorana}'  , selection, 'hnl_m_1_v2_4p5Em02_majorana' ,  'darkorange'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       193.3, toplot=False), 
                Sample('HN3L_M_1_V_0p707106781187_mu_massiveAndCKM_LO'             , ['HN3L_M_1_V_0p707106781187_mu_massiveAndCKM_LO'             ], channel, '#splitline{m=1 GeV, |V|^{2}=5.0 10^{-1}}{Majorana}'  , selection, 'hnl_m_1_v2_5p0Em01_majorana' ,  'darkorange'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      2146.0, toplot=False), 
                Sample('HN3L_M_2_V_0p0110905365064_mu_massiveAndCKM_LO'            , ['HN3L_M_2_V_0p0110905365064_mu_massiveAndCKM_LO'            ], channel, '#splitline{m=2 GeV, |V|^{2}=1.2 10^{-4}}{Majorana}'  , selection, 'hnl_m_2_v2_1p2Em04_majorana' ,  'forestgreen'  , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.5278, toplot=True, is_generator=True), 
                Sample('HN3L_M_2_V_0p0248394846967_mu_massiveAndCKM_LO'            , ['HN3L_M_2_V_0p0248394846967_mu_massiveAndCKM_LO'            ], channel, '#splitline{m=2 GeV, |V|^{2}=6.2 10^{-4}}{Majorana}'  , selection, 'hnl_m_2_v2_6p2Em04_majorana' ,  'forestgreen'  , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       2.647, toplot=False), 
                Sample('HN3L_M_2_V_0p22360679775_mu_massiveAndCKM_LO'              , ['HN3L_M_2_V_0p22360679775_mu_massiveAndCKM_LO'              ], channel, '#splitline{m=2 GeV, |V|^{2}=5.0 10^{-2}}{Majorana}'  , selection, 'hnl_m_2_v2_5p0Em02_majorana' ,  'forestgreen'  , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       214.4, toplot=False), 
                Sample('HN3L_M_2_V_0p707106781187_mu_massiveAndCKM_LO'             , ['HN3L_M_2_V_0p707106781187_mu_massiveAndCKM_LO'             ], channel, '#splitline{m=2 GeV, |V|^{2}=5.0 10^{-1}}{Majorana}'  , selection, 'hnl_m_2_v2_5p0Em01_majorana' ,  'forestgreen'  , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      2143.0, toplot=False), 
                Sample('HN3L_M_3_V_0p00707813534767_mu_massiveAndCKM_LO'           , ['HN3L_M_3_V_0p00707813534767_mu_massiveAndCKM_LO'           ], channel, '#splitline{m=3 GeV, |V|^{2}=5.0 10^{-5}}{Majorana}'  , selection, 'hnl_m_3_v2_5p0Em05_majorana' ,  'firebrick'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.2014, toplot=False, is_generator=True), 
                Sample('HN3L_M_3_V_0p22360679775_mu_massiveAndCKM_LO'              , ['HN3L_M_3_V_0p22360679775_mu_massiveAndCKM_LO'              ], channel, '#splitline{m=3 GeV, |V|^{2}=5.0 10^{-2}}{Majorana}'  , selection, 'hnl_m_3_v2_5p0Em02_majorana' ,  'firebrick'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       201.1, toplot=False), 
                Sample('HN3L_M_3_V_0p707106781187_mu_massiveAndCKM_LO'             , ['HN3L_M_3_V_0p707106781187_mu_massiveAndCKM_LO'             ], channel, '#splitline{m=3 GeV, |V|^{2}=5.0 10^{-1}}{Majorana}'  , selection, 'hnl_m_3_v2_5p0Em01_majorana' ,  'firebrick'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      2005.0, toplot=False), 
                Sample('HN3L_M_4_V_0p00290516780927_mu_massiveAndCKM_LO'           , ['HN3L_M_4_V_0p00290516780927_mu_massiveAndCKM_LO'           ], channel, '#splitline{m=4 GeV, |V|^{2}=8.4 10^{-6}}{Majorana}'  , selection, 'hnl_m_4_v2_8p4Em06_majorana' ,  'indigo'       , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.0335, toplot=False, is_generator=True), 
                Sample('HN3L_M_4_V_0p0707106781187_mu_massiveAndCKM_LO'            , ['HN3L_M_4_V_0p0707106781187_mu_massiveAndCKM_LO'            ], channel, '#splitline{m=4 GeV, |V|^{2}=5.0 10^{-3}}{Majorana}'  , selection, 'hnl_m_4_v2_5p0Em03_majorana' ,  'indigo'       , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       19.86, toplot=False), 
                Sample('HN3L_M_4_V_0p22360679775_mu_massiveAndCKM_LO'              , ['HN3L_M_4_V_0p22360679775_mu_massiveAndCKM_LO'              ], channel, '#splitline{m=4 GeV, |V|^{2}=5.0 10^{-2}}{Majorana}'  , selection, 'hnl_m_4_v2_5p0Em02_majorana' ,  'indigo'       , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       198.7, toplot=False), 
                Sample('HN3L_M_5_V_0p000316227766017_mu_massiveAndCKM_LO'          , ['HN3L_M_5_V_0p000316227766017_mu_massiveAndCKM_LO'          ], channel, '#splitline{m=5 GeV, |V|^{2}=1.0 10^{-7}}{Majorana}'  , selection, 'hnl_m_5_v2_1p0Em07_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,   0.0003981, toplot=False), 
                Sample('HN3L_M_5_V_0p000547722557505_mu_massiveAndCKM_LO'          , ['HN3L_M_5_V_0p000547722557505_mu_massiveAndCKM_LO'          ], channel, '#splitline{m=5 GeV, |V|^{2}=3.0 10^{-7}}{Majorana}'  , selection, 'hnl_m_5_v2_3p0Em07_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.001194, toplot=False), 
                Sample('HN3L_M_5_V_0p001_mu_massiveAndCKM_LO'                      , ['HN3L_M_5_V_0p001_mu_massiveAndCKM_LO'                      ], channel, '#splitline{m=5 GeV, |V|^{2}=1.0 10^{-6}}{Majorana}'  , selection, 'hnl_m_5_v2_1p0Em06_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.003977, toplot=False), 
                Sample('HN3L_M_5_V_0p00145602197786_mu_massiveAndCKM_LO'           , ['HN3L_M_5_V_0p00145602197786_mu_massiveAndCKM_LO'           ], channel, '#splitline{m=5 GeV, |V|^{2}=2.1 10^{-6}}{Majorana}'  , selection, 'hnl_m_5_v2_2p1Em06_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.008434, toplot=True, is_generator=True), 
                Sample('HN3L_M_5_V_0p0707106781187_mu_massiveAndCKM_LO'            , ['HN3L_M_5_V_0p0707106781187_mu_massiveAndCKM_LO'            ], channel, '#splitline{m=5 GeV, |V|^{2}=5.0 10^{-3}}{Majorana}'  , selection, 'hnl_m_5_v2_5p0Em03_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       19.88, toplot=False), 
                Sample('HN3L_M_5_V_0p22360679775_mu_massiveAndCKM_LO'              , ['HN3L_M_5_V_0p22360679775_mu_massiveAndCKM_LO'              ], channel, '#splitline{m=5 GeV, |V|^{2}=5.0 10^{-2}}{Majorana}'  , selection, 'hnl_m_5_v2_5p0Em02_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       198.8, toplot=False), 
                Sample('HN3L_M_6_V_0p00202484567313_mu_massiveAndCKM_LO'           , ['HN3L_M_6_V_0p00202484567313_mu_massiveAndCKM_LO'           ], channel, '#splitline{m=6 GeV, |V|^{2}=4.1 10^{-6}}{Majorana}'  , selection, 'hnl_m_6_v2_4p1Em06_majorana' ,  'olive'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.01655, toplot=False, is_generator=True), 
                Sample('HN3L_M_6_V_0p0316227766017_mu_massiveAndCKM_LO'            , ['HN3L_M_6_V_0p0316227766017_mu_massiveAndCKM_LO'            ], channel, '#splitline{m=6 GeV, |V|^{2}=1.0 10^{-3}}{Majorana}'  , selection, 'hnl_m_6_v2_1p0Em03_majorana' ,  'olive'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       4.045, toplot=False), 
                Sample('HN3L_M_6_V_0p1_mu_massiveAndCKM_LO'                        , ['HN3L_M_6_V_0p1_mu_massiveAndCKM_LO'                        ], channel, '#splitline{m=6 GeV, |V|^{2}=1.0 10^{-2}}{Majorana}'  , selection, 'hnl_m_6_v2_1p0Em02_majorana' ,  'olive'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       40.33, toplot=False), 
                Sample('HN3L_M_7_V_0p0022361_mu_massiveAndCKM_LO'                  , ['HN3L_M_7_V_0p0022361_mu_massiveAndCKM_LO'                  ], channel, '#splitline{m=7 GeV, |V|^{2}=5.0 10^{-6}}{Majorana}'  , selection, 'hnl_m_7_v2_5p0Em06_majorana' ,  'peru'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.02035, toplot=False, is_generator=True), 
                Sample('HN3L_M_7_V_0p0316227766017_mu_massiveAndCKM_LO'            , ['HN3L_M_7_V_0p0316227766017_mu_massiveAndCKM_LO'            ], channel, '#splitline{m=7 GeV, |V|^{2}=1.0 10^{-3}}{Majorana}'  , selection, 'hnl_m_7_v2_1p0Em03_majorana' ,  'peru'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       4.075, toplot=False), 
#                 Sample('HN3L_M_7_V_0p1_mu_massiveAndCKM_LO'                        , ['HN3L_M_7_V_0p1_mu_massiveAndCKM_LO'                        ], channel, '#splitline{m=7 GeV, |V|^{2}=1.0 10^{-2}}{Majorana}'  , selection, 'hnl_m_7_v2_1p0Em02_majorana' ,  'peru'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       40.65, toplot=False), 
                Sample('HN3L_M_8_V_0p000547722557505_mu_massiveAndCKM_LO'          , ['HN3L_M_8_V_0p000547722557505_mu_massiveAndCKM_LO'          ], channel, '#splitline{m=8 GeV, |V|^{2}=3.0 10^{-7}}{Majorana}'  , selection, 'hnl_m_8_v2_3p0Em07_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.001229, toplot=False), 
                Sample('HN3L_M_8_V_0p001_mu_massiveAndCKM_LO'                      , ['HN3L_M_8_V_0p001_mu_massiveAndCKM_LO'                      ], channel, '#splitline{m=8 GeV, |V|^{2}=1.0 10^{-6}}{Majorana}'  , selection, 'hnl_m_8_v2_1p0Em06_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.004104, toplot=False, is_generator=True), 
                Sample('HN3L_M_8_V_0p00151327459504_mu_massiveAndCKM_LO'           , ['HN3L_M_8_V_0p00151327459504_mu_massiveAndCKM_LO'           ], channel, '#splitline{m=8 GeV, |V|^{2}=2.3 10^{-6}}{Majorana}'  , selection, 'hnl_m_8_v2_2p3Em06_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.009374, toplot=False), 
                Sample('HN3L_M_8_V_0p0022360679775_mu_massiveAndCKM_LO'            , ['HN3L_M_8_V_0p0022360679775_mu_massiveAndCKM_LO'            ], channel, '#splitline{m=8 GeV, |V|^{2}=5.0 10^{-6}}{Majorana}'  , selection, 'hnl_m_8_v2_5p0Em06_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.0205, toplot=False), 
                Sample('HN3L_M_8_V_0p0316227766017_mu_massiveAndCKM_LO'            , ['HN3L_M_8_V_0p0316227766017_mu_massiveAndCKM_LO'            ], channel, '#splitline{m=8 GeV, |V|^{2}=1.0 10^{-3}}{Majorana}'  , selection, 'hnl_m_8_v2_1p0Em03_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       4.102, toplot=False), 
                Sample('HN3L_M_8_V_0p1_mu_massiveAndCKM_LO'                        , ['HN3L_M_8_V_0p1_mu_massiveAndCKM_LO'                        ], channel, '#splitline{m=8 GeV, |V|^{2}=1.0 10^{-2}}{Majorana}'  , selection, 'hnl_m_8_v2_1p0Em02_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       41.03, toplot=False), 
                Sample('HN3L_M_9_V_0p00316227766017_mu_massiveAndCKM_LO'           , ['HN3L_M_9_V_0p00316227766017_mu_massiveAndCKM_LO'           ], channel, '#splitline{m=9 GeV, |V|^{2}=1.0 10^{-5}}{Majorana}'  , selection, 'hnl_m_9_v2_1p0Em05_majorana' ,  'plum'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.04105, toplot=False, is_generator=True), 
                Sample('HN3L_M_9_V_0p0316227766017_mu_massiveAndCKM_LO'            , ['HN3L_M_9_V_0p0316227766017_mu_massiveAndCKM_LO'            ], channel, '#splitline{m=9 GeV, |V|^{2}=1.0 10^{-3}}{Majorana}'  , selection, 'hnl_m_9_v2_1p0Em03_majorana' ,  'plum'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       4.115, toplot=False), 
                Sample('HN3L_M_10_V_0p000316227766017_mu_massiveAndCKM_LO'         , ['HN3L_M_10_V_0p000316227766017_mu_massiveAndCKM_LO'         ], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-7}}{Majorana}' , selection, 'hnl_m_10_v2_1p0Em07_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,   0.0004118, toplot=False), 
                Sample('HN3L_M_10_V_0p000547722557505_mu_massiveAndCKM_LO'         , ['HN3L_M_10_V_0p000547722557505_mu_massiveAndCKM_LO'         ], channel, '#splitline{m=10 GeV, |V|^{2}=3.0 10^{-7}}{Majorana}' , selection, 'hnl_m_10_v2_3p0Em07_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.001237, toplot=False), 
                Sample('HN3L_M_10_V_0p000756967634711_mu_massiveAndCKM_LO'         , ['HN3L_M_10_V_0p000756967634711_mu_massiveAndCKM_LO'         ], channel, '#splitline{m=10 GeV, |V|^{2}=5.7 10^{-7}}{Majorana}' , selection, 'hnl_m_10_v2_5p7Em07_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.002362, toplot=False, is_generator=True), 
                Sample('HN3L_M_10_V_0p001_mu_massiveAndCKM_LO'                     , ['HN3L_M_10_V_0p001_mu_massiveAndCKM_LO'                     ], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-6}}{Majorana}' , selection, 'hnl_m_10_v2_1p0Em06_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.004121, toplot=True), 
                Sample('HN3L_M_10_V_0p00316227766017_mu_massiveAndCKM_LO'          , ['HN3L_M_10_V_0p00316227766017_mu_massiveAndCKM_LO'          ], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-5}}{Majorana}' , selection, 'hnl_m_10_v2_1p0Em05_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.04123, toplot=False), 
                Sample('HN3L_M_10_V_0p01_mu_massiveAndCKM_LO'                      , ['HN3L_M_10_V_0p01_mu_massiveAndCKM_LO'                      ], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-4}}{Majorana}' , selection, 'hnl_m_10_v2_1p0Em04_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.4128, toplot=False), 
                Sample('HN3L_M_11_V_0p00316227766017_mu_massiveAndCKM_LO'          , ['HN3L_M_11_V_0p00316227766017_mu_massiveAndCKM_LO'          ], channel, '#splitline{m=11 GeV, |V|^{2}=1.0 10^{-5}}{Majorana}' , selection, 'hnl_m_11_v2_1p0Em05_majorana',  'seagreen'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.04127, toplot=False, is_generator=True), 
                Sample('HN3L_M_11_V_0p01_mu_massiveAndCKM_LO'                      , ['HN3L_M_11_V_0p01_mu_massiveAndCKM_LO'                      ], channel, '#splitline{m=11 GeV, |V|^{2}=1.0 10^{-4}}{Majorana}' , selection, 'hnl_m_11_v2_1p0Em04_majorana',  'seagreen'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.4125, toplot=False), 
                Sample('HN3L_M_12_V_0p00316227766017_mu_massiveAndCKM_LO'          , ['HN3L_M_12_V_0p00316227766017_mu_massiveAndCKM_LO'          ], channel, '#splitline{m=12 GeV, |V|^{2}=1.0 10^{-5}}{Majorana}' , selection, 'hnl_m_12_v2_1p0Em05_majorana',  'coral'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.0411, toplot=False, is_generator=True), 
                Sample('HN3L_M_12_V_0p01_mu_massiveAndCKM_LO'                      , ['HN3L_M_12_V_0p01_mu_massiveAndCKM_LO'                      ], channel, '#splitline{m=12 GeV, |V|^{2}=1.0 10^{-4}}{Majorana}' , selection, 'hnl_m_12_v2_1p0Em04_majorana',  'coral'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.4117, toplot=False), 
                Sample('HN3L_M_20_V_0p001_mu_massiveAndCKM_LO'                     , ['HN3L_M_20_V_0p001_mu_massiveAndCKM_LO'                     ], channel, '#splitline{m=20 GeV, |V|^{2}=1.0 10^{-6}}{Majorana}' , selection, 'hnl_m_20_v2_1p0Em06_majorana',  'crimson'      , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.003856, toplot=False), 
                Sample('HN3L_M_20_V_0p00316227766017_mu_massiveAndCKM_LO'          , ['HN3L_M_20_V_0p00316227766017_mu_massiveAndCKM_LO'          ], channel, '#splitline{m=20 GeV, |V|^{2}=1.0 10^{-5}}{Majorana}' , selection, 'hnl_m_20_v2_1p0Em05_majorana',  'crimson'      , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.03854, toplot=False, is_generator=True), 
                Sample('HN3L_M_20_V_0p01_mu_massiveAndCKM_LO'                      , ['HN3L_M_20_V_0p01_mu_massiveAndCKM_LO'                      ], channel, '#splitline{m=20 GeV, |V|^{2}=1.0 10^{-4}}{Majorana}' , selection, 'hnl_m_20_v2_1p0Em04_majorana',  'crimson'      , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.3904, toplot=False), 
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
                Sample('HN3L_M_1_V_0p022360679775_e_massiveAndCKM_LO'                , ['HN3L_M_1_V_0p022360679775_e_massiveAndCKM_LO'                ], channel, '#splitline{m= 1 GeV, |V|^{2}=5.0 10^{-4}}{Majorana}' , selection, 'hnl_m_1_v2_5p0Em04_majorana' ,  'darkorange'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       2.118, toplot=False, is_generator=True), 
                Sample('HN3L_M_1_V_0p0949736805647_e_massiveAndCKM_LO'               , ['HN3L_M_1_V_0p0949736805647_e_massiveAndCKM_LO'               ], channel, '#splitline{m= 1 GeV, |V|^{2}=9.0 10^{-3}}{Majorana}' , selection, 'hnl_m_1_v2_9p0Em03_majorana' ,  'darkorange'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       38.23, toplot=False), 
                Sample('HN3L_M_1_V_0p212367605816_e_massiveAndCKM_LO'                , ['HN3L_M_1_V_0p212367605816_e_massiveAndCKM_LO'                ], channel, '#splitline{m= 1 GeV, |V|^{2}=4.5 10^{-2}}{Majorana}' , selection, 'hnl_m_1_v2_4p5Em02_majorana' ,  'darkorange'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       191.1, toplot=False), 
                Sample('HN3L_M_1_V_0p707106781187_e_massiveAndCKM_LO'                , ['HN3L_M_1_V_0p707106781187_e_massiveAndCKM_LO'                ], channel, '#splitline{m= 1 GeV, |V|^{2}=5.0 10^{-1}}{Majorana}' , selection, 'hnl_m_1_v2_5p0Em01_majorana' ,  'darkorange'   , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      2120.0, toplot=False), 
                Sample('HN3L_M_2_V_0p0110905365064_e_massiveAndCKM_LO'               , ['HN3L_M_2_V_0p0110905365064_e_massiveAndCKM_LO'               ], channel, '#splitline{m= 2 GeV, |V|^{2}=1.2 10^{-4}}{Majorana}' , selection, 'hnl_m_2_v2_1p2Em04_majorana' ,  'forestgreen'  , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.5273, toplot=True, is_generator=True), 
                Sample('HN3L_M_2_V_0p0248394846967_e_massiveAndCKM_LO'               , ['HN3L_M_2_V_0p0248394846967_e_massiveAndCKM_LO'               ], channel, '#splitline{m= 2 GeV, |V|^{2}=6.2 10^{-4}}{Majorana}' , selection, 'hnl_m_2_v2_6p2Em04_majorana' ,  'forestgreen'  , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       2.648, toplot=False), 
                Sample('HN3L_M_2_V_0p22360679775_e_massiveAndCKM_LO'                 , ['HN3L_M_2_V_0p22360679775_e_massiveAndCKM_LO'                 ], channel, '#splitline{m= 2 GeV, |V|^{2}=5.0 10^{-2}}{Majorana}' , selection, 'hnl_m_2_v2_5p0Em02_majorana' ,  'forestgreen'  , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       214.3, toplot=False), 
                Sample('HN3L_M_2_V_0p707106781187_e_massiveAndCKM_LO'                , ['HN3L_M_2_V_0p707106781187_e_massiveAndCKM_LO'                ], channel, '#splitline{m= 2 GeV, |V|^{2}=5.0 10^{-1}}{Majorana}' , selection, 'hnl_m_2_v2_5p0Em01_majorana' ,  'forestgreen'  , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      2141.0, toplot=False), 
                Sample('HN3L_M_3_V_0p00707813534767_e_massiveAndCKM_LO'              , ['HN3L_M_3_V_0p00707813534767_e_massiveAndCKM_LO'              ], channel, '#splitline{m= 3 GeV, |V|^{2}=5.0 10^{-5}}{Majorana}' , selection, 'hnl_m_3_v2_5p0Em05_majorana' ,  'firebrick'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.2022, toplot=False, is_generator=True), 
                Sample('HN3L_M_3_V_0p22360679775_e_massiveAndCKM_LO'                 , ['HN3L_M_3_V_0p22360679775_e_massiveAndCKM_LO'                 ], channel, '#splitline{m= 3 GeV, |V|^{2}=5.0 10^{-2}}{Majorana}' , selection, 'hnl_m_3_v2_5p0Em02_majorana' ,  'firebrick'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       202.1, toplot=False), 
                Sample('HN3L_M_3_V_0p707106781187_e_massiveAndCKM_LO'                , ['HN3L_M_3_V_0p707106781187_e_massiveAndCKM_LO'                ], channel, '#splitline{m= 3 GeV, |V|^{2}=5.0 10^{-1}}{Majorana}' , selection, 'hnl_m_3_v2_5p0Em01_majorana' ,  'firebrick'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      2019.0, toplot=False), 
                Sample('HN3L_M_4_V_0p00290516780927_e_massiveAndCKM_LO'              , ['HN3L_M_4_V_0p00290516780927_e_massiveAndCKM_LO'              ], channel, '#splitline{m= 4 GeV, |V|^{2}=8.4 10^{-6}}{Majorana}' , selection, 'hnl_m_4_v2_8p4Em06_majorana' ,  'indigo'       , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.03365, toplot=False, is_generator=True), 
                Sample('HN3L_M_4_V_0p0707106781187_e_massiveAndCKM_LO'               , ['HN3L_M_4_V_0p0707106781187_e_massiveAndCKM_LO'               ], channel, '#splitline{m= 4 GeV, |V|^{2}=5.0 10^{-3}}{Majorana}' , selection, 'hnl_m_4_v2_5p0Em03_majorana' ,  'indigo'       , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       19.91, toplot=False), 
                Sample('HN3L_M_4_V_0p22360679775_e_massiveAndCKM_LO'                 , ['HN3L_M_4_V_0p22360679775_e_massiveAndCKM_LO'                 ], channel, '#splitline{m= 4 GeV, |V|^{2}=5.0 10^{-2}}{Majorana}' , selection, 'hnl_m_4_v2_5p0Em02_majorana' ,  'indigo'       , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       198.9, toplot=False), 
                Sample('HN3L_M_5_V_0p000316227766017_e_massiveAndCKM_LO'             , ['HN3L_M_5_V_0p000316227766017_e_massiveAndCKM_LO'             ], channel, '#splitline{m= 5 GeV, |V|^{2}=1.0 10^{-7}}{Majorana}' , selection, 'hnl_m_5_v2_1p0Em07_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,   0.0003987, toplot=False), 
#                 Sample('HN3L_M_5_V_0p000547722557505_e_massiveAndCKM_LO'             , ['HN3L_M_5_V_0p000547722557505_e_massiveAndCKM_LO'             ], channel, '#splitline{m= 5 GeV, |V|^{2}=3.0 10^{-7}}{Majorana}' , selection, 'hnl_m_5_v2_3p0Em07_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.001195, toplot=False), 
                Sample('HN3L_M_5_V_0p001_e_massiveAndCKM_LO'                         , ['HN3L_M_5_V_0p001_e_massiveAndCKM_LO'                         ], channel, '#splitline{m= 5 GeV, |V|^{2}=1.0 10^{-6}}{Majorana}' , selection, 'hnl_m_5_v2_1p0Em06_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.00399, toplot=False), 
                Sample('HN3L_M_5_V_0p00145602197786_e_massiveAndCKM_LO'              , ['HN3L_M_5_V_0p00145602197786_e_massiveAndCKM_LO'              ], channel, '#splitline{m= 5 GeV, |V|^{2}=2.1 10^{-6}}{Majorana}' , selection, 'hnl_m_5_v2_2p1Em06_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.008479, toplot=False, is_generator=True), 
                Sample('HN3L_M_5_V_0p0707106781187_e_massiveAndCKM_LO'               , ['HN3L_M_5_V_0p0707106781187_e_massiveAndCKM_LO'               ], channel, '#splitline{m= 5 GeV, |V|^{2}=5.0 10^{-3}}{Majorana}' , selection, 'hnl_m_5_v2_5p0Em03_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,        20.0, toplot=False), 
                Sample('HN3L_M_5_V_0p22360679775_e_massiveAndCKM_LO'                 , ['HN3L_M_5_V_0p22360679775_e_massiveAndCKM_LO'                 ], channel, '#splitline{m= 5 GeV, |V|^{2}=5.0 10^{-2}}{Majorana}' , selection, 'hnl_m_5_v2_5p0Em02_majorana' ,  'chocolate'    , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       199.8, toplot=False), 
                Sample('HN3L_M_6_V_0p00202484567313_e_massiveAndCKM_LO'              , ['HN3L_M_6_V_0p00202484567313_e_massiveAndCKM_LO'              ], channel, '#splitline{m= 6 GeV, |V|^{2}=4.1 10^{-6}}{Majorana}' , selection, 'hnl_m_6_v2_4p1Em06_majorana' ,  'olive'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.01655, toplot=False, is_generator=True), 
                Sample('HN3L_M_6_V_0p0316227766017_e_massiveAndCKM_LO'               , ['HN3L_M_6_V_0p0316227766017_e_massiveAndCKM_LO'               ], channel, '#splitline{m= 6 GeV, |V|^{2}=1.0 10^{-3}}{Majorana}' , selection, 'hnl_m_6_v2_1p0Em03_majorana' ,  'olive'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       4.038, toplot=False), 
                Sample('HN3L_M_6_V_0p1_e_massiveAndCKM_LO'                           , ['HN3L_M_6_V_0p1_e_massiveAndCKM_LO'                           ], channel, '#splitline{m= 6 GeV, |V|^{2}=1.0 10^{-2}}{Majorana}' , selection, 'hnl_m_6_v2_1p0Em02_majorana' ,  'olive'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       40.29, toplot=False), 
                Sample('HN3L_M_7_V_0p0022361_e_massiveAndCKM_LO'                     , ['HN3L_M_7_V_0p0022361_e_massiveAndCKM_LO'                     ], channel, '#splitline{m= 7 GeV, |V|^{2}=5.0 10^{-6}}{Majorana}' , selection, 'hnl_m_7_v2_5p0Em06_majorana' ,  'peru'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.02035, toplot=False, is_generator=True), 
                Sample('HN3L_M_7_V_0p0316227766017_e_massiveAndCKM_LO'               , ['HN3L_M_7_V_0p0316227766017_e_massiveAndCKM_LO'               ], channel, '#splitline{m= 7 GeV, |V|^{2}=1.0 10^{-3}}{Majorana}' , selection, 'hnl_m_7_v2_1p0Em03_majorana' ,  'peru'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,        4.07, toplot=False), 
                Sample('HN3L_M_7_V_0p1_e_massiveAndCKM_LO'                           , ['HN3L_M_7_V_0p1_e_massiveAndCKM_LO'                           ], channel, '#splitline{m= 7 GeV, |V|^{2}=1.0 10^{-2}}{Majorana}' , selection, 'hnl_m_7_v2_1p0Em02_majorana' ,  'peru'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       40.74, toplot=False), 
                Sample('HN3L_M_8_V_0p000316227766017_e_massiveAndCKM_LO'             , ['HN3L_M_8_V_0p000316227766017_e_massiveAndCKM_LO'             ], channel, '#splitline{m= 8 GeV, |V|^{2}=1.0 10^{-7}}{Majorana}' , selection, 'hnl_m_8_v2_1p0Em07_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,   0.0004096, toplot=False), 
                Sample('HN3L_M_8_V_0p000547722557505_e_massiveAndCKM_LO'             , ['HN3L_M_8_V_0p000547722557505_e_massiveAndCKM_LO'             ], channel, '#splitline{m= 8 GeV, |V|^{2}=3.0 10^{-7}}{Majorana}' , selection, 'hnl_m_8_v2_3p0Em07_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.001228, toplot=False), 
                Sample('HN3L_M_8_V_0p001_e_massiveAndCKM_LO'                         , ['HN3L_M_8_V_0p001_e_massiveAndCKM_LO'                         ], channel, '#splitline{m= 8 GeV, |V|^{2}=1.0 10^{-6}}{Majorana}' , selection, 'hnl_m_8_v2_1p0Em06_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.004096, toplot=False), 
                Sample('HN3L_M_8_V_0p00151327459504_e_massiveAndCKM_LO'              , ['HN3L_M_8_V_0p00151327459504_e_massiveAndCKM_LO'              ], channel, '#splitline{m= 8 GeV, |V|^{2}=2.3 10^{-6}}{Majorana}' , selection, 'hnl_m_8_v2_2p3Em06_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.009383, toplot=False, is_generator=True), 
                Sample('HN3L_M_8_V_0p0022360679775_e_massiveAndCKM_LO'               , ['HN3L_M_8_V_0p0022360679775_e_massiveAndCKM_LO'               ], channel, '#splitline{m= 8 GeV, |V|^{2}=5.0 10^{-6}}{Majorana}' , selection, 'hnl_m_8_v2_5p0Em06_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.02046, toplot=True), 
                Sample('HN3L_M_8_V_0p0316227766017_e_massiveAndCKM_LO'               , ['HN3L_M_8_V_0p0316227766017_e_massiveAndCKM_LO'               ], channel, '#splitline{m= 8 GeV, |V|^{2}=1.0 10^{-3}}{Majorana}' , selection, 'hnl_m_8_v2_1p0Em03_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       4.095, toplot=False), 
                Sample('HN3L_M_8_V_0p1_e_massiveAndCKM_LO'                           , ['HN3L_M_8_V_0p1_e_massiveAndCKM_LO'                           ], channel, '#splitline{m= 8 GeV, |V|^{2}=1.0 10^{-2}}{Majorana}' , selection, 'hnl_m_8_v2_1p0Em02_majorana' ,  'darkgray'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       40.94, toplot=False), 
                Sample('HN3L_M_9_V_0p00316227766017_e_massiveAndCKM_LO'              , ['HN3L_M_9_V_0p00316227766017_e_massiveAndCKM_LO'              ], channel, '#splitline{m= 9 GeV, |V|^{2}=1.0 10^{-5}}{Majorana}' , selection, 'hnl_m_9_v2_1p0Em05_majorana' ,  'plum'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.04102, toplot=False, is_generator=True), 
                Sample('HN3L_M_9_V_0p0316227766017_e_massiveAndCKM_LO'               , ['HN3L_M_9_V_0p0316227766017_e_massiveAndCKM_LO'               ], channel, '#splitline{m= 9 GeV, |V|^{2}=1.0 10^{-3}}{Majorana}' , selection, 'hnl_m_9_v2_1p0Em03_majorana' ,  'plum'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,       4.118, toplot=False), 
                Sample('HN3L_M_10_V_0p000316227766017_e_massiveAndCKM_LO'            , ['HN3L_M_10_V_0p000316227766017_e_massiveAndCKM_LO'            ], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-7}}{Majorana}' , selection, 'hnl_m_10_v2_1p0Em07_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,   0.0004118, toplot=False), 
                Sample('HN3L_M_10_V_0p000547722557505_e_massiveAndCKM_LO'            , ['HN3L_M_10_V_0p000547722557505_e_massiveAndCKM_LO'            ], channel, '#splitline{m=10 GeV, |V|^{2}=3.0 10^{-7}}{Majorana}' , selection, 'hnl_m_10_v2_3p0Em07_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.001237, toplot=False), 
                Sample('HN3L_M_10_V_0p000756967634711_e_massiveAndCKM_LO'            , ['HN3L_M_10_V_0p000756967634711_e_massiveAndCKM_LO'            ], channel, '#splitline{m=10 GeV, |V|^{2}=5.7 10^{-7}}{Majorana}' , selection, 'hnl_m_10_v2_5p7Em07_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.002366, toplot=False), 
                Sample('HN3L_M_10_V_0p001_e_massiveAndCKM_LO'                        , ['HN3L_M_10_V_0p001_e_massiveAndCKM_LO'                        ], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-6}}{Majorana}' , selection, 'hnl_m_10_v2_1p0Em06_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.004114, toplot=False, is_generator=True), 
                Sample('HN3L_M_10_V_0p00316227766017_e_massiveAndCKM_LO'             , ['HN3L_M_10_V_0p00316227766017_e_massiveAndCKM_LO'             ], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-5}}{Majorana}' , selection, 'hnl_m_10_v2_1p0Em05_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.04116, toplot=False), 
                Sample('HN3L_M_10_V_0p01_e_massiveAndCKM_LO'                         , ['HN3L_M_10_V_0p01_e_massiveAndCKM_LO'                         ], channel, '#splitline{m=10 GeV, |V|^{2}=1.0 10^{-4}}{Majorana}' , selection, 'hnl_m_10_v2_1p0Em04_majorana',  'teal'         , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.4122, toplot=False), 
                Sample('HN3L_M_11_V_0p00316227766017_e_massiveAndCKM_LO'             , ['HN3L_M_11_V_0p00316227766017_e_massiveAndCKM_LO'             ], channel, '#splitline{m=11 GeV, |V|^{2}=1.0 10^{-5}}{Majorana}' , selection, 'hnl_m_11_v2_1p0Em05_majorana',  'seagreen'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.0412, toplot=False, is_generator=True), 
                Sample('HN3L_M_11_V_0p01_e_massiveAndCKM_LO'                         , ['HN3L_M_11_V_0p01_e_massiveAndCKM_LO'                         ], channel, '#splitline{m=11 GeV, |V|^{2}=1.0 10^{-4}}{Majorana}' , selection, 'hnl_m_11_v2_1p0Em04_majorana',  'seagreen'     , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.4117, toplot=False), 
                Sample('HN3L_M_12_V_0p00316227766017_e_massiveAndCKM_LO'             , ['HN3L_M_12_V_0p00316227766017_e_massiveAndCKM_LO'             ], channel, '#splitline{m=12 GeV, |V|^{2}=1.0 10^{-5}}{Majorana}' , selection, 'hnl_m_12_v2_1p0Em05_majorana',  'coral'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.04113, toplot=False, is_generator=True), 
                Sample('HN3L_M_12_V_0p01_e_massiveAndCKM_LO'                         , ['HN3L_M_12_V_0p01_e_massiveAndCKM_LO'                         ], channel, '#splitline{m=12 GeV, |V|^{2}=1.0 10^{-4}}{Majorana}' , selection, 'hnl_m_12_v2_1p0Em04_majorana',  'coral'        , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.4113, toplot=True), 
                Sample('HN3L_M_20_V_0p001_e_massiveAndCKM_LO'                        , ['HN3L_M_20_V_0p001_e_massiveAndCKM_LO'                        ], channel, '#splitline{m=20 GeV, |V|^{2}=1.0 10^{-6}}{Majorana}' , selection, 'hnl_m_20_v2_1p0Em06_majorana',  'crimson'      , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,    0.003853, toplot=False), 
                Sample('HN3L_M_20_V_0p00316227766017_e_massiveAndCKM_LO'             , ['HN3L_M_20_V_0p00316227766017_e_massiveAndCKM_LO'             ], channel, '#splitline{m=20 GeV, |V|^{2}=1.0 10^{-5}}{Majorana}' , selection, 'hnl_m_20_v2_1p0Em05_majorana',  'crimson'      , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,     0.03856, toplot=False, is_generator=True), 
                Sample('HN3L_M_20_V_0p01_e_massiveAndCKM_LO'                         , ['HN3L_M_20_V_0p01_e_massiveAndCKM_LO'                         ], channel, '#splitline{m=20 GeV, |V|^{2}=1.0 10^{-4}}{Majorana}' , selection, 'hnl_m_20_v2_1p0Em04_majorana',  'crimson'      , 10, '/'.join([basedir, 'sig']), postfix, False, True, True, 1.,      0.3854, toplot=False), 
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
            merged_sample.xs            = np.mean([isample.xs for isample in weighed_samples])
            merged_sample.nevents       = np.sum([isample.nevents for isample in weighed_samples])
            merged_sample.lumi_scaling  = merged_sample.xs / merged_sample.nevents
        
            merged_signals.append(merged_sample)

            # free up some memory
            for already_merged in weighed_samples:
                signal.remove(already_merged)
                del already_merged
                gc.collect()
        
        signal = merged_signals
            
    return signal

if __name__ == '__main__':

    # Study using all signal for signal reweighing
    plot_dir = '/'.join([env['BASE_DIR'], 'plotter', 'plots', '2018']) 
    base_dir = '/'.join([env['BASE_DIR'], 'ntuples', 'may20', '2018'])
    signals = get_signal_samples('mmm', base_dir, 'HNLTreeProducer_mmm/tree.root', '1', mini=False)

    for i, isig in enumerate(signals): print(i, isig.datacard_name, isig.name)

#     signals =  [isample for isample in signals if isample.name.startswith('HN3L_M_2_V_')]
#     0   hnl_m_2_v2_1p2Em04_majorana HN3L_M_2_V_0p0110905365064_mu_massiveAndCKM_LO
#     1   hnl_m_2_v2_6p2Em04_majorana HN3L_M_2_V_0p0248394846967_mu_massiveAndCKM_LO
#     2   hnl_m_2_v2_5p0Em02_majorana HN3L_M_2_V_0p22360679775_mu_massiveAndCKM_LO
#     3   hnl_m_2_v2_5p0Em01_majorana HN3L_M_2_V_0p707106781187_mu_massiveAndCKM_LO
#     14  hnl_m_2_v2_1p0Em04_majorana HN3L_M_2_V_0p0110905365064_mu_massiveAndCKM_LO
#     60  hnl_m_2_v2_1p0Em04_majorana HN3L_M_2_V_0p0248394846967_mu_massiveAndCKM_LO
#     106 hnl_m_2_v2_1p0Em04_majorana HN3L_M_2_V_0p22360679775_mu_massiveAndCKM_LO
#     152 hnl_m_2_v2_1p0Em04_majorana HN3L_M_2_V_0p707106781187_mu_massiveAndCKM_LO
    
#     ss = OrderedDict()
#     ss['1p2em4_orig'      ] = signals[  0]
#     ss['6p2em4_orig'      ] = signals[  1]
#     ss['5p0em2_orig'      ] = signals[  2]
#     ss['5p0em1_orig'      ] = signals[  3]
#     ss['1p0em4_from1p2em4'] = signals[ 14]
#     ss['1p0em4_from6p2em4'] = signals[ 60]
#     ss['1p0em4_from5p0em2'] = signals[106]
#     ss['1p0em4_from5p0em1'] = signals[152]
    
#     print(ss['1p0em4_from1p2em4'].xs * ss['1p2em4_orig'].df['xs_w_v2_1.0em04'][0])
#     print(ss['1p0em4_from6p2em4'].xs * ss['6p2em4_orig'].df['xs_w_v2_1.0em04'][0])
#     print(ss['1p0em4_from5p0em2'].xs * ss['5p0em2_orig'].df['xs_w_v2_1.0em04'][0])
#     print(ss['1p0em4_from5p0em1'].xs * ss['5p0em1_orig'].df['xs_w_v2_1.0em04'][0])

    # the cross sections must be all the same
#     print(ss['1p0em4_from1p2em4'].xs)
#     print(ss['1p0em4_from6p2em4'].xs)
#     print(ss['1p0em4_from5p0em2'].xs)
#     print(ss['1p0em4_from5p0em1'].xs)
    
    # now let's group the samples by their datacard name
    datacard_names = set([isample.datacard_name for isample in signals])
    
    merged_signals = []
    for ii, iname in enumerate(datacard_names):
    
        print ('\t', iname, '\t%d/%d'%(ii+1, len(datacard_names)))
        weighed_samples = [isample for isample in signals if isample.datacard_name==iname]
        
        merged_sample               = copy(weighed_samples[0])
        merged_sample.datacard_name = iname
        merged_sample.df            = pd.concat([isample.df for isample in weighed_samples])
        merged_sample.xs            = np.mean([isample.xs for isample in weighed_samples])
        merged_sample.nevents       = np.sum([isample.nevents for isample in weighed_samples])
        merged_sample.lumi_scaling  = merged_sample.xs / merged_sample.nevents
        
        merged_signals.append(merged_sample)

        # free up some memory
        for already_merged in weighed_samples:
            signals.remove(already_merged)
            del already_merged
            gc.collect()
    
    
    from rootpy.plotting import Hist, HistStack, Legend, Canvas, Graph, Pad

#     Variable('hnl_2d_disp'        , np.linspace( 0   ,  30   , 25 + 1) , 'L_{xy} (cm)'                 , 'events'),
#     Variable('hnl_2d_disp'        , np.linspace( 0   ,  10   , 25 + 1) , 'L_{xy} (cm)'                 , 'events', extra_label='narrow'),
#     Variable('hnl_2d_disp'        , np.linspace( 0   ,   2   , 25 + 1) , 'L_{xy} (cm)'                 , 'events', extra_label='very_narrow'),
#     Variable('log_hnl_2d_disp'    , np.linspace( -1  ,   2   , 25 + 1) , 'log_{10}(L_{xy}) (cm)'       , 'events'),

    hnl_m_2_v2_1p2Em04_majorana = merged_signals[42]
    hnl_m_2_v2_1p0Em04_majorana = merged_signals[44]

    histo_tight = Hist(np.linspace(-1, 2, 25 + 1), title='log 10 2D disp', markersize=0, legendstyle='F')
    weights = hnl_m_2_v2_1p0Em04_majorana.lumi_scaling * 59700. * hnl_m_2_v2_1p0Em04_majorana.df['ctau_w_v2_1.0em04']
    histo_tight.fill_array(hnl_m_2_v2_1p0Em04_majorana.df['log_hnl_2d_disp'], weights=weights)

    # ====> 16437 entries
    # integral 97.74961962550879
    # mean 1.081
    # std 0.5336
    # average weight 0.005988899396367671

    histo_tight = Hist(np.linspace(-1, 2, 25 + 1), title='log 10 2D disp', markersize=0, legendstyle='F')
    weights = np.ones(hnl_m_2_v2_1p2Em04_majorana.df.shape[0]) * hnl_m_2_v2_1p2Em04_majorana.lumi_scaling * 59700.
    histo_tight.fill_array(hnl_m_2_v2_1p2Em04_majorana.df['log_hnl_2d_disp'], weights=weights)

    # ====> 1426 entries
    # integral 178.49046748876572
    # mean 1.148
    # std 0.5057
    # average weight 0.1261415715223623
