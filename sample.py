import re
import numpy as np
import pandas as pd
from root_pandas import read_root

class Sample(object):
    def __init__(self, 
                 name, 
                 channel,
                 label,
                 selection,
                 datacard_name,
                 colour,
                 position_in_stack, 
                 base_dir, 
                 post_fix, 
                 isdata, 
                 ismc, 
                 issignal, 
                 weight,
                 xs,
                 toplot=True):
        self.name              = name ; print('loading', self.name)
        self.channel           = channel
        self.label             = label   
        self.selection         = selection         
        self.datacard_name     = datacard_name            
        self.colour            = colour           
        self.position_in_stack = position_in_stack
        self.base_dir          = base_dir          
        self.post_fix          = post_fix          
        self.isdata            = isdata           
        self.ismc              = ismc             
        self.issignal          = issignal         
        self.weight            = weight           
        self.xs                = xs        
        self.nevents           = 1.
        self.file              = '/'.join([base_dir, self.name, post_fix])       
        self.toplot            = toplot   
        
        if not self.isdata:
            nevents_file = '/'.join([base_dir, self.name, 'SkimAnalyzerCount/SkimReport.txt'])
            with open(nevents_file) as ff:
                lines = ff.readlines()
                for line in lines:
                    if 'Sum Norm Weights' in line:
                        self.nevents = float(re.findall(r'\d+', lines[2])[0])
                        break
        tree_file = '/'.join([self.base_dir, self.name, self.post_fix])
        
        # selection = self.selection.replace('&', 'and').replace('|', 'or').replace('!', 'not') 
        # self.df = uproot.open(tree_file)['tree'].pandas.df().query(selection) # can't apply any selection with uproot...
        # self.df = pd.DataFrame( root2array(tree_file, 'tree', selection=self.selection) )
        self.df = read_root( tree_file, 'tree', where=self.selection )
        # scale to 1/pb 
        self.lumi_scaling = 1. if self.isdata else (self.xs / self.nevents)
 


def get_data_samples(channel, base_dir, post_fix, selection):
    if   channel [0] == 'm': lep = 'mu'
    elif channel [0] == 'e': lep = 'ele'
    assert lep == 'ele' or lep == 'mu', 'Lepton flavor error'
    data = [
        Sample('Single_{lep}_2018A'.format(lep=lep), channel, '2018A', selection, 'data_obs', 'black', 9999, base_dir, post_fix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2018B'.format(lep=lep), channel, '2018B', selection, 'data_obs', 'black', 9999, base_dir, post_fix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2018C'.format(lep=lep), channel, '2018C', selection, 'data_obs', 'black', 9999, base_dir, post_fix, True, False, False, 1., 1.),
        Sample('Single_{lep}_2018D'.format(lep=lep), channel, '2018D', selection, 'data_obs', 'black', 9999, base_dir, post_fix, True, False, False, 1., 1.),
    ]
    return data

def get_mc_samples(channel, base_dir, post_fix, selection):
    mc = [
        Sample('DYJetsToLL_M50_ext', channel,  r'DY$\to\ell\ell$', selection, 'DY', 'gold'     ,10, base_dir, post_fix, False, True, False, 1.,  6077.22),
        Sample('TTJets_ext'        , channel,  r'$t\bar{t}$'     , selection, 'TT', 'slateblue', 0, base_dir, post_fix, False, True, False, 1.,   831.76),
        Sample('WW'                , channel,  'WW'              , selection, 'WW', 'blue'     , 5, base_dir, post_fix, False, True, False, 1.,    75.88),
        Sample('WZ'                , channel,  'WZ'              , selection, 'WZ', 'blue'     , 5, base_dir, post_fix, False, True, False, 1.,    27.6 ),
        Sample('ZZ'                , channel,  'ZZ'              , selection, 'ZZ', 'blue'     , 5, base_dir, post_fix, False, True, False, 1.,    12.14),
    ]   
    return mc         

def get_signal_samples(channel, base_dir, post_fix, selection):
    assert channel[0] == 'e' or channel[0] == 'm', 'Lepton flavor error'
    if channel [0] == 'm': 
        signal = [ 
            Sample('HN3L_M_1_V_0p0949736805647_mu_massiveAndCKM_LO'             , channel, 'HNL m = 1, V^{2} = 9.0E-03, Majorana' , selection, 'hnl_m_1_v2_9p0Em03_majorana' ,
                   'darkorange' ,10, base_dir, post_fix, False, True, True, 1.,  38.67    , toplot=False),

            Sample('HN3L_M_1_V_0p13416407865_mu_Dirac_massiveAndCKM_LO'         , channel, 'HNL m = 1, V^{2} = 1.8E-02, Dirac'    , selection, 'hnl_m_1_v2_1p8Em02_dirac'    ,
                   'darkorange' ,10, base_dir, post_fix, False, True, True, 1.,  44.46    , toplot=False),

            Sample('HN3L_M_1_V_0p13416407865_mu_Dirac_cc_massiveAndCKM_LO'      , channel, 'HNL m = 1, V^{2} = 1.8E-02, Dirac cc' , selection, 'hnl_m_1_v2_1p8Em02_dirac_cc' ,
                   'darkorange' ,10, base_dir, post_fix, False, True, True, 1.,  33.21    , toplot=False),

#         Sample('HN3L_M_1_V_0p212367605816_mu_massiveAndCKM_LO'              , channel, 'HNL m = 1, V^{2} = 4.5E-02, Majorana' , selection, 'hnl_m_1_v2_4p5Em02_majorana' ,
               # 'darkorange' ,10, base_dir, post_fix, False, True, True, 1.,  193.3    , toplot=False),
         
            Sample('HN3L_M_1_V_0p300333148354_mu_Dirac_massiveAndCKM_LO'        , channel, 'HNL m = 1, V^{2} = 9.0E-02, Dirac'    , selection, 'hnl_m_1_v2_9p0Em02_dirac'    ,
                   'darkorange' ,10, base_dir, post_fix, False, True, True, 1.,  222.7    , toplot=False),

            Sample('HN3L_M_1_V_0p300333148354_mu_Dirac_cc_massiveAndCKM_LO'     , channel, 'HNL m = 1, V^{2} = 9.0E-02, Dirac cc' , selection, 'hnl_m_1_v2_9p0Em02_dirac_cc' ,
                   'darkorange' ,10, base_dir, post_fix, False, True, True, 1.,  166.9    , toplot=False),

            Sample('HN3L_M_2_V_0p0110905365064_mu_massiveAndCKM_LO'             , channel, 'HNL m = 2, V^{2} = 1.2E-04, Majorana' , selection, 'hnl_m_2_v2_1p2Em04_majorana' ,
                   'forestgreen',10, base_dir, post_fix, False, True, True, 1.,  0.5278   , toplot=True ),

            Sample('HN3L_M_2_V_0p0137840487521_mu_Dirac_cc_massiveAndCKM_LO'    , channel, 'HNL m = 2, V^{2} = 1.9E-04, Dirac cc' , selection, 'hnl_m_2_v2_1p9Em04_dirac_cc' ,
                   'forestgreen',10, base_dir, post_fix, False, True, True, 1.,  0.352    , toplot=False),

            Sample('HN3L_M_2_V_0p0157162336455_mu_Dirac_massiveAndCKM_LO'       , channel, 'HNL m = 2, V^{2} = 2.5E-04, Dirac'    , selection, 'hnl_m_2_v2_2p5Em04_dirac'    ,
                   'forestgreen',10, base_dir, post_fix, False, True, True, 1.,  0.6117   , toplot=False),

            Sample('HN3L_M_2_V_0p0157162336455_mu_Dirac_cc_massiveAndCKM_LO'    , channel, 'HNL m = 2, V^{2} = 2.5E-04, Dirac cc' , selection, 'hnl_m_2_v2_2p5Em04_dirac_cc' ,
                   'forestgreen',10, base_dir, post_fix, False, True, True, 1.,  0.458    , toplot=False),

            Sample('HN3L_M_2_V_0p0248394846967_mu_massiveAndCKM_LO'             , channel, 'HNL m = 2, V^{2} = 6.2E-04, Majorana' , selection, 'hnl_m_2_v2_6p2Em04_majorana' ,
                   'forestgreen',10, base_dir, post_fix, False, True, True, 1.,  2.647    , toplot=False),

            Sample('HN3L_M_2_V_0p0307896086367_mu_Dirac_cc_massiveAndCKM_LO'    , channel, 'HNL m = 2, V^{2} = 9.5E-04, Dirac cc' , selection, 'hnl_m_2_v2_9p5Em04_dirac_cc' ,
                   'forestgreen',10, base_dir, post_fix, False, True, True, 1.,  1.75     , toplot=False),

            Sample('HN3L_M_2_V_0p0350713558335_mu_Dirac_massiveAndCKM_LO'       , channel, 'HNL m = 2, V^{2} = 1.2E-03, Dirac'    , selection, 'hnl_m_2_v2_1p2Em03_dirac'    ,
                   'forestgreen',10, base_dir, post_fix, False, True, True, 1.,  3.047    , toplot=False),

            Sample('HN3L_M_3_V_0p00443846820423_mu_Dirac_cc_massiveAndCKM_LO'   , channel, 'HNL m = 3, V^{2} = 2.0E-05, Dirac cc' , selection, 'hnl_m_3_v2_2p0Em05_dirac_cc' ,
                   'firebrick'  ,10, base_dir, post_fix, False, True, True, 1.,  0.03459  , toplot=True ),

            Sample('HN3L_M_3_V_0p00707813534767_mu_massiveAndCKM_LO'            , channel, 'HNL m = 3, V^{2} = 5.0E-05, Majorana' , selection, 'hnl_m_3_v2_5p0Em05_majorana' ,
                   'firebrick'  ,10, base_dir, post_fix, False, True, True, 1.,  0.2014   , toplot=False),

            Sample('HN3L_M_3_V_0p01_mu_Dirac_massiveAndCKM_LO'                  , channel, 'HNL m = 3, V^{2} = 1.0E-04, Dirac'    , selection, 'hnl_m_3_v2_1p0Em04_dirac'    ,
                   'firebrick'  ,10, base_dir, post_fix, False, True, True, 1.,  0.233    , toplot=False),

            Sample('HN3L_M_3_V_0p0140356688476_mu_Dirac_cc_massiveAndCKM_LO'    , channel, 'HNL m = 3, V^{2} = 2.0E-04, Dirac cc' , selection, 'hnl_m_3_v2_2p0Em04_dirac_cc' ,
                   'firebrick'  ,10, base_dir, post_fix, False, True, True, 1.,  0.3434   , toplot=False),

            Sample('HN3L_M_4_V_0p00183575597507_mu_Dirac_cc_massiveAndCKM_LO'   , channel, 'HNL m = 4, V^{2} = 3.4E-06, Dirac cc' , selection, 'hnl_m_4_v2_3p4Em06_dirac_cc' ,
                   'indigo'     ,10, base_dir, post_fix, False, True, True, 1.,  0.005818 , toplot=False),

            Sample('HN3L_M_4_V_0p00290516780927_mu_massiveAndCKM_LO'            , channel, 'HNL m = 4, V^{2} = 8.4E-06, Majorana' , selection, 'hnl_m_4_v2_8p4Em06_majorana' ,
                   'indigo'     ,10, base_dir, post_fix, False, True, True, 1.,  0.0335   , toplot=False),

            Sample('HN3L_M_4_V_0p00354964786986_mu_Dirac_cc_massiveAndCKM_LO'   , channel, 'HNL m = 4, V^{2} = 1.3E-05, Dirac cc' , selection, 'hnl_m_4_v2_1p3Em05_dirac_cc' ,
                   'indigo'     ,10, base_dir, post_fix, False, True, True, 1.,  0.02173  , toplot=False),

            Sample('HN3L_M_4_V_0p00411096095822_mu_Dirac_massiveAndCKM_LO'      , channel, 'HNL m = 4, V^{2} = 1.7E-05, Dirac'    , selection, 'hnl_m_4_v2_1p7Em05_dirac'    ,
                   'indigo'     ,10, base_dir, post_fix, False, True, True, 1.,  0.03904  , toplot=False),

            Sample('HN3L_M_4_V_0p0101980390272_mu_Dirac_cc_massiveAndCKM_LO'    , channel, 'HNL m = 4, V^{2} = 1.0E-04, Dirac cc' , selection, 'hnl_m_4_v2_1p0Em04_dirac_cc' ,
                   'indigo'     ,10, base_dir, post_fix, False, True, True, 1.,  0.18     , toplot=False),

            Sample('HN3L_M_5_V_0p000316227766017_mu_Dirac_massiveAndCKM_LO'     , channel, 'HNL m = 5, V^{2} = 1.0E-07, Dirac'    , selection, 'hnl_m_5_v2_1p0Em07_dirac'    ,
                   'chocolate'  ,10, base_dir, post_fix, False, True, True, 1.,  0.0002326, toplot=False),

            Sample('HN3L_M_5_V_0p000316227766017_mu_massiveAndCKM_LO'           , channel, 'HNL m = 5, V^{2} = 1.0E-07, Majorana' , selection, 'hnl_m_5_v2_1p0Em07_majorana' ,
                   'chocolate'  ,10, base_dir, post_fix, False, True, True, 1.,  0.0003981, toplot=False),

            Sample('HN3L_M_5_V_0p000547722557505_mu_Dirac_massiveAndCKM_LO'     , channel, 'HNL m = 5, V^{2} = 3.0E-07, Dirac'    , selection, 'hnl_m_5_v2_3p0Em07_dirac'    ,
                   'chocolate'  ,10, base_dir, post_fix, False, True, True, 1.,  0.0006979, toplot=False),

            Sample('HN3L_M_5_V_0p000547722557505_mu_massiveAndCKM_LO'           , channel, 'HNL m = 5, V^{2} = 3.0E-07, Majorana' , selection, 'hnl_m_5_v2_3p0Em07_majorana' ,
                   'chocolate'  ,10, base_dir, post_fix, False, True, True, 1.,  0.001194 , toplot=False),

            Sample('HN3L_M_5_V_0p000920326029187_mu_Dirac_cc_massiveAndCKM_LO'  , channel, 'HNL m = 5, V^{2} = 8.5E-07, Dirac cc' , selection, 'hnl_m_5_v2_8p5Em07_dirac_cc' ,
                   'chocolate'  ,10, base_dir, post_fix, False, True, True, 1.,  0.001473 , toplot=False),

#         Sample('HN3L_M_5_V_0p001_mu_Dirac_massiveAndCKM_LO'                 , channel, 'HNL m = 5, V^{2} = 1.0E-06, Dirac'    , selection, 'hnl_m_5_v2_1p0Em06_dirac'    ,
               # 'chocolate'  ,10, base_dir, post_fix, False, True, True, 1.,  0.002324 , toplot=False),

#         Sample('HN3L_M_5_V_0p001_mu_massiveAndCKM_LO'                       , channel, 'HNL m = 5, V^{2} = 1.0E-06, Majorana' , selection, 'hnl_m_5_v2_1p0Em06_majorana' ,
               # 'chocolate'  ,10, base_dir, post_fix, False, True, True, 1.,  0.003977 , toplot=False),

            Sample('HN3L_M_5_V_0p00145602197786_mu_massiveAndCKM_LO'            , channel, 'HNL m = 5, V^{2} = 2.1E-06, Majorana' , selection, 'hnl_m_5_v2_2p1Em06_majorana' ,
                   'chocolate'  ,10, base_dir, post_fix, False, True, True, 1.,  0.008434 , toplot=True ),

            Sample('HN3L_M_5_V_0p00178044938148_mu_Dirac_cc_massiveAndCKM_LO'   , channel, 'HNL m = 5, V^{2} = 3.2E-06, Dirac cc' , selection, 'hnl_m_5_v2_3p2Em06_dirac_cc' ,
                   'chocolate'  ,10, base_dir, post_fix, False, True, True, 1.,  0.005518 , toplot=False),

            Sample('HN3L_M_5_V_0p00205669638012_mu_Dirac_massiveAndCKM_LO'      , channel, 'HNL m = 5, V^{2} = 4.2E-06, Dirac'    , selection, 'hnl_m_5_v2_4p2Em06_dirac'    ,
                   'chocolate'  ,10, base_dir, post_fix, False, True, True, 1.,  0.009836 , toplot=False),

#         Sample('HN3L_M_5_V_0p0065574385243_mu_Dirac_cc_massiveAndCKM_LO'    , channel, 'HNL m = 5, V^{2} = 4.3E-05, Dirac cc' , selection, 'hnl_m_5_v2_4p3Em05_dirac_cc' ,
               # 'chocolate'  ,10, base_dir, post_fix, False, True, True, 1.,  0.07522  , toplot=False),

            Sample('HN3L_M_6_V_0p000522494019105_mu_Dirac_cc_massiveAndCKM_LO'  , channel, 'HNL m = 6, V^{2} = 2.7E-07, Dirac cc' , selection, 'hnl_m_6_v2_2p7Em07_dirac_cc' ,
                   'olive'      ,10, base_dir, post_fix, False, True, True, 1.,  0.0004745, toplot=False),

            Sample('HN3L_M_6_V_0p00101488915651_mu_Dirac_cc_massiveAndCKM_LO'   , channel, 'HNL m = 6, V^{2} = 1.0E-06, Dirac cc' , selection, 'hnl_m_6_v2_1p0Em06_dirac_cc' ,
                   'olive'      ,10, base_dir, post_fix, False, True, True, 1.,  0.001795 , toplot=False),

            Sample('HN3L_M_6_V_0p00202484567313_mu_massiveAndCKM_LO'            , channel, 'HNL m = 6, V^{2} = 4.1E-06, Majorana' , selection, 'hnl_m_6_v2_4p1Em06_majorana' ,
                   'olive'      ,10, base_dir, post_fix, False, True, True, 1.,  0.01655  , toplot=False),

            Sample('HN3L_M_6_V_0p00286356421266_mu_Dirac_massiveAndCKM_LO'      , channel, 'HNL m = 6, V^{2} = 8.2E-06, Dirac'    , selection, 'hnl_m_6_v2_8p2Em06_dirac'    ,
                   'olive'      ,10, base_dir, post_fix, False, True, True, 1.,  0.01926  , toplot=False),

            Sample('HN3L_M_6_V_0p00299666481275_mu_Dirac_cc_massiveAndCKM_LO'   , channel, 'HNL m = 6, V^{2} = 9.0E-06, Dirac cc' , selection, 'hnl_m_6_v2_9p0Em06_dirac_cc' ,
                   'olive'      ,10, base_dir, post_fix, False, True, True, 1.,  0.01568  , toplot=False),

            Sample('HN3L_M_8_V_0p000316227766017_mu_Dirac_massiveAndCKM_LO'     , channel, 'HNL m = 8, V^{2} = 1.0E-07, Dirac'    , selection, 'hnl_m_8_v2_1p0Em07_dirac'    ,
                   'darkgray'   ,10, base_dir, post_fix, False, True, True, 1.,  0.0002387, toplot=False),

            Sample('HN3L_M_8_V_0p000415932686862_mu_Dirac_cc_massiveAndCKM_LO'  , channel, 'HNL m = 8, V^{2} = 1.7E-07, Dirac cc' , selection, 'hnl_m_8_v2_1p7Em07_dirac_cc' ,
                   'darkgray'   ,10, base_dir, post_fix, False, True, True, 1.,  0.0003071, toplot=False),

            Sample('HN3L_M_8_V_0p000547722557505_mu_Dirac_massiveAndCKM_LO'     , channel, 'HNL m = 8, V^{2} = 3.0E-07, Dirac'    , selection, 'hnl_m_8_v2_3p0Em07_dirac'    ,
                   'darkgray'   ,10, base_dir, post_fix, False, True, True, 1.,  0.0007165, toplot=False),

            Sample('HN3L_M_8_V_0p000547722557505_mu_massiveAndCKM_LO'           , channel, 'HNL m = 8, V^{2} = 3.0E-07, Majorana' , selection, 'hnl_m_8_v2_3p0Em07_majorana' ,
                   'darkgray'   ,10, base_dir, post_fix, False, True, True, 1.,  0.00123  , toplot=False),

            Sample('HN3L_M_8_V_0p001_mu_Dirac_massiveAndCKM_LO'                 , channel, 'HNL m = 8, V^{2} = 1.0E-06, Dirac'    , selection, 'hnl_m_8_v2_1p0Em06_dirac'    ,
                   'darkgray'   ,10, base_dir, post_fix, False, True, True, 1.,  0.002389 , toplot=False),

#         Sample('HN3L_M_8_V_0p001_mu_massiveAndCKM_LO'                       , channel, 'HNL m = 8, V^{2} = 1.0E-06, Majorana' , selection, 'hnl_m_8_v2_1p0Em06_majorana' ,
               # 'darkgray'   ,10, base_dir, post_fix, False, True, True, 1.,  0.004104 , toplot=False),

#         Sample('HN3L_M_8_V_0p00151327459504_mu_massiveAndCKM_LO'            , channel, 'HNL m = 8, V^{2} = 2.3E-06, Majorana' , selection, 'hnl_m_8_v2_2p3Em06_majorana' ,
               # 'darkgray'   ,10, base_dir, post_fix, False, True, True, 1.,  0.009374 , toplot=False),

#         Sample('HN3L_M_8_V_0p00214242852856_mu_Dirac_massiveAndCKM_LO'      , channel, 'HNL m = 8, V^{2} = 4.6E-06, Dirac'    , selection, 'hnl_m_8_v2_4p6Em06_dirac'    ,
               # 'darkgray'   ,10, base_dir, post_fix, False, True, True, 1.,  0.01096  , toplot=False),

            Sample('HN3L_M_8_V_0p00363318042492_mu_Dirac_cc_massiveAndCKM_LO'   , channel, 'HNL m = 8, V^{2} = 1.3E-05, Dirac cc' , selection, 'hnl_m_8_v2_1p3Em05_dirac_cc' ,
                   'darkgray'   ,10, base_dir, post_fix, False, True, True, 1.,  0.02351  , toplot=False),

            Sample('HN3L_M_10_V_0p000208566536146_mu_Dirac_cc_massiveAndCKM_LO' , channel, 'HNL m = 10, V^{2} = 4.3E-08, Dirac cc', selection, 'hnl_m_10_v2_4p3Em08_dirac_cc',
                   'teal'       ,10, base_dir, post_fix, False, True, True, 1.,  7.797e-05, toplot=False),

            Sample('HN3L_M_10_V_0p000316227766017_mu_Dirac_massiveAndCKM_LO'    , channel, 'HNL m = 10, V^{2} = 1.0E-07, Dirac'   , selection, 'hnl_m_10_v2_1p0Em07_dirac'   ,
                   'teal'       ,10, base_dir, post_fix, False, True, True, 1.,  0.0002398, toplot=False),

#         Sample('HN3L_M_10_V_0p000316227766017_mu_massiveAndCKM_LO'          , channel, 'HNL m = 10, V^{2} = 1.0E-07, Majorana', selection, 'hnl_m_10_v2_1p0Em07_majorana',
               # 'teal'       ,10, base_dir, post_fix, False, True, True, 1.,  0.0004118, toplot=False),

            Sample('HN3L_M_10_V_0p000547722557505_mu_Dirac_massiveAndCKM_LO'    , channel, 'HNL m = 10, V^{2} = 3.0E-07, Dirac'   , selection, 'hnl_m_10_v2_3p0Em07_dirac'   ,
                   'teal'       ,10, base_dir, post_fix, False, True, True, 1.,  0.0007193, toplot=False),

            Sample('HN3L_M_10_V_0p000547722557505_mu_massiveAndCKM_LO'          , channel, 'HNL m = 10, V^{2} = 3.0E-07, Majorana', selection, 'hnl_m_10_v2_3p0Em07_majorana',
                   'teal'       ,10, base_dir, post_fix, False, True, True, 1.,  0.001237 , toplot=False),

            Sample('HN3L_M_10_V_0p000756967634711_mu_massiveAndCKM_LO'          , channel, 'HNL m = 10, V^{2} = 5.7E-07, Majorana', selection, 'hnl_m_10_v2_5p7Em07_majorana',
                   'teal'       ,10, base_dir, post_fix, False, True, True, 1.,  0.002362 , toplot=False),

            Sample('HN3L_M_10_V_0p001_mu_Dirac_massiveAndCKM_LO'                , channel, 'HNL m = 10, V^{2} = 1.0E-06, Dirac'   , selection, 'hnl_m_10_v2_1p0Em06_dirac'   ,
                   'teal'       ,10, base_dir, post_fix, False, True, True, 1.,  0.002405 , toplot=False),

            Sample('HN3L_M_10_V_0p001_mu_massiveAndCKM_LO'                      , channel, 'HNL m = 10, V^{2} = 1.0E-06, Majorana', selection, 'hnl_m_10_v2_1p0Em06_majorana',
                   'teal'       ,10, base_dir, post_fix, False, True, True, 1.,  0.004121 , toplot=True ),

            Sample('HN3L_M_10_V_0p00107238052948_mu_Dirac_massiveAndCKM_LO'     , channel, 'HNL m = 10, V^{2} = 1.2E-06, Dirac'   , selection, 'hnl_m_10_v2_1p2Em06_dirac'   ,
                   'teal'       ,10, base_dir, post_fix, False, True, True, 1.,  0.002761 , toplot=False),

            Sample('HN3L_M_10_V_0p00112249721603_mu_Dirac_cc_massiveAndCKM_LO'  , channel, 'HNL m = 10, V^{2} = 1.3E-06, Dirac cc', selection, 'hnl_m_10_v2_1p3Em06_dirac_cc',
                   'teal'       ,10, base_dir, post_fix, False, True, True, 1.,  0.00227  , toplot=False),

            Sample('HN3L_M_15_V_0p00003021588986_mu_Dirac_cc_massiveAndCKM_LO'  , channel, 'HNL m = 15, V^{2} = 9.1E-10, Dirac cc', selection, 'hnl_m_15_v2_9p1Em10_dirac_cc',
                   'gold'       ,10, base_dir, post_fix, False, True, True, 1.,  1.606e-06, toplot=False),

            Sample('HN3L_M_15_V_0p00006760177512_mu_Dirac_cc_massiveAndCKM_LO'  , channel, 'HNL m = 15, V^{2} = 4.6E-09, Dirac cc', selection, 'hnl_m_15_v2_4p6Em09_dirac_cc',
                   'gold'       ,10, base_dir, post_fix, False, True, True, 1.,  8.068e-06, toplot=False),

            Sample('HN3L_M_20_V_0p00001224744871_mu_Dirac_cc_massiveAndCKM_LO'  , channel, 'HNL m = 20, V^{2} = 1.5E-10, Dirac cc', selection, 'hnl_m_20_v2_1p5Em10_dirac_cc',
                   'crimson'    ,10, base_dir, post_fix, False, True, True, 1.,  2.524e-07, toplot=False),

            Sample('HN3L_M_20_V_0p00002734958866_mu_Dirac_cc_massiveAndCKM_LO'  , channel, 'HNL m = 20, V^{2} = 7.5E-10, Dirac cc', selection, 'hnl_m_20_v2_7p5Em10_dirac_cc',
                   'crimson'    ,10, base_dir, post_fix, False, True, True, 1.,  1.246e-06, toplot=False),

            Sample('HN3L_M_20_V_0p001_mu_Dirac_massiveAndCKM_LO'                , channel, 'HNL m = 20, V^{2} = 1.0E-06, Dirac'   , selection, 'hnl_m_20_v2_1p0Em06_dirac'   , 
                   'crimson'    ,10, base_dir, post_fix, False, True, True, 1.,  0.00224  , toplot=False),

            Sample('HN3L_M_20_V_0p001_mu_massiveAndCKM_LO'                      , channel, 'HNL m = 20, V^{2} = 1.0E-06, Majorana', selection, 'hnl_m_20_v2_1p0Em06_majorana',
                   'crimson'    ,10, base_dir, post_fix, False, True, True, 1.,  0.003856 , toplot=False),

            Sample('HN3L_M_20_V_0p00316227766017_mu_Dirac_massiveAndCKM_LO'     , channel, 'HNL m = 20, V^{2} = 1.0E-05, Dirac'   , selection, 'hnl_m_20_v2_1p0Em05_dirac'   ,
                   'crimson'    ,10, base_dir, post_fix, False, True, True, 1.,  0.02239  , toplot=False),

            Sample('HN3L_M_20_V_0p00316227766017_mu_massiveAndCKM_LO'           , channel, 'HNL m = 20, V^{2} = 1.0E-05, Majorana', selection, 'hnl_m_20_v2_1p0Em05_majorana',
                   'crimson'    ,10, base_dir, post_fix, False, True, True, 1.,  0.03854  , toplot=False),
        ]
    elif channel [0] == 'e': 
         signal = [
            # Sample('HN3L_M_1_V_0p0949736805647_e_massiveAndCKM_LO'		, channel, 'HNL m = 1.0, V^{2} = 9.0E-03, Majorana' , selection, 'hnl_m_1.0_v2_9p0Em03_majorana',
                   # 'darkorange' 	,10, base_dir, post_fix, False, True, True, 1.,  38.23    , toplot=False),
                   # resub'd; had job error

            # Sample('HN3L_M_1_V_0p13416407865_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 1.0, V^{2} = 1.8E-02, Dirac' , selection, 'hnl_m_1.0_v2_1p8Em02_dirac',
                   # 'darkorange' 	,10, base_dir, post_fix, False, True, True, 1.,  44.19    , toplot=False),
                   # resub'd; had job error

            Sample('HN3L_M_1_V_0p13416407865_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 1.0, V^{2} = 1.8E-02, Dirac_cc' , selection, 'hnl_m_1.0_v2_1p8Em02_dirac_cc',
                   'darkorange' 	,10, base_dir, post_fix, False, True, True, 1.,  33.2    , toplot=False),

            # Sample('HN3L_M_1_V_0p212367605816_e_massiveAndCKM_LO'		, channel, 'HNL m = 1.0, V^{2} = 4.5E-02, Majorana' , selection, 'hnl_m_1.0_v2_4p5Em02_majorana',
                   # 'darkorange' 	,10, base_dir, post_fix, False, True, True, 1.,  191.1    , toplot=False),
                   # resub'd; had job error

            # Sample('HN3L_M_1_V_0p300333148354_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 1.0, V^{2} = 9.0E-02, Dirac' , selection, 'hnl_m_1.0_v2_9p0Em02_dirac',
                   # 'darkorange' 	,10, base_dir, post_fix, False, True, True, 1.,  221.1    , toplot=False),

            Sample('HN3L_M_1_V_0p300333148354_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 1.0, V^{2} = 9.0E-02, Dirac_cc' , selection, 'hnl_m_1.0_v2_9p0Em02_dirac_cc',
                   'darkorange' 	,10, base_dir, post_fix, False, True, True, 1.,  166.7    , toplot=False),

            # Sample('HN3L_M_2_V_0p0110905365064_e_massiveAndCKM_LO'		, channel, 'HNL m = 2.0, V^{2} = 1.2E-04, Majorana' , selection, 'hnl_m_2.0_v2_1p2Em04_majorana',
                   # 'forestgreen' 	,10, base_dir, post_fix, False, True, True, 1.,  0.5273    , toplot=True),
                   # resub'd; had job error

            Sample('HN3L_M_2_V_0p0137840487521_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 2.0, V^{2} = 1.9E-04, Dirac_cc' , selection, 'hnl_m_2.0_v2_1p9Em04_dirac_cc',
                   'forestgreen' 	,10, base_dir, post_fix, False, True, True, 1.,  0.3538    , toplot=True),

            # Sample('HN3L_M_2_V_0p0157162336455_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 2.0, V^{2} = 2.5E-04, Dirac' , selection, 'hnl_m_2.0_v2_2p5Em04_dirac',
                   # 'forestgreen' 	,10, base_dir, post_fix, False, True, True, 1.,  0.6093    , toplot=False),
                   # resub'd; had job error

            # Sample('HN3L_M_2_V_0p0157162336455_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 2.0, V^{2} = 2.5E-04, Dirac_cc' , selection, 'hnl_m_2.0_v2_2p5Em04_dirac_cc',
                   # 'forestgreen' 	,10, base_dir, post_fix, False, True, True, 1.,  0.4596    , toplot=False),
                   # resub'd; had job error

            # Sample('HN3L_M_2_V_0p0248394846967_e_massiveAndCKM_LO'		, channel, 'HNL m = 2.0, V^{2} = 6.2E-04, Majorana' , selection, 'hnl_m_2.0_v2_6p2Em04_majorana',
                   # 'forestgreen' 	,10, base_dir, post_fix, False, True, True, 1.,  2.648    , toplot=False),
                   # resub'd; had job error

            Sample('HN3L_M_2_V_0p0307896086367_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 2.0, V^{2} = 9.5E-04, Dirac_cc' , selection, 'hnl_m_2.0_v2_9p5Em04_dirac_cc',
                   'forestgreen' 	,10, base_dir, post_fix, False, True, True, 1.,  1.758    , toplot=False),

            # Sample('HN3L_M_2_V_0p0350713558335_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 2.0, V^{2} = 1.2E-03, Dirac' , selection, 'hnl_m_2.0_v2_1p2Em03_dirac',
                   # 'forestgreen' 	,10, base_dir, post_fix, False, True, True, 1.,  3.039    , toplot=False),
                   # resub'd; had job error

            Sample('HN3L_M_3_V_0p00443846820423_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 3.0, V^{2} = 2.0E-05, Dirac_cc' , selection, 'hnl_m_3.0_v2_2p0Em05_dirac_cc',
                   'firebrick' 	,10, base_dir, post_fix, False, True, True, 1.,  0.03459    , toplot=True),

            # Sample('HN3L_M_3_V_0p00707813534767_e_massiveAndCKM_LO'		, channel, 'HNL m = 3.0, V^{2} = 5.0E-05, Majorana' , selection, 'hnl_m_3.0_v2_5p0Em05_majorana',
                   # 'firebrick' 	,10, base_dir, post_fix, False, True, True, 1.,  0.2022    , toplot=False),
                   # resub'd; had job error

            # Sample('HN3L_M_3_V_0p01_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 3.0, V^{2} = 1.0E-04, Dirac' , selection, 'hnl_m_3.0_v2_1p0Em04_dirac',
                   # 'firebrick' 	,10, base_dir, post_fix, False, True, True, 1.,  0.2325    , toplot=False),

            Sample('HN3L_M_3_V_0p0140356688476_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 3.0, V^{2} = 2.0E-04, Dirac_cc' , selection, 'hnl_m_3.0_v2_2p0Em04_dirac_cc',
                   'firebrick' 	,10, base_dir, post_fix, False, True, True, 1.,  0.3464    , toplot=False),

            Sample('HN3L_M_4_V_0p00183575597507_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 4.0, V^{2} = 3.4E-06, Dirac_cc' , selection, 'hnl_m_4.0_v2_3p4Em06_dirac_cc',
                   'indigo' 	,10, base_dir, post_fix, False, True, True, 1.,  0.005871    , toplot=False),

            # Sample('HN3L_M_4_V_0p00290516780927_e_massiveAndCKM_LO'		, channel, 'HNL m = 4.0, V^{2} = 8.4E-06, Majorana' , selection, 'hnl_m_4.0_v2_8p4Em06_majorana',
                   # 'indigo' 	,10, base_dir, post_fix, False, True, True, 1.,  0.03365    , toplot=False),
                   # resub'd; had job error

            Sample('HN3L_M_4_V_0p00354964786986_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 4.0, V^{2} = 1.3E-05, Dirac_cc' , selection, 'hnl_m_4.0_v2_1p3Em05_dirac_cc',
                   'indigo' 	,10, base_dir, post_fix, False, True, True, 1.,  0.02195    , toplot=False),

            # Sample('HN3L_M_4_V_0p00411096095822_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 4.0, V^{2} = 1.7E-05, Dirac' , selection, 'hnl_m_4.0_v2_1p7Em05_dirac',
                   # 'indigo' 	,10, base_dir, post_fix, False, True, True, 1.,  0.03909    , toplot=False),
                   # resub'd; had job error

            Sample('HN3L_M_4_V_0p0101980390272_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 4.0, V^{2} = 1.0E-04, Dirac_cc' , selection, 'hnl_m_4.0_v2_1p0Em04_dirac_cc',
                   'indigo' 	,10, base_dir, post_fix, False, True, True, 1.,  0.1809    , toplot=False),

            Sample('HN3L_M_5_V_0p000316227766017_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 5.0, V^{2} = 1.0E-07, Dirac' , selection, 'hnl_m_5.0_v2_1p0Em07_dirac',
                   'chocolate' 	,10, base_dir, post_fix, False, True, True, 1.,  0.0002324    , toplot=False),

            Sample('HN3L_M_5_V_0p000316227766017_e_massiveAndCKM_LO'		, channel, 'HNL m = 5.0, V^{2} = 1.0E-07, Majorana' , selection, 'hnl_m_5.0_v2_1p0Em07_majorana',
                   'chocolate' 	,10, base_dir, post_fix, False, True, True, 1.,  0.0003987    , toplot=False),

            Sample('HN3L_M_5_V_0p000547722557505_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 5.0, V^{2} = 3.0E-07, Dirac' , selection, 'hnl_m_5.0_v2_3p0Em07_dirac',
                   'chocolate' 	,10, base_dir, post_fix, False, True, True, 1.,  0.0006978    , toplot=False),

            Sample('HN3L_M_5_V_0p000547722557505_e_massiveAndCKM_LO'		, channel, 'HNL m = 5.0, V^{2} = 3.0E-07, Majorana' , selection, 'hnl_m_5.0_v2_3p0Em07_majorana',
                   'chocolate' 	,10, base_dir, post_fix, False, True, True, 1.,  0.001195    , toplot=False),

            Sample('HN3L_M_5_V_0p000920326029187_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 5.0, V^{2} = 8.5E-07, Dirac_cc' , selection, 'hnl_m_5.0_v2_8p5Em07_dirac_cc',
                   'chocolate' 	,10, base_dir, post_fix, False, True, True, 1.,  0.001481    , toplot=False),

            Sample('HN3L_M_5_V_0p001_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 5.0, V^{2} = 1.0E-06, Dirac' , selection, 'hnl_m_5.0_v2_1p0Em06_dirac',
                   'chocolate' 	,10, base_dir, post_fix, False, True, True, 1.,  0.002325    , toplot=False),

            Sample('HN3L_M_5_V_0p001_e_massiveAndCKM_LO'		, channel, 'HNL m = 5.0, V^{2} = 1.0E-06, Majorana' , selection, 'hnl_m_5.0_v2_1p0Em06_majorana',
                   'chocolate' 	,10, base_dir, post_fix, False, True, True, 1.,  0.00399    , toplot=False),

            # Sample('HN3L_M_5_V_0p00145602197786_e_massiveAndCKM_LO'		, channel, 'HNL m = 5.0, V^{2} = 2.1E-06, Majorana' , selection, 'hnl_m_5.0_v2_2p1Em06_majorana',
                   # 'chocolate' 	,10, base_dir, post_fix, False, True, True, 1.,  0.008479    , toplot=True),
                   # resub'd; had job error

            Sample('HN3L_M_5_V_0p00178044938148_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 5.0, V^{2} = 3.2E-06, Dirac_cc' , selection, 'hnl_m_5.0_v2_3p2Em06_dirac_cc',
                   'chocolate' 	,10, base_dir, post_fix, False, True, True, 1.,  0.005548    , toplot=True),

            # Sample('HN3L_M_5_V_0p00205669638012_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 5.0, V^{2} = 4.2E-06, Dirac' , selection, 'hnl_m_5.0_v2_4p2Em06_dirac',
                   # 'chocolate' 	,10, base_dir, post_fix, False, True, True, 1.,  0.009818    , toplot=False),
                   # resub'd; had job error

            Sample('HN3L_M_5_V_0p0065574385243_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 5.0, V^{2} = 4.3E-05, Dirac_cc' , selection, 'hnl_m_5.0_v2_4p3Em05_dirac_cc',
                   'chocolate' 	,10, base_dir, post_fix, False, True, True, 1.,  0.07512    , toplot=False),

            Sample('HN3L_M_6_V_0p000522494019105_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 6.0, V^{2} = 2.7E-07, Dirac_cc' , selection, 'hnl_m_6.0_v2_2p7Em07_dirac_cc',
                   'olive' 	,10, base_dir, post_fix, False, True, True, 1.,  0.0004838    , toplot=False),

            Sample('HN3L_M_6_V_0p00101488915651_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 6.0, V^{2} = 1.0E-06, Dirac_cc' , selection, 'hnl_m_6.0_v2_1p0Em06_dirac_cc',
                   'olive' 	,10, base_dir, post_fix, False, True, True, 1.,  0.001823    , toplot=False),

            # Sample('HN3L_M_6_V_0p00202484567313_e_massiveAndCKM_LO'		, channel, 'HNL m = 6.0, V^{2} = 4.1E-06, Majorana' , selection, 'hnl_m_6.0_v2_4p1Em06_majorana',
                   # 'olive' 	,10, base_dir, post_fix, False, True, True, 1.,  0.01655    , toplot=False),
                   # resub'd; had job error

            # Sample('HN3L_M_6_V_0p00286356421266_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 6.0, V^{2} = 8.2E-06, Dirac' , selection, 'hnl_m_6.0_v2_8p2Em06_dirac',
                   # 'olive' 	,10, base_dir, post_fix, False, True, True, 1.,  0.01928    , toplot=False),
                   # resub'd; had job error

            Sample('HN3L_M_6_V_0p00299666481275_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 6.0, V^{2} = 9.0E-06, Dirac_cc' , selection, 'hnl_m_6.0_v2_9p0Em06_dirac_cc',
                   'olive' 	,10, base_dir, post_fix, False, True, True, 1.,  0.01595    , toplot=False),

            Sample('HN3L_M_8_V_0p000316227766017_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 8.0, V^{2} = 1.0E-07, Dirac' , selection, 'hnl_m_8.0_v2_1p0Em07_dirac',
                   'darkgray' 	,10, base_dir, post_fix, False, True, True, 1.,  0.0002387    , toplot=False),

            Sample('HN3L_M_8_V_0p000316227766017_e_massiveAndCKM_LO'		, channel, 'HNL m = 8.0, V^{2} = 1.0E-07, Majorana' , selection, 'hnl_m_8.0_v2_1p0Em07_majorana',
                   'darkgray' 	,10, base_dir, post_fix, False, True, True, 1.,  0.0004093    , toplot=False),

            Sample('HN3L_M_8_V_0p000415932686862_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 8.0, V^{2} = 1.7E-07, Dirac_cc' , selection, 'hnl_m_8.0_v2_1p7Em07_dirac_cc',
                   'darkgray' 	,10, base_dir, post_fix, False, True, True, 1.,  0.0003091    , toplot=False),

            Sample('HN3L_M_8_V_0p000547722557505_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 8.0, V^{2} = 3.0E-07, Dirac' , selection, 'hnl_m_8.0_v2_3p0Em07_dirac',
                   'darkgray' 	,10, base_dir, post_fix, False, True, True, 1.,  0.000716    , toplot=False),

            Sample('HN3L_M_8_V_0p000547722557505_e_massiveAndCKM_LO'		, channel, 'HNL m = 8.0, V^{2} = 3.0E-07, Majorana' , selection, 'hnl_m_8.0_v2_3p0Em07_majorana',
                   'darkgray' 	,10, base_dir, post_fix, False, True, True, 1.,  0.001228    , toplot=False),

            Sample('HN3L_M_8_V_0p001_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 8.0, V^{2} = 1.0E-06, Dirac' , selection, 'hnl_m_8.0_v2_1p0Em06_dirac',
                   'darkgray' 	,10, base_dir, post_fix, False, True, True, 1.,  0.002387    , toplot=False),

            Sample('HN3L_M_8_V_0p001_e_massiveAndCKM_LO'		, channel, 'HNL m = 8.0, V^{2} = 1.0E-06, Majorana' , selection, 'hnl_m_8.0_v2_1p0Em06_majorana',
                   'darkgray' 	,10, base_dir, post_fix, False, True, True, 1.,  0.004096    , toplot=False),

            # Sample('HN3L_M_8_V_0p00151327459504_e_massiveAndCKM_LO'		, channel, 'HNL m = 8.0, V^{2} = 2.3E-06, Majorana' , selection, 'hnl_m_8.0_v2_2p3Em06_majorana',
                   # 'darkgray' 	,10, base_dir, post_fix, False, True, True, 1.,  0.009383    , toplot=False),
                   # resub'd; had job error

            # Sample('HN3L_M_8_V_0p00214242852856_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 8.0, V^{2} = 4.6E-06, Dirac' , selection, 'hnl_m_8.0_v2_4p6Em06_dirac',
                   # 'darkgray' 	,10, base_dir, post_fix, False, True, True, 1.,  0.01096    , toplot=False),
                   # resub'd; had job error

            Sample('HN3L_M_8_V_0p00363318042492_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 8.0, V^{2} = 1.3E-05, Dirac_cc' , selection, 'hnl_m_8.0_v2_1p3Em05_dirac_cc',
                   'darkgray' 	,10, base_dir, post_fix, False, True, True, 1.,  0.02359    , toplot=False),

            Sample('HN3L_M_10_V_0p000208566536146_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 10.0, V^{2} = 4.3E-08, Dirac_cc' , selection, 'hnl_m_10.0_v2_4p3Em08_dirac_cc',
                   'teal' 	,10, base_dir, post_fix, False, True, True, 1.,  7.843e-05    , toplot=False),

            Sample('HN3L_M_10_V_0p000316227766017_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 10.0, V^{2} = 1.0E-07, Dirac' , selection, 'hnl_m_10.0_v2_1p0Em07_dirac',
                   'teal' 	,10, base_dir, post_fix, False, True, True, 1.,  0.0002404    , toplot=False),

            Sample('HN3L_M_10_V_0p000316227766017_e_massiveAndCKM_LO'		, channel, 'HNL m = 10.0, V^{2} = 1.0E-07, Majorana' , selection, 'hnl_m_10.0_v2_1p0Em07_majorana',
                   'teal' 	,10, base_dir, post_fix, False, True, True, 1.,  0.0004118    , toplot=False),

            Sample('HN3L_M_10_V_0p000547722557505_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 10.0, V^{2} = 3.0E-07, Dirac' , selection, 'hnl_m_10.0_v2_3p0Em07_dirac',
                   'teal' 	,10, base_dir, post_fix, False, True, True, 1.,  0.0007208    , toplot=False),

            Sample('HN3L_M_10_V_0p000547722557505_e_massiveAndCKM_LO'		, channel, 'HNL m = 10.0, V^{2} = 3.0E-07, Majorana' , selection, 'hnl_m_10.0_v2_3p0Em07_majorana',
                   'teal' 	,10, base_dir, post_fix, False, True, True, 1.,  0.001237    , toplot=False),

            # Sample('HN3L_M_10_V_0p000756967634711_e_massiveAndCKM_LO'		, channel, 'HNL m = 10.0, V^{2} = 5.7E-07, Majorana' , selection, 'hnl_m_10.0_v2_5p7Em07_majorana',
                   # 'teal' 	,10, base_dir, post_fix, False, True, True, 1.,  0.002366    , toplot=False),
                   # resub'd; had job error

            Sample('HN3L_M_10_V_0p001_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 10.0, V^{2} = 1.0E-06, Dirac' , selection, 'hnl_m_10.0_v2_1p0Em06_dirac',
                   'teal' 	,10, base_dir, post_fix, False, True, True, 1.,  0.002403    , toplot=True),

            Sample('HN3L_M_10_V_0p001_e_massiveAndCKM_LO'		, channel, 'HNL m = 10.0, V^{2} = 1.0E-06, Majorana' , selection, 'hnl_m_10.0_v2_1p0Em06_majorana',
                   'teal' 	,10, base_dir, post_fix, False, True, True, 1.,  0.004114    , toplot=False),

            # Sample('HN3L_M_10_V_0p00107238052948_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 10.0, V^{2} = 1.2E-06, Dirac' , selection, 'hnl_m_10.0_v2_1p2Em06_dirac',
                   # 'teal' 	,10, base_dir, post_fix, False, True, True, 1.,  0.002764    , toplot=False),
                   # resub'd; had job error

            Sample('HN3L_M_10_V_0p00112249721603_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 10.0, V^{2} = 1.3E-06, Dirac_cc' , selection, 'hnl_m_10.0_v2_1p3Em06_dirac_cc',
                   'teal' 	,10, base_dir, post_fix, False, True, True, 1.,  0.002265    , toplot=False),

            Sample('HN3L_M_15_V_0p00003021588986_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 15.0, V^{2} = 9.1E-10, Dirac_cc' , selection, 'hnl_m_15.0_v2_9p1Em10_dirac_cc',
                   'gold' 	,10, base_dir, post_fix, False, True, True, 1.,  1.616e-06    , toplot=False),

            Sample('HN3L_M_15_V_0p00006760177512_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 15.0, V^{2} = 4.6E-09, Dirac_cc' , selection, 'hnl_m_15.0_v2_4p6Em09_dirac_cc',
                   'gold' 	,10, base_dir, post_fix, False, True, True, 1.,  8.092e-06    , toplot=False),

            Sample('HN3L_M_20_V_0p00001224744871_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 20.0, V^{2} = 1.5E-10, Dirac_cc' , selection, 'hnl_m_20.0_v2_1p5Em10_dirac_cc',
                   'crimson' 	,10, base_dir, post_fix, False, True, True, 1.,  2.511e-07    , toplot=False),

            Sample('HN3L_M_20_V_0p00002734958866_e_Dirac_cc_massiveAndCKM_LO'		, channel, 'HNL m = 20.0, V^{2} = 7.5E-10, Dirac_cc' , selection, 'hnl_m_20.0_v2_7p5Em10_dirac_cc',
                   'crimson' 	,10, base_dir, post_fix, False, True, True, 1.,  1.26e-06    , toplot=False),

            Sample('HN3L_M_20_V_0p001_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 20.0, V^{2} = 1.0E-06, Dirac' , selection, 'hnl_m_20.0_v2_1p0Em06_dirac',
                   'crimson' 	,10, base_dir, post_fix, False, True, True, 1.,  0.002232    , toplot=False),

            Sample('HN3L_M_20_V_0p001_e_massiveAndCKM_LO'		, channel, 'HNL m = 20.0, V^{2} = 1.0E-06, Majorana' , selection, 'hnl_m_20.0_v2_1p0Em06_majorana',
                   'crimson' 	,10, base_dir, post_fix, False, True, True, 1.,  0.003853    , toplot=False),

            Sample('HN3L_M_20_V_0p00316227766017_e_Dirac_massiveAndCKM_LO'		, channel, 'HNL m = 20.0, V^{2} = 1.0E-05, Dirac' , selection, 'hnl_m_20.0_v2_1p0Em05_dirac',
                   'crimson' 	,10, base_dir, post_fix, False, True, True, 1.,  0.02233    , toplot=False),

            Sample('HN3L_M_20_V_0p00316227766017_e_massiveAndCKM_LO'		, channel, 'HNL m = 20.0, V^{2} = 1.0E-05, Majorana' , selection, 'hnl_m_20.0_v2_1p0Em05_majorana',
                   'crimson' 	,10, base_dir, post_fix, False, True, True, 1.,  0.03856    , toplot=False),
         ]
    
    return signal
