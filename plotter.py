import re
import ROOT
import rootpy
import root_pandas
import numpy as np
import pandas as pd
from rootpy.plotting import Hist
from root_numpy import root2array
from collections import OrderedDict
from selections import baseline, tight, ispromptlepton, zmm
import matplotlib.pyplot as plt

basedir = '/Users/manzoni/Documents/efficiencyNN/HNL/mmm/ntuples/'
postfix = 'HNLTreeProducer/tree.root'
lumi    = 59700. # fb-1

class Sample(object):
    def __init__(self, 
                 name, 
                 label,
                 datacard_name,
                 colour,
                 position_in_stack, 
                 basedir, 
                 postfix, 
                 isdata, 
                 ismc, 
                 issignal, 
                 weight,
                 xs):
        self.name              = name             
        self.label             = label            
        self.datacard_name     = datacard_name            
        self.colour            = colour           
        self.position_in_stack = position_in_stack
        self.basedir           = basedir          
        self.postfix           = postfix          
        self.isdata            = isdata           
        self.ismc              = ismc             
        self.issignal          = issignal         
        self.weight            = weight           
        self.xs                = xs        
        self.file              =   '/'.join([basedir, self.name, postfix])          
        self.nevents           = 1.
        if not self.isdata:
            nevents_file = '/'.join([basedir, self.name, 'SkimAnalyzerCount/SkimReport.txt'])
            with open(nevents_file) as ff:
                lines = ff.readlines()
                for line in lines:
                    if 'Sum Norm Weights' in line:
                        self.nevents = float(re.findall(r'\d+', lines[2])[0])
                        break
                    
data = [
    Sample('Single_mu_2018A', '2018A', 'data_obs', 'black', 9999, basedir, postfix, True, False, False, 1., 1.),
    Sample('Single_mu_2018B', '2018B', 'data_obs', 'black', 9999, basedir, postfix, True, False, False, 1., 1.),
    Sample('Single_mu_2018C', '2018C', 'data_obs', 'black', 9999, basedir, postfix, True, False, False, 1., 1.),
    Sample('Single_mu_2018D', '2018D', 'data_obs', 'black', 9999, basedir, postfix, True, False, False, 1., 1.),
]

mc = [
    Sample('DYJetsToLL_M50_ext', r'DY$\to\ell\ell$'           ,       'DY', 'gold'         ,10, basedir, postfix, False, True, False, 1.,  6077.22),
    Sample('TTJets_ext'        , r'$t\bar{t}$'                ,       'TT', 'slateblue'    , 0, basedir, postfix, False, True, False, 1.,   831.76),
    Sample('WW'                , 'WW'                         ,       'WW', 'blue'         , 5, basedir, postfix, False, True, False, 1.,    75.88),
    Sample('WZ'                , 'WZ'                         ,       'WZ', 'blue'         , 5, basedir, postfix, False, True, False, 1.,    27.6 ),
    Sample('ZZ'                , 'ZZ'                         ,       'ZZ', 'blue'         , 5, basedir, postfix, False, True, False, 1.,    12.14),
#     Sample('WJetsToLNu'        , r'$W\to\ell\nu$'             ,        'W', 'firebrick'    , 2, basedir, postfix, False, True, False, 1., 59850.  ),
#     Sample('ST_sch_lep'        , r'single $t$ s-channel'      ,    'STsch', 'darkslateblue', 3, basedir, postfix, False, True, False, 1.,     3.68),
#     Sample('ST_tW_inc'         , 'tW'                         ,       'TW', 'darkslateblue', 3, basedir, postfix, False, True, False, 1.,    35.6 ),
#     Sample('ST_tch_inc'        , r'single $t$ t-channel'      ,    'STtch', 'darkslateblue', 3, basedir, postfix, False, True, False, 1.,    44.07),
#     Sample('STbar_tW_inc'      , r'$\bar{t}$W'                ,    'TbarW', 'darkslateblue', 3, basedir, postfix, False, True, False, 1.,    35.6 ),
#     Sample('STbar_tch_inc'     , r'single $\bar{t}$ t-channel', 'STbartch', 'darkslateblue', 3, basedir, postfix, False, True, False, 1.,    26.23),

#     Sample('DYJetsToLL_M5to50' , r'DY$\to\ell\ell$ low mass', 'gold'     , 1, basedir, postfix, False, True, False, 1., 81880.0 ),
#     Sample('DYJetsToLL_M50' , r'DY$\to\ell\ell$'         , 'gold'     , 2, basedir, postfix, False, True, False, 1.,  6077.22),
#     Sample('TTJets'         , r'$t\bar{t}$'              , 'slateblue', 0, basedir, postfix, False, True, False, 1.,   831.76),
#     Sample('WGamma'         , r'$t\bar{t}$'              , 'slateblue', 0, basedir, postfix, False, True, False, 1.,   831.76),
#     Sample('ZGamma'            , r'$t\bar{t}$'              , 'slateblue', 0, basedir, postfix, False, True, False, 1.,   831.76),
]            

signal = [
]

datasets = OrderedDict()

for i, isample in enumerate(data+mc+signal):
    print 'loading', isample.name
    file = '/'.join([basedir, isample.name, postfix])     
    df = pd.DataFrame( root2array(file, 'tree', selection=zmm) )
    df['sample'] = isample.name
    df['lumi_scaling'] = 1. if isample.isdata else (lumi * isample.xs / isample.nevents)
    # append the dataframe to the sample
    isample.df = df
    datasets[isample.name] = df

# merge the data together
df_data = pd.concat([idata.df for idata in data])

# variables
variables = [
    ('hnl_m_01', np.linspace(0.,120., 30 + 1), r'$m_{01}$ (GeV)', 'events'),
    ('hnl_m_12', np.linspace(0., 12., 12 + 1), r'$m_{12}$ (GeV)', 'events'),
    ('hnl_m_02', np.linspace(0.,120., 30 + 1), r'$m_{02}$ (GeV)', 'events'),

    ('hnl_2d_disp'    , np.linspace( 0, 30, 25 + 1), r'$L_{xy}$ (cm)'       , 'events'),
    ('hnl_2d_disp_sig', np.linspace( 0,200, 25 + 1), r'$L_{xy}/\sigma_{xy}$', 'events'),
    ('nbj'            , np.linspace( 0,  5,  5 + 1), '#b-jet'               , 'events'),
    ('hnl_w_vis_m'    , np.linspace( 0,150, 40 + 1), r'$m_{3l}$'            , 'events'),
    ('hnl_q_01'       , np.linspace(-3,  3,  3 + 1), r'$q_{01}$'            , 'events'),
    ('sv_cos'         , np.linspace( 0,  1, 30 + 1), r'$\cos\alpha$'        , 'events'),
    ('sv_prob'        , np.linspace( 0,  1, 30 + 1), 'SV probability'       , 'events'),
]

# now we plot
plt.figure()
for variable, bins, xlabel, ylabel in variables:
    
    print 'plotting', variable
      
    plt.clf()
    
    # sort depending on their position in the stack
    mc.sort(key = lambda x : x.position_in_stack)

    # plot MC stack
    stack   = [getattr(imc.df, variable)                               for imc in mc] 
    labels  = [imc.label                                               for imc in mc]
    weights = [imc.df.weight * imc.df.lumi_scaling * imc.df.lhe_weight for imc in mc] 
    colours = [imc.colour                                              for imc in mc] 
    
    plt.hist(stack, bins, stacked=True, label=labels, weights=weights, color=colours)

    # plot data
    bin_centres = np.array([0.5*(bins[i] + bins[i+1]) for i in np.arange(len(bins)-1)])
    counts = np.array([df_data.query( '%s > %f and %s <= %f' %(variable, bins[i], variable, bins[i+1]) ).shape[0] for i in np.arange(len(bins)-1)])
    plt.errorbar(bin_centres, counts, yerr=np.sqrt(counts), fmt='o', color='black', label='observed')
            
    # legend and save it!
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('%s.pdf' %variable)

    # save a ROOT file with histograms, aka datacard
    outfile = ROOT.TFile.Open('datacard_%s.root' %variable, 'recreate')
    outfile.cd()
    h_data = Hist(bins, name='data_obs')
    h_data.fill_array(df_data[variable])
    h_data.Write()
    for imc in mc:
        h_mc = Hist(bins, name=imc.datacard_name)
        h_mc.fill_array(imc.df[variable], imc.df.weight * imc.df.lumi_scaling * imc.df.lhe_weight)
        h_mc.Write()
    outfile.Close()

