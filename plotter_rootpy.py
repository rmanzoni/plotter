# https://indico.cern.ch/event/759388/contributions/3306849/attachments/1816254/2968550/root_conda_forge.pdf
# https://conda-forge.org/feedstocks/

import re
import time
import ROOT
import root_pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import OrderedDict
from selections import selections, selections_df
from evaluate_nn import Evaluator
from sample import Sample, get_data_samples, get_mc_samples, get_signal_samples
from variables import variables

from rootpy.plotting import Hist, HistStack, Legend, Canvas, Graph
from rootpy.plotting.style import get_style, set_style
from rootpy.plotting.utils import draw

import logging
logging.disable(logging.DEBUG)

ROOT.gROOT.SetBatch(True)

# set the style
# style = get_style('ATLAS')
# style.SetEndErrorSize(3)
# set_style(style)

basedir        = '/Users/manzoni/Documents/efficiencyNN/HNL/mmm/ntuples/'
postfix        = 'HNLTreeProducer/tree.root'
lumi           = 59700. # fb-1
selection_data = selections['baseline']
# selection_mc   = selections['baseline'] 
selection_mc   = '&'.join([selections['baseline'], selections['ispromptlepton']])

# NN evaluator
model          = '/Users/manzoni/Documents/efficiencyNN/HNL/mmm/net_model_weighted.h5'
transformation = '/Users/manzoni/Documents/efficiencyNN/HNL/mmm/input_tranformation_weighted.pck'
features       = '/Users/manzoni/Documents/efficiencyNN/HNL/mmm/input_features.pck'
evaluator      = Evaluator(model, transformation, features)

print('============> starting reading the trees')
now = time.time()
signal = get_signal_samples(basedir, postfix, selection_data)
data   = get_data_samples  (basedir, postfix, selection_data)
mc     = get_mc_samples    (basedir, postfix, selection_mc)
print('============> it took %.2f seconds' %(time.time() - now))

# evaluate FR
for isample in (mc+data):
    isample.df['fr'] = evaluator.evaluate(isample.df)
    # already corrected, ready to be applied in loose-not-tight
    isample.df['fr_corr'] = isample.df['fr'] / (1. - isample.df['fr']) 

# split the dataframe in tight and loose-not-tight (called simply loose for short)
for isample in (mc+data+signal):
    isample.df_tight = isample.df.query(selections_df['tight'])
    isample.df_loose = isample.df.query('not(%s)'%selections_df['tight'])

# sort depending on their position in the stack
mc.sort(key = lambda x : x.position_in_stack)

# now we plot
canvas = Canvas(width=700, height=700)

for variable, bins, xlabel, ylabel in variables:
    
    print('plotting', variable)

    # clean the figure
    plt.clf()
    
    ######################################################################################
    # plot MC stacks, in tight and loose
    ######################################################################################
    
    stack_prompt    = []
    stack_nonprompt = []
    
    for imc in mc:

        histo_tight = Hist(bins, title=imc.label, markersize=0, legendstyle='F', name=imc.datacard_name)
        histo_tight.fill_array(imc.df_tight[variable], weights=lumi * imc.df_tight.weight * imc.lumi_scaling * imc.df_tight.lhe_weight)

        histo_tight.fillstyle = 'solid'
        histo_tight.fillcolor = 'steelblue'
        histo_tight.linewidth = 0

        stack_prompt.append(histo_tight)

        histo_loose = Hist(bins, title=imc.label, markersize=0, legendstyle='F')
        histo_loose.fill_array(imc.df_tight[variable], weights=-1.* lumi * imc.df_loose.weight * imc.lumi_scaling * imc.df_loose.lhe_weight * imc.df_loose.fr_corr)

        histo_loose.fillstyle = 'solid'
        histo_loose.fillcolor = 'skyblue'
        histo_loose.linewidth = 0

        stack_nonprompt.append(histo_loose)

    ######################################################################################
    # plot the signals
    ######################################################################################
    
    all_signals     = []
    signals_to_plot = []
    
    for isig in signal:
        histo_tight = Hist(bins, title=isig.label, markersize=0, legendstyle='L', name=isig.datacard_name)
        histo_tight.fill_array(isig.df_tight[variable], weights=lumi * isig.df_tight.weight * isig.lumi_scaling * isig.df_tight.lhe_weight)
        histo_tight.color     = isig.colour
        histo_tight.fillstyle = 'hollow'
        histo_tight.linewidth = 2
        histo_tight.linestyle = 'dashed'
        histo_tight.drawstyle = 'HIST'

        all_signals.append(histo_tight)
        if isig.toplot: signals_to_plot.append(histo_tight)
    
    ######################################################################################
    # plot the data
    ######################################################################################

    data_prompt    = []
    data_nonprompt = []
    
    for idata in data:
        histo_tight = Hist(bins, title=idata.label, markersize=1, legendstyle='LEP')
        histo_tight.fill_array(idata.df_tight[variable])
        
        data_prompt.append(histo_tight)

        histo_loose = Hist(bins, title=idata.label, markersize=0, legendstyle='F')
        histo_loose.fill_array(idata.df_loose[variable], weights=idata.df_loose.fr_corr)
        
        data_nonprompt.append(histo_loose)

    # put the prompt backgrounds together
    all_exp_prompt = sum(stack_prompt)
    all_exp_prompt.title = 'prompt'

    # put the nonprompt backgrounds together
    all_exp_nonprompt = sum(stack_nonprompt+data_nonprompt)
    all_exp_nonprompt.title = 'nonprompt'

    # create the stacks
    stack = HistStack([all_exp_prompt, all_exp_nonprompt], drawstyle='HIST', title='')

    # stat uncertainty
    hist_error = sum([all_exp_prompt, all_exp_nonprompt])    
    hist_error.drawstyle = 'E2'
    hist_error.fillstyle = '/'
    hist_error.color     = 'gray'
    hist_error.title     = 'stat. unc.'
    hist_error.legendstyle = 'F'

    # put the data together
    all_obs_prompt = sum(data_prompt)
    all_obs_prompt.title = 'observed'

    # prepare the legend
    legend = Legend([all_obs_prompt, stack, hist_error] + signals_to_plot, leftmargin=0.45, margin=0.3, textsize=0.023, textfont=42, entrysep=0.012)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)

    # plot with ROOT, linear and log scale
    for islogy in [False, True]:
        draw([stack, hist_error, all_obs_prompt] + signals_to_plot, xtitle=xlabel, ytitle=ylabel, pad=canvas, logy=islogy)
        legend.Draw('same')
        canvas.Modified()
        canvas.Update()
        canvas.SaveAs('%s%s.pdf' %(variable, islogy*'_log'))

    # save a ROOT file with histograms, aka datacard
    outfile = ROOT.TFile.Open('datacard_%s.root' %variable, 'recreate')
    outfile.cd()
    
    # data in tight
    all_obs_prompt.name = 'data_obs'
    all_obs_prompt.Write()
    
    # non prompt backgrounds in tight
    all_exp_nonprompt.name = 'nonprompt'
    all_exp_nonprompt.drawstyle = 'HIST E'
    all_exp_nonprompt.linewidth = 2
    all_exp_nonprompt.Write()

    # prompt backgrounds in tight
    all_exp_prompt.name = 'prompt'
    all_exp_prompt.drawstyle = 'HIST E'
    all_exp_prompt.linewidth = 2
    all_exp_prompt.Write()
    
    # signals
    for isig in all_signals:
        isig.drawstyle = 'HIST E'
        isig.Write()
        
    outfile.Close()

