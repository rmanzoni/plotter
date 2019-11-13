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
from selections import selections_df
from evaluate_nn import Evaluator
from sample import Sample, get_data_samples, get_mc_samples, get_signal_samples
from variables import variables

from rootpy.plotting import Hist, HistStack, Legend, Canvas, Graph
from rootpy.plotting.style import get_style, set_style
from rootpy.plotting.utils import draw

import logging
logging.disable(logging.DEBUG)

ROOT.gROOT.SetBatch(True)

class Plotter(object):

    def __init__(self          , 
                 channel       , 
                 basedir       ,
                 postfix       ,
                 lumi          ,
                 selection_data,
                 selection_mc  ,
                 model         , 
                 transformation,
                 features       ):

        self.channel        = channel 
        self.basedir        = basedir 
        self.postfix        = postfix 
        self.lumi           = lumi
        self.selection_data = selection_data
        self.selection_mc   = selection_mc
        self.model          = model          
        self.transformation = transformation 
        self.features       = features       

    def plot(self):

        evaluator = Evaluator(self.model, self.transformation, self.features)

# NN evaluator

        print('============> starting reading the trees')
        now = time.time()
        signal = get_signal_samples(self.basedir, self.postfix, self.selection_data)
        data   = get_data_samples  (self.basedir, self.postfix, self.selection_data)
        mc     = get_mc_samples    (self.basedir, self.postfix, self.selection_mc)
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

        for ivar in variables:
            
            variable, bins, label, xlabel, ylabel, extra_sel = ivar.var, ivar.bins, ivar.label, ivar.xlabel, ivar.ylabel, ivar.extra_selection
            
            print('plotting', label)

            # clean the figure
            plt.clf()
            
            ######################################################################################
            # plot MC stacks, in tight and loose
            ######################################################################################
            
            stack_prompt    = []
            stack_nonprompt = []
            
            for imc in mc:
                
                if extra_sel:
                    mc_df_tight = imc.df_tight.query(extra_sel) 
                    mc_df_loose = imc.df_loose.query(extra_sel) 
                else:
                    mc_df_tight = imc.df_tight
                    mc_df_loose = imc.df_loose
                
                histo_tight = Hist(bins, title=imc.label, markersize=0, legendstyle='F', name=imc.datacard_name)
                histo_tight.fill_array(mc_df_tight[variable], weights=self.lumi * mc_df_tight.weight * imc.lumi_scaling * mc_df_tight.lhe_weight)

                histo_tight.fillstyle = 'solid'
                histo_tight.fillcolor = 'steelblue'
                histo_tight.linewidth = 0

                stack_prompt.append(histo_tight)

                histo_loose = Hist(bins, title=imc.label, markersize=0, legendstyle='F')
                histo_tight.fill_array(mc_df_tight[variable], weights=self.lumi * mc_df_tight.weight * imc.lumi_scaling * mc_df_tight.lhe_weight)


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

                if extra_sel:
                    isig_df_tight = isig.df_tight.query(extra_sel) 
                else:
                    isig_df_tight = isig.df_tight

                histo_tight = Hist(bins, title=isig.label, markersize=0, legendstyle='L', name=isig.datacard_name)
                histo_tight.fill_array(isig_df_tight[variable], weights=self.lumi * isig_df_tight.weight * isig.lumi_scaling * isig_df_tight.lhe_weight)
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

                if extra_sel:
                    idata_df_tight = idata.df_tight.query(extra_sel) 
                    idata_df_loose = idata.df_loose.query(extra_sel) 
                else:
                    idata_df_tight = idata.df_tight
                    idata_df_loose = idata.df_loose

                histo_tight = Hist(bins, title=idata.label, markersize=1, legendstyle='LEP')
                histo_tight.fill_array(idata_df_tight[variable])
                
                data_prompt.append(histo_tight)

                histo_loose = Hist(bins, title=idata.label, markersize=0, legendstyle='F')
                histo_loose.fill_array(idata_df_loose[variable], weights=idata_df_loose.fr_corr)
                
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
                canvas.SaveAs('plots/%s%s.pdf' %(variable, islogy*'_log'))

            # save a ROOT file with histograms, aka datacard
            outfile = ROOT.TFile.Open('plots/datacard_%s.root' %variable, 'recreate')
            outfile.cd()
            
            # data in tight
            all_obs_prompt.name = 'data_obs'
            all_obs_prompt.Write()
            
            # non prompt backgrounds in tight
            all_exp_nonprompt.name = 'nonprompt'
            all_exp_nonprompt.drawstyle = 'HIST E'
            all_exp_nonprompt.color = 'black'
            all_exp_nonprompt.linewidth = 2
            all_exp_nonprompt.Write()

            # prompt backgrounds in tight
            all_exp_prompt.name = 'prompt'
            all_exp_prompt.drawstyle = 'HIST E'
            all_exp_prompt.color = 'black'
            all_exp_prompt.linewidth = 2
            all_exp_prompt.Write()
            
            # signals
            for isig in all_signals:
                isig.drawstyle = 'HIST E'
                isig.color = 'black'
                isig.Write()
                
            outfile.Close()
