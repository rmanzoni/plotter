import os
import re
import gc # free mem up
import ROOT
import root_pandas
import numpy as np
import pandas as pd
from copy import copy, deepcopy
from os import makedirs
from time import time
from collections import OrderedDict
from itertools import groupby
from functools import partial, reduce

from plotter.objects.evaluate_nn import Evaluator
from plotter.objects.variables import variables
from plotter.objects.utils import get_time_str
from plotter.objects.cmsstyle import CMS_lumi
from plotter.objects.sample import groups, togroup

from rootpy.plotting import Hist, HistStack, Legend, Canvas, Graph, Pad
from rootpy.plotting.style import get_style, set_style
from rootpy.plotting.utils import draw
from pdb import set_trace

import logging
logging.disable(logging.DEBUG)

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(False)

class Plotter(object):

    def __init__(self                 , 
                 channel              ,
                 year                 ,
                 plot_dir             , 
                 base_dir             ,
                 post_fix             ,
                 selection_data       ,
                 selection_mc         ,
                 selection_tight      ,
                 pandas_selection     ,
                 lumi                 ,
                 model                , 
                 transformation       ,
                 features             ,
                 process_signals      , 
                 plot_signals         ,
                 blinded              ,
                 datacards=[]         ,
                 mini_signals=False   ,
                 do_ratio=True        ,
                 mc_subtraction=True  ,
                 dir_suffix=''        ,
                 relaxed_mc_scaling=1.,
                 data_driven=True     ,    
                 save_synch_tuple=[]):

        self.channel              = channel.split('_')[0]
        self.year                 = year
        self.full_channel         = channel
        self.plt_dir              = '/'.join([plot_dir, channel, '_'.join([dir_suffix, get_time_str()]) ])
        self.base_dir             = base_dir 
        self.post_fix             = post_fix 
        self.selection_data       = ' & '.join(selection_data)
        self.selection_mc         = ' & '.join(selection_mc)
        self.selection_tight      = selection_tight
        self.pandas_selection     = pandas_selection
        self.lumi                 = lumi
        self.model                = model          
        self.transformation       = transformation 
        self.features             = features 
        self.process_signals      = process_signals    
        self.plot_signals         = plot_signals if self.process_signals else []
        self.blinded              = blinded      
        self.selection_lnt        = 'not (%s)' %self.selection_tight
        self.do_ratio             = do_ratio
        self.mini_signals         = mini_signals
        self.datacards            = datacards
        self.mc_subtraction       = mc_subtraction
        self.relaxed_mc_scaling   = relaxed_mc_scaling
        self.data_driven          = data_driven
        self.save_synch_tuple     = save_synch_tuple
        
        if self.year==2018:
            from plotter.samples.samples_2018 import get_data_samples, get_mc_samples, get_signal_samples
        if self.year==2017:
            from plotter.samples.samples_2017 import get_data_samples, get_mc_samples, get_signal_samples
        if self.year==2016:
            from plotter.samples.samples_2016 import get_data_samples, get_mc_samples, get_signal_samples
        
        self.get_data_samples   = get_data_samples
        self.get_mc_samples     = get_mc_samples
        self.get_signal_samples = get_signal_samples
            
    def total_weight_calculator(self, df, weight_list, scalar_weights=[]):
        total_weight = df[weight_list[0]].to_numpy().astype(np.float)
        for iw in weight_list[1:]:
            total_weight *= df[iw].to_numpy().astype(np.float)
        for iw in scalar_weights:
            total_weight *= iw
        return total_weight

    def create_canvas(self, ratio=True):
        if ratio:
            self.canvas = Canvas(width=700, height=700) ; self.canvas.Draw()
            self.canvas.cd() ; self.main_pad   = Pad(0.  , 0.25, 1. , 1.  ) ; self.main_pad .Draw()
            self.canvas.cd() ; self.ratio_pad  = Pad(0.  , 0.  , 1. , 0.25) ; self.ratio_pad.Draw()

            self.main_pad.SetTicks(True)
            self.main_pad.SetBottomMargin(0.)
            self.main_pad.SetLeftMargin(0.15)
            self.main_pad.SetRightMargin(0.15)
            self.ratio_pad.SetLeftMargin(0.15)
            self.ratio_pad.SetRightMargin(0.15)
            self.ratio_pad.SetTopMargin(0.)   
            self.ratio_pad.SetGridy()
            self.ratio_pad.SetBottomMargin(0.3)
        
        else:
            self.canvas = Canvas(width=700, height=700) ; self.canvas.Draw()
            self.canvas.cd() ; self.main_pad   = Pad(0. , 0. , 1., 1.  )    ; self.main_pad .Draw()
            self.canvas.cd() ; self.ratio_pad  = Pad(-1., -1., -.9, -.9)    ; self.ratio_pad.Draw() # put it outside the canvas
            self.main_pad.SetTicks(True)
            self.main_pad.SetTopMargin(0.15)
            self.main_pad.SetBottomMargin(0.15)
            self.main_pad.SetLeftMargin(0.15)
            self.main_pad.SetRightMargin(0.15)

    def create_datacards(self, data, bkgs, signals, label, protect_empty_bins=['nonprompt']):  
        '''
        FIXME! For now this is specific to the data-driven case
        '''  
        # save a ROOT file with histograms, aka datacard
        datacard_dir = '/'.join([self.plt_dir, 'datacards'])
        makedirs(datacard_dir, exist_ok=True)
        outfile = ROOT.TFile.Open('/'.join([datacard_dir, 'datacard_%s.root' %label]), 'recreate')
        outfile.cd()
        
        # data in tight
        data.name = 'data_obs'
        data.Write()
        
        # reads off a dictionary
        for bkg_name, bkg in bkgs.items():
            bkg.name = bkg_name.split('#')[0]
            bkg.drawstyle = 'HIST E'
            bkg.color = 'black'
            bkg.linewidth = 2
            
            # manual protection against empty bins, that would make combine crash
            if bkg_name in protect_empty_bins:
                for ibin in bkg.bins_range():
                    if bkg.GetBinContent(ibin)<=0.:
                        bkg.SetBinContent(ibin, 1e-2)
                        bkg.SetBinError(ibin, np.sqrt(1e-2))
            
            bkg.Write()

        # signals
        for isig in signals:
            isig.name = isig.name.split('#')[0]
            isig.drawstyle = 'HIST E'
            isig.color = 'black'
            isig.Write()

            # print out the txt datacard
            with open('/'.join([datacard_dir, 'datacard_%s_%s.txt' %(label, isig.name)]), 'w') as card:
                card.write(
'''
imax 1 number of bins
jmax * number of processes minus 1
kmax * number of nuisance parameters
--------------------------------------------------------------------------------------------------------------------------------------------
shapes *    {cat} datacard_{cat}.root $PROCESS $PROCESS_$SYSTEMATIC
--------------------------------------------------------------------------------------------------------------------------------------------
bin               {cat}
observation       {obs:d}
--------------------------------------------------------------------------------------------------------------------------------------------
bin                                                     {cat}                          {cat}                            {cat}
process                                                 {signal_name}                  nonprompt                        prompt
process                                                 0                              1                                2
rate                                                    {signal:.4f}                   {nonprompt:.4f}                  {prompt:.4f}
--------------------------------------------------------------------------------------------------------------------------------------------
lumi                                    lnN             1.025                          -                                -   
norm_prompt_{ch}_{y}_{cat}                  lnN             -                              -                                1.15   
norm_nonprompt_{ch}_{y}_{cat}               lnN             -                              1.20                             -   
norm_sig_{ch}_{y}_{cat}                     lnN             1.2                            -                                -   
--------------------------------------------------------------------------------------------------------------------------------------------
{cat} autoMCStats 0 0 1
'''.format(cat         = label,
           obs         = int(data.integral()) if self.blinded==False else -1,
           signal_name = isig.name,
           signal      = isig.integral(),
           ch          = self.full_channel,
           y           = self.year,
           prompt      = bkgs['prompt'].integral(),
           nonprompt   = bkgs['nonprompt'].integral(),
           )
        )

        outfile.Close()

    def plot(self):

        evaluator = Evaluator(self.model, self.transformation, self.features)
        makedirs(self.plt_dir, exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'lin']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'log']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'lin', 'png']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'lin', 'root']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'log', 'png']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'log', 'root']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'lnt_region', 'lin']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'lnt_region', 'log']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'lnt_region', 'lin', 'png']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'lnt_region', 'lin', 'root']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'lnt_region', 'log', 'png']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'lnt_region', 'log', 'root']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'shapes', 'lin']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'shapes', 'log']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'shapes', 'lin', 'png']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'shapes', 'lin', 'root']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'shapes', 'log', 'png']), exist_ok=True)
        makedirs('/'.join([self.plt_dir, 'shapes', 'log', 'root']), exist_ok=True)
        if len(self.save_synch_tuple):
            makedirs('/'.join([self.plt_dir, 'synch_ntuples']), exist_ok=True)

        # NN evaluator
        print('============> starting reading the trees')
        print ('Plots will be stored in: ', self.plt_dir)
        now = time()
        signal = []
        if self.process_signals:
            signal = self.get_signal_samples(self.channel, self.base_dir, self.post_fix, self.selection_data, mini=self.mini_signals)
        else:
            signal = []  
        data = self.get_data_samples(self.channel, self.base_dir, self.post_fix, self.selection_data)
        mc = self.get_mc_samples(self.channel, self.base_dir, self.post_fix, self.selection_mc)
        print('============> it took %.2f seconds' %(time() - now))

        # evaluate FR
        for isample in (mc+data): #+signal):
            isample.df['fr'] = evaluator.evaluate(isample.df)
            # already corrected, ready to be applied in lnt-not-tight
            isample.df['fr_corr'] = isample.df['fr'] / (1. - isample.df['fr']) 
                 
        # apply an extra selection to the pandas dataframes
        if len(self.pandas_selection):
            for isample in (mc+data+signal):
                isample.df = isample.df.query(self.pandas_selection)

        # split the dataframe in tight and lnt-not-tight (called simply lnt for short)
        print('============> splitting dataframe in tight and loose not tight')
        for isample in (mc+data+signal):
            isample.df_tight = isample.df.query(self.selection_tight)
            if isample not in signal:
                isample.df_lnt = isample.df.query(self.selection_lnt)
            # free some mem
            del isample.df
            gc.collect()
        print('============> ... done')

        # sort depending on their position in the stack
        mc.sort(key = lambda x : x.position_in_stack)

        # now we plot 
        self.create_canvas(self.do_ratio)

        for var_index, ivar in enumerate(variables):
                        
            variable, bins, label, xlabel, ylabel, extra_sel = ivar.var, ivar.bins, ivar.label, ivar.xlabel, ivar.ylabel, ivar.extra_selection

            print('plotting', label)
                        
            ######################################################################################
            # plot MC stacks, in tight and lnt
            ######################################################################################
            
            stack_prompt    = []
            stack_nonprompt = []
            stack_nonprompt_control = []
            
            for imc in mc:
                
                if extra_sel:
                    mc_df_tight = imc.df_tight.query(extra_sel) 
                    mc_df_lnt = imc.df_lnt.query(extra_sel) 
                else:
                    mc_df_tight = imc.df_tight
                    mc_df_lnt = imc.df_lnt
                
                histo_tight = Hist(bins, title=imc.label, markersize=0, legendstyle='F', name=imc.datacard_name+'#'+label+'#t')
                weights = self.total_weight_calculator(mc_df_tight, ['weight']+imc.extra_signal_weights, [self.lumi, imc.lumi_scaling])
                histo_tight.fill_array(mc_df_tight[variable], weights=weights*self.relaxed_mc_scaling)

                histo_tight.fillstyle = 'solid'
                histo_tight.fillcolor = 'steelblue' if self.data_driven else imc.colour 
                histo_tight.linewidth = 0
                
                stack_prompt.append(histo_tight)
                    
                # optionally remove the MC subtraction in loose-not-tight
                # may help if MC stats is terrible (and it often is)
                if self.data_driven:
                    if self.mc_subtraction:
                        histo_lnt = Hist(bins, title=imc.label, markersize=0, legendstyle='F', name=imc.datacard_name+'#'+label+'#lnt')
                        weights = self.total_weight_calculator(mc_df_lnt, ['weight', 'fr_corr']+imc.extra_signal_weights, [-1., self.lumi, imc.lumi_scaling])
                        histo_lnt.fill_array(mc_df_lnt[variable], weights=weights*self.relaxed_mc_scaling)

                        histo_lnt.fillstyle = 'solid'
                        histo_lnt.fillcolor = 'skyblue' if self.data_driven else imc.colour
                        histo_lnt.linewidth = 0
                        stack_nonprompt.append(histo_lnt)

                    histo_lnt_control = Hist(bins, title=imc.label, markersize=0, legendstyle='F', name=imc.datacard_name+'#'+label+'#lntcontrol')
                    weights_control = self.total_weight_calculator(mc_df_lnt, ['weight']+imc.extra_signal_weights, [self.lumi, imc.lumi_scaling])
                    histo_lnt_control.fill_array(mc_df_lnt[variable], weights=weights_control*self.relaxed_mc_scaling)

                    histo_lnt_control.fillstyle = 'solid'
                    histo_lnt_control.fillcolor = imc.colour
                    histo_lnt_control.linewidth = 0
                
#                     print(histo_lnt_control)
#                     print(histo_lnt_control.fillcolor)
#                     print(imc.name, imc.colour)
#                     print(histo_lnt_control.integral())
                    stack_nonprompt_control.append(histo_lnt_control)

            # merge different samples together (add the histograms)                
            # prepare two temporary containers for the post-grouping histograms
            stack_prompt_tmp = []
            stack_nonprompt_tmp = []
            stack_nonprompt_control_tmp = []
            for ini, fin in [(stack_prompt           , stack_prompt_tmp           ), 
                             (stack_nonprompt        , stack_nonprompt_tmp        ), 
                             (stack_nonprompt_control, stack_nonprompt_control_tmp)]:
                for k, v in groups.items():
                    grouped = []
                    for ihist in ini:
                        if ihist.name.split('#')[0] in v:
                            grouped.append(ihist)
                        elif ihist.name.split('#')[0] not in togroup:
                            fin.append(ihist)
                    if len(grouped):
                        group = sum(grouped)
                        group.title = k
                        group.name = '#'.join([k] + ihist.name.split('#')[1:])
                        group.fillstyle = grouped[0].fillstyle
                        group.fillcolor = grouped[0].fillcolor
                        group.linewidth = grouped[0].linewidth
                    fin.append(group)

            stack_prompt            = stack_prompt_tmp 
            stack_nonprompt         = stack_nonprompt_tmp
            stack_nonprompt_control = stack_nonprompt_control_tmp
            
            ######################################################################################
            # plot the signals
            ######################################################################################
            
            all_signals     = []
            signals_to_plot = []
            
            for isig in signal:

                if variable not in self.datacards:
                    if not isig.toplot:
                        continue
                
                if variable=='fr' or variable=='fr_corr':
                    continue

                if extra_sel:
                    isig_df_tight = isig.df_tight.query(extra_sel) 
                else:
                    isig_df_tight = isig.df_tight

                histo_tight = Hist(bins, title=isig.label, markersize=0, legendstyle='L', name=isig.datacard_name+'#'+label) # the "#" thing is a trick to give hists unique name, else ROOT complains
                weights = self.total_weight_calculator(isig_df_tight, ['weight']+isig.extra_signal_weights, [self.lumi, isig.lumi_scaling])
                histo_tight.fill_array(isig_df_tight[variable], weights=weights)
                histo_tight.color     = isig.colour
                histo_tight.fillstyle = 'hollow'
                histo_tight.linewidth = 2
                histo_tight.linestyle = 'dashed'
                histo_tight.drawstyle = 'HIST'

                all_signals.append(histo_tight)
                if isig.toplot: signals_to_plot.append(histo_tight)
                
                if isig.name in self.save_synch_tuple and var_index==0:
                    landing_spot = '/'.join([self.plt_dir, 'synch_ntuples', isig.name + '_' + self.full_channel + '.root'])
                    # do not overwrite
                    if not os.path.isfile(landing_spot):
                        # save synch tree
                        isig_df_tight.to_root( landing_spot, key='tree' )
            
            ######################################################################################
            # plot the data
            ######################################################################################

            data_prompt    = []
            data_nonprompt = []
            data_nonprompt_control = []
            
            for idata in data:

                if extra_sel:
                    idata_df_tight = idata.df_tight.query(extra_sel) 
                    idata_df_lnt = idata.df_lnt.query(extra_sel) 
                else:
                    idata_df_tight = idata.df_tight
                    idata_df_lnt = idata.df_lnt

                histo_tight = Hist(bins, title=idata.label, markersize=1, legendstyle='LEP')
                histo_tight.fill_array(idata_df_tight[variable])
                
                data_prompt.append(histo_tight)

                if 'data' in self.save_synch_tuple and var_index==0:
                    landing_spot = '/'.join([self.plt_dir, 'synch_ntuples', 'data_' + self.full_channel + '.root'])
                    # save synch tree
                    idata_df_tight.to_root( landing_spot, key='tree', mode='a' )
                
                if self.data_driven:
                    histo_lnt = Hist(bins, title=idata.label, markersize=0, legendstyle='F')
                    histo_lnt.fill_array(idata_df_lnt[variable], weights=idata_df_lnt.fr_corr)
                
                    histo_lnt.fillstyle = 'solid'
                    histo_lnt.fillcolor = 'skyblue'
                    histo_lnt.linewidth = 0

                    histo_lnt_control = Hist(bins, title=idata.label, markersize=1, legendstyle='LEP')
                    histo_lnt_control.fill_array(idata_df_lnt[variable])
                                
                    data_nonprompt.append(histo_lnt)
                    data_nonprompt_control.append(histo_lnt_control)

            if self.data_driven:
                # put the prompt backgrounds together
                all_exp_prompt = sum(stack_prompt)
                all_exp_prompt.title = 'prompt'

                # put the nonprompt backgrounds together
                all_exp_nonprompt = sum(stack_nonprompt+data_nonprompt)
                all_exp_nonprompt.fillstyle = 'solid'
                all_exp_nonprompt.fillcolor = 'skyblue'
                all_exp_nonprompt.linewidth = 0               
                all_exp_nonprompt.title = 'nonprompt'

                # create the stacks
                stack = HistStack([all_exp_prompt, all_exp_nonprompt], drawstyle='HIST', title='')
                stack_control = HistStack(stack_nonprompt_control, drawstyle='HIST', title='')
            
            else:
                stack = HistStack(stack_prompt, drawstyle='HIST', title='')

            # stat uncertainty
            hist_error = stack.sum #sum([all_exp_prompt, all_exp_nonprompt])    
            hist_error.drawstyle = 'E2'
            hist_error.fillstyle = '/'
            hist_error.color     = 'gray'
            hist_error.title     = 'stat. unc.'
            hist_error.legendstyle = 'F'

            if self.data_driven:
                hist_error_control = stack_control.sum    
                hist_error_control.drawstyle = 'E2'
                hist_error_control.fillstyle = '/'
                hist_error_control.color     = 'gray'
                hist_error_control.title     = 'stat. unc.'
                hist_error_control.legendstyle = 'F'

            # put the data together
            all_obs_prompt = sum(data_prompt)
            all_obs_prompt.title = 'observed'

            if self.data_driven:
                all_obs_nonprompt_control = sum(data_nonprompt_control)
                all_obs_nonprompt_control.title = 'observed'
                all_obs_nonprompt_control.drawstyle = 'EP'

            # prepare the legend
            print(signals_to_plot)
            for jj in signals_to_plot: print(jj.name, jj.integral())
            if len(signals_to_plot):
                legend         = Legend([all_obs_prompt, stack, hist_error], pad=self.main_pad, leftmargin=0., rightmargin=0., topmargin=0., textfont=42, textsize=0.025, entrysep=0.01, entryheight=0.04)
                legend_signals = Legend(signals_to_plot                    , pad=self.main_pad, leftmargin=0., rightmargin=0., topmargin=0., textfont=42, textsize=0.025, entrysep=0.01, entryheight=0.04)
                legend_signals.SetBorderSize(0)
                legend_signals.x1 = 0.42
                legend_signals.y1 = 0.74
                legend_signals.x2 = 0.88
                legend_signals.y2 = 0.90
                legend_signals.SetFillColor(0)
                legend.SetBorderSize(0)
                legend.x1 = 0.2
                legend.y1 = 0.74
                legend.x2 = 0.45
                legend.y2 = 0.90
                legend.SetFillColor(0)
            else:
                legend = Legend([all_obs_prompt, stack, hist_error], pad=self.main_pad, leftmargin=0., rightmargin=0., topmargin=0., textfont=42, textsize=0.03, entrysep=0.01, entryheight=0.04)
                legend.SetBorderSize(0)
                legend.x1 = 0.55
                legend.y1 = 0.74
                legend.x2 = 0.88
                legend.y2 = 0.90
                legend.SetFillColor(0)
            

            # plot with ROOT, linear and log scale
            for islogy in [False, True]:

                things_to_plot = [stack, hist_error]
                if not self.blinded: 
                    things_to_plot.append(all_obs_prompt)
                
                # plot signals, as an option
                if self.plot_signals: 
                    things_to_plot += signals_to_plot
                
                # set the y axis range 
                # FIXME! setting it by hand to each object as it doesn't work if passed to draw
                if islogy : yaxis_max = 40.   * max([ithing.max() for ithing in things_to_plot])
                else      : yaxis_max =  1.65 * max([ithing.max() for ithing in things_to_plot])
                if islogy : yaxis_min = 0.01
                else      : yaxis_min = 0.

                for ithing in things_to_plot:
                    ithing.SetMaximum(yaxis_max)   
                draw(things_to_plot, xtitle=xlabel, ytitle=ylabel, pad=self.main_pad, logy=islogy)
                                
                # expectation uncertainty in the ratio pad
                ratio_exp_error = Hist(bins)
                ratio_data = Hist(bins)
                for ibin in hist_error.bins_range():
                    ratio_exp_error.set_bin_content(ibin, 1.)
                    ratio_exp_error.set_bin_error  (ibin, hist_error.get_bin_error(ibin)      / hist_error.get_bin_content(ibin) if hist_error.get_bin_content(ibin)!=0. else 0.)
                    ratio_data.set_bin_content     (ibin, all_obs_prompt.get_bin_content(ibin)/ hist_error.get_bin_content(ibin) if hist_error.get_bin_content(ibin)!=0. else 0.)
                    ratio_data.set_bin_error       (ibin, all_obs_prompt.get_bin_error(ibin)  / hist_error.get_bin_content(ibin) if hist_error.get_bin_content(ibin)!=0. else 0.)

                ratio_data.drawstyle = 'EP'
                ratio_data.title     = ''

                ratio_exp_error.drawstyle  = 'E2'
                ratio_exp_error.markersize = 0
                ratio_exp_error.title      = ''
                ratio_exp_error.fillstyle  = '/'
                ratio_exp_error.color      = 'gray'

                for ithing in [ratio_data, ratio_exp_error]:
                    ithing.xaxis.set_label_size(ithing.xaxis.get_label_size() * 3.) # the scale should match that of the main/ratio pad size ratio
                    ithing.yaxis.set_label_size(ithing.yaxis.get_label_size() * 3.) # the scale should match that of the main/ratio pad size ratio
                    ithing.xaxis.set_title_size(ithing.xaxis.get_title_size() * 3.) # the scale should match that of the main/ratio pad size ratio
                    ithing.yaxis.set_title_size(ithing.yaxis.get_title_size() * 3.) # the scale should match that of the main/ratio pad size ratio
                    ithing.yaxis.set_ndivisions(405)
                    ithing.yaxis.set_title_offset(0.4)
                    
                things_to_plot = [ratio_exp_error]
                if not self.blinded: 
                    things_to_plot.append(ratio_data)

                draw(things_to_plot, xtitle=xlabel, ytitle='obs/exp', pad=self.ratio_pad, logy=False, ylimits=(0.5, 1.5))

                line = ROOT.TLine(min(bins), 1., max(bins), 1.)
                line.SetLineColor(ROOT.kBlack)
                line.SetLineWidth(1)
                self.ratio_pad.cd()
                line.Draw('same')

#                 chi2_score_text = '\chi^{2}/NDF = %.1f' %(all_obs_prompt.Chi2Test(hist_error, 'UW CHI2/NDF'))
#                 chi2_score_text = 'p-value = %.2f' %(all_obs_prompt.Chi2Test(hist_error, 'UW'))
#                 chi2_score = ROOT.TLatex(0.7, 0.81, chi2_score_text)
#                 chi2_score.SetTextFont(43)
#                 chi2_score.SetTextSize(15)
#                 chi2_score.SetNDC()
#                 chi2_score.Draw('same')

                self.canvas.cd()
                # FIXME! add SS and OS channels
                if   self.full_channel == 'mmm': channel = '\mu\mu\mu'
                elif self.full_channel == 'eee': channel = 'eee'
                elif self.full_channel == 'mem_os': channel = '\mu^{\pm}\mu^{\mp}e'
                elif self.full_channel == 'mem_ss': channel = '\mu^{\pm}\mu^{\pm}e'
                elif self.full_channel == 'eem_os': channel = 'e^{\pm}e^{\mp}\mu'
                elif self.full_channel == 'eem_ss': channel = 'e^{\pm}e^{\pm}\mu'
                else: assert False, 'ERROR: Channel not valid.'
                finalstate = ROOT.TLatex(0.68, 0.68, channel)
                finalstate.SetTextFont(43)
                finalstate.SetTextSize(25)
                finalstate.SetNDC()
                finalstate.Draw('same')
                
                self.canvas.cd()
                # remove old legend
                for iprim in self.canvas.primitives:
                    if isinstance(iprim, Legend):
                        self.canvas.primitives.remove(iprim)
                legend.Draw('same')
                if self.plot_signals: 
                    legend_signals.Draw('same')
                CMS_lumi(self.main_pad, 4, 0, lumi_13TeV="%d, L = %.1f fb^{-1}" %(self.year, self.lumi/1000.))
                self.canvas.Modified()
                self.canvas.Update()
                for iformat in ['pdf', 'png', 'root']:
                    self.canvas.SaveAs('/'.join([self.plt_dir, 'log' if islogy else 'lin', iformat if iformat!='pdf' else '','%s%s.%s'  %(label, '_log' if islogy else '_lin', iformat)]))

                # plot distributions in loose not tight
                # check MC contamination there
                if self.data_driven and variable not in ['fr', 'fr_corr']:
                    things_to_plot = [stack_control, hist_error_control, all_obs_nonprompt_control]
                    # set the y axis range 
                    # FIXME! setting it by hand to each object as it doesn't work if passed to draw
                    if islogy : yaxis_max = 40.   * max([ithing.max() for ithing in things_to_plot])
                    else      : yaxis_max =  1.65 * max([ithing.max() for ithing in things_to_plot])
                    if islogy : yaxis_min = 0.01
                    else      : yaxis_min = 0.

                    for ithing in things_to_plot:
                        ithing.SetMaximum(yaxis_max)   
                        ithing.SetMinimum(yaxis_min)   
                    
                    draw(things_to_plot, xtitle=xlabel, ytitle=ylabel, pad=self.main_pad, logy=islogy, ylimits=(yaxis_min, max(1.,yaxis_max)))
                    
                    new_legend = Legend(stack_control.hists+[hist_error_control, all_obs_nonprompt_control], pad=self.main_pad, leftmargin=0., rightmargin=0., topmargin=0., textfont=42, textsize=0.03, entrysep=0.01, entryheight=0.04)
                    new_legend.SetBorderSize(0)
                    new_legend.x1 = 0.55
                    new_legend.y1 = 0.71
                    new_legend.x2 = 0.88
                    new_legend.y2 = 0.90
                    new_legend.SetFillColor(0)
                    
                    # divide MC to subtract by data
                    stack_nonprompt_control_scaled_list = []
                    for ihist in stack_control.hists:
                        new_hist = copy(ihist)
                        for ibin in new_hist.bins_range():                        
                            new_hist.SetBinContent(ibin, np.nan_to_num(np.divide(new_hist.GetBinContent(ibin), all_obs_nonprompt_control.GetBinContent(ibin))))
                            new_hist.SetBinError  (ibin, np.nan_to_num(np.divide(new_hist.GetBinError  (ibin), all_obs_nonprompt_control.GetBinContent(ibin))))
                        stack_nonprompt_control_scaled_list.append(new_hist)

                    stack_control_scaled = HistStack(stack_nonprompt_control_scaled_list, drawstyle='HIST', title='')
                    stack_control_scaled_err = stack_control_scaled.sum
                    stack_control_scaled_err.drawstyle = 'E2'
                    stack_control_scaled_err.fillstyle = '/'
                    stack_control_scaled_err.color     = 'gray'
                    stack_control_scaled_err.title     = 'stat. unc.'
                    stack_control_scaled_err.legendstyle = 'F'

                    draw([stack_control_scaled, stack_control_scaled_err], xtitle=xlabel, ytitle='MC/data', pad=self.ratio_pad, logy=False)

                    stack_control_scaled.xaxis.set_label_size(stack_control_scaled.xaxis.get_label_size() * 3.) # the scale should match that of the main/ratio pad size ratio
                    stack_control_scaled.yaxis.set_label_size(stack_control_scaled.yaxis.get_label_size() * 3.) # the scale should match that of the main/ratio pad size ratio
                    stack_control_scaled.xaxis.set_title_size(stack_control_scaled.xaxis.get_title_size() * 3.) # the scale should match that of the main/ratio pad size ratio
                    stack_control_scaled.yaxis.set_title_size(stack_control_scaled.yaxis.get_title_size() * 3.) # the scale should match that of the main/ratio pad size ratio
                    stack_control_scaled.yaxis.set_ndivisions(405)
                    stack_control_scaled.yaxis.set_title_offset(0.4)
                    stack_control_scaled.SetMinimum(0.)
                    stack_control_scaled.SetMaximum(1.5)

                    CMS_lumi(self.main_pad, 4, 0, lumi_13TeV="%d, L = %.1f fb^{-1}" %(self.year, self.lumi/1000.))
                    
                    self.canvas.cd()
                    # remove old legend
                    for iprim in self.canvas.primitives:
                        if isinstance(iprim, Legend):
                            self.canvas.primitives.remove(iprim)
                    
                    # draw new legend    
                    new_legend.Draw('same')

                    self.canvas.Modified()
                    self.canvas.Update()

                    for iformat in ['pdf', 'png', 'root']:
                        self.canvas.SaveAs('/'.join([self.plt_dir, 'lnt_region', 'log' if islogy else 'lin', iformat if iformat!='pdf' else '','%s%s.%s'  %(label, '_log' if islogy else '_lin', iformat)]))
                
                    # compare shapes in tight and loose not tight
                    
                    # data in tight
                    all_obs_prompt_norm = copy(all_obs_prompt)
                    if all_obs_prompt_norm.integral()>0.:
                        all_obs_prompt_norm.Scale(np.nan_to_num(np.divide(1., all_obs_prompt_norm.integral())))
                    all_obs_prompt_norm.drawstyle = 'hist e'
                    all_obs_prompt_norm.linecolor = 'black'
                    all_obs_prompt_norm.markersize = 0
                    all_obs_prompt_norm.legendstyle='LE'
                    all_obs_prompt_norm.title=''
                    all_obs_prompt_norm.label='data - tight'
                    
                    # data MC subtracted in loose
                    all_obs_prompt_mc_sub_norm = copy(all_obs_prompt)
                    all_obs_prompt_mc_sub_norm.add(all_exp_prompt, -1)
                    if all_obs_prompt_mc_sub_norm.integral()>0.:
                        all_obs_prompt_mc_sub_norm.Scale(np.nan_to_num(np.divide(1., all_obs_prompt_mc_sub_norm.integral())))
                    all_obs_prompt_mc_sub_norm.drawstyle = 'hist e'
                    all_obs_prompt_mc_sub_norm.linecolor = 'green'
                    all_obs_prompt_mc_sub_norm.markersize = 0
                    all_obs_prompt_mc_sub_norm.legendstyle='LE'
                    all_obs_prompt_mc_sub_norm.title=''
                    all_obs_prompt_mc_sub_norm.label='(data-MC) - tight'

                    # data in loose
                    all_obs_nonprompt_control_norm = copy(all_obs_nonprompt_control)
                    if all_obs_nonprompt_control_norm.integral()>0.:
                        all_obs_nonprompt_control_norm.Scale(np.nan_to_num(np.divide(1., all_obs_nonprompt_control_norm.integral())))
                    all_obs_nonprompt_control_norm.drawstyle = 'hist e'
                    all_obs_nonprompt_control_norm.linecolor = 'red'
                    all_obs_nonprompt_control_norm.markersize = 0
                    all_obs_nonprompt_control_norm.legendstyle='LE'
                    all_obs_nonprompt_control_norm.title=''
                    all_obs_nonprompt_control_norm.label='data - l-n-t'
                    
                    # data MC subtracted in loose
                    all_obs_nonprompt_control_mc_sub_norm = copy(all_obs_nonprompt_control)
                    all_obs_nonprompt_control_mc_sub_norm.add(stack_control.sum, -1)
                    if all_obs_nonprompt_control_mc_sub_norm.integral()>0.:
                        all_obs_nonprompt_control_mc_sub_norm.Scale(np.nan_to_num(np.divide(1., all_obs_nonprompt_control_mc_sub_norm.integral())))
                    all_obs_nonprompt_control_mc_sub_norm.drawstyle = 'hist e'
                    all_obs_nonprompt_control_mc_sub_norm.linecolor = 'blue'
                    all_obs_nonprompt_control_mc_sub_norm.markersize = 0
                    all_obs_nonprompt_control_mc_sub_norm.legendstyle='LE'
                    all_obs_nonprompt_control_mc_sub_norm.title=''
                    all_obs_nonprompt_control_mc_sub_norm.label='(data-MC) - l-n-t'
                                                            
                    things_to_plot = [
                        all_obs_prompt_norm,
                        all_obs_prompt_mc_sub_norm,
                        all_obs_nonprompt_control_norm,
                        all_obs_nonprompt_control_mc_sub_norm,
                    ]
                    
                    yaxis_max = max([ii.GetMaximum() for ii in things_to_plot])
                    
                    draw(things_to_plot, xtitle=xlabel, ytitle=ylabel, pad=self.main_pad, logy=islogy, ylimits=(yaxis_min, max(1., 1.55*yaxis_max)))

                    self.canvas.cd()
                    # remove old legend
                    for iprim in self.canvas.primitives:
                        if isinstance(iprim, Legend):
                            self.canvas.primitives.remove(iprim)

                    shape_legend = Legend([], pad=self.main_pad, leftmargin=0., rightmargin=0., topmargin=0., textfont=42, textsize=0.03, entrysep=0.01, entryheight=0.04)
                    shape_legend.AddEntry(all_obs_prompt_norm                  , all_obs_prompt_norm                  .label, all_obs_prompt_norm                  .legendstyle)
                    shape_legend.AddEntry(all_obs_prompt_mc_sub_norm           , all_obs_prompt_mc_sub_norm           .label, all_obs_prompt_mc_sub_norm           .legendstyle)
                    shape_legend.AddEntry(all_obs_nonprompt_control_norm       , all_obs_nonprompt_control_norm       .label, all_obs_nonprompt_control_norm       .legendstyle)
                    shape_legend.AddEntry(all_obs_nonprompt_control_mc_sub_norm, all_obs_nonprompt_control_mc_sub_norm.label, all_obs_nonprompt_control_mc_sub_norm.legendstyle)
                    shape_legend.SetBorderSize(0)
                    shape_legend.x1 = 0.50
                    shape_legend.y1 = 0.71
                    shape_legend.x2 = 0.88
                    shape_legend.y2 = 0.90
                    shape_legend.SetFillColor(0)
                    shape_legend.Draw('same')

                    # plot ratios
                    all_obs_prompt_norm_ratio                   = copy(all_obs_prompt_norm                  )
                    all_obs_prompt_mc_sub_norm_ratio            = copy(all_obs_prompt_mc_sub_norm           )
                    all_obs_nonprompt_control_norm_ratio        = copy(all_obs_nonprompt_control_norm       )
                    all_obs_nonprompt_control_mc_sub_norm_ratio = copy(all_obs_nonprompt_control_mc_sub_norm)

                    all_obs_prompt_norm_ratio                  .Divide(all_obs_prompt_mc_sub_norm_ratio)
                    all_obs_nonprompt_control_norm_ratio       .Divide(all_obs_prompt_mc_sub_norm_ratio)
                    all_obs_nonprompt_control_mc_sub_norm_ratio.Divide(all_obs_prompt_mc_sub_norm_ratio)

                    things_to_plot_ratio = [
                        all_obs_prompt_norm_ratio                  ,
                        all_obs_nonprompt_control_norm_ratio       ,
                        all_obs_nonprompt_control_mc_sub_norm_ratio,
                    ]
                    
                    for ithing in things_to_plot_ratio:
                        ithing.xaxis.set_label_size(ithing.xaxis.get_label_size() * 3.) # the scale should match that of the main/ratio pad size ratio
                        ithing.yaxis.set_label_size(ithing.yaxis.get_label_size() * 3.) # the scale should match that of the main/ratio pad size ratio
                        ithing.xaxis.set_title_size(ithing.xaxis.get_title_size() * 3.) # the scale should match that of the main/ratio pad size ratio
                        ithing.yaxis.set_title_size(ithing.yaxis.get_title_size() * 3.) # the scale should match that of the main/ratio pad size ratio
                        ithing.yaxis.set_ndivisions(405)
                        ithing.yaxis.set_title_offset(0.4)
                        ithing.SetMinimum(0.)
                        ithing.SetMaximum(2.)
                    
                    draw(things_to_plot_ratio, xtitle=xlabel, ytitle='1/(data-MC)_{tight}', pad=self.ratio_pad, logy=False, ylimits=(0., 2.))
                    self.ratio_pad.cd()
                    line.Draw('same')
                    
                    CMS_lumi(self.main_pad, 4, 0, lumi_13TeV="%d, L = %.1f fb^{-1}" %(self.year, self.lumi/1000.))

                    self.canvas.Modified()
                    self.canvas.Update()

                    for iformat in ['pdf', 'png', 'root']:
                        self.canvas.SaveAs('/'.join([self.plt_dir, 'shapes', 'log' if islogy else 'lin', iformat if iformat!='pdf' else '','%s%s.%s'  %(label, '_log' if islogy else '_lin', iformat)]))
                    
                    
            # save only the datacards you want, don't flood everything
            if len(self.datacards) and label not in self.datacards:
                continue
            
            # FIXME! allow it to save datacards even for non data driven bkgs            
            if self.data_driven:    
                self.create_datacards(data=all_obs_prompt, 
                                      bkgs={'prompt':all_exp_prompt, 'nonprompt':all_exp_nonprompt}, 
                                      signals=all_signals, 
                                      label=label)  
