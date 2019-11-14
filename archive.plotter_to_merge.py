# https://indico.cern.ch/event/759388/contributions/3306849/attachments/1816254/2968550/root_conda_forge.pdf
# https://conda-forge.org/feedstocks/
import os
import re
import time
import ROOT
import root_pandas
import numpy as np
import pandas as pd
<<<<<<< HEAD
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import OrderedDict
from selections import Selections
from evaluate_nn import Evaluator
from sample import Sample, get_data_samples, get_mc_samples, get_signal_samples
from variables import variables

from rootpy.plotting import Hist, HistStack, Legend, Canvas, Graph
from rootpy.plotting.style import get_style, set_style
from rootpy.plotting.utils import draw
from pdb import set_trace

import logging
logging.disable(logging.DEBUG)

ROOT.gROOT.SetBatch(True)

class Plotter(object):

    def __init__(self          , 
                 channel       , 
                 base_dir       ,
                 post_fix       ,
                 lumi          ,
                 model         , 
                 transformation,
                 features       ):

        self.channel        = channel 
        self.base_dir        = base_dir 
        self.post_fix        = post_fix 
        self.lumi           = lumi
        self.model          = model          
        self.transformation = transformation 
        self.features       = features       

    def plot(self):

        evaluator = Evaluator(self.model, self.transformation, self.features)

        cuts = Selections(self.channel)
        selection_data = cuts.selections['baseline']
        selection_mc   = ' & '.join( [cuts.selections['baseline'], cuts.selections['is_prompt_lepton']] )

# NN evaluator

        print('============> starting reading the trees')
        now = time.time()
        signal = get_signal_samples(self.channel, self.base_dir+'all_channels/', self.post_fix, selection_data)
        mc     = get_mc_samples    (self.channel, self.base_dir+'all_channels/', self.post_fix, selection_mc)
        data   = get_data_samples  (self.channel, self.base_dir+'%s/'%self.channel, 'HNLTreeProducer/tree.root', selection_data)
        print('============> it took %.2f seconds' %(time.time() - now))

# evaluate FR
        for isample in (mc+data):
            isample.df['fr'] = evaluator.evaluate(isample.df)
            # already corrected, ready to be applied in loose-not-tight
            isample.df['fr_corr'] = isample.df['fr'] / (1. - isample.df['fr']) 

# split the dataframe in tight and loose-not-tight (called simply loose for short)
        for isample in (mc+data+signal):
            isample.df_tight = isample.df.query(cuts.selections_df['tight'])
            isample.df_loose = isample.df.query('not(%s)'%cuts.selections_df['tight'])

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
                canvas.SaveAs('plots/%s_%s%s.pdf' %(self.channel, label, islogy*'_log'))

            # save a ROOT file with histograms, aka datacard
            outfile = ROOT.TFile.Open('plots/%s_%s.datacard.root' %(self.channel, label), 'recreate')
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
=======
# import modin.pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import OrderedDict
from selections import selections, selections_df
from evaluate_nn import Evaluator
from sample import Sample, get_data_samples, get_mc_samples, get_signal_samples
from variables import variables
from cmsstyle import CMS_lumi

# os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray

from rootpy.plotting import Hist, HistStack, Legend, Canvas, Graph, Pad
from rootpy.plotting.style import get_style, set_style
from rootpy.plotting.utils import draw

import logging
logging.disable(logging.DEBUG)

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(False)

basedir        = '/Users/manzoni/Documents/efficiencyNN/HNL/mmm/ntuples/'
postfix        = 'HNLTreeProducer/tree.root'
lumi           = 59700. # fb-1
selection_data = selections['baseline']
selection_mc   = '&'.join([selections['baseline'], selections['ispromptlepton']])
plot_signals   = True
blinded        = True

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
canvas = Canvas(width=700, height=700) ; canvas.Draw()
canvas.cd() ; main_pad  = Pad(0., 0.25, 1., 1.  ) ; main_pad .Draw()
canvas.cd() ; ratio_pad = Pad(0., 0.  , 1., 0.25) ; ratio_pad.Draw()

main_pad.SetTicks(True)
main_pad.SetBottomMargin(0.)
main_pad.SetLeftMargin(0.15)
main_pad.SetRightMargin(0.15)
ratio_pad.SetLeftMargin(0.15)
ratio_pad.SetRightMargin(0.15)
ratio_pad.SetTopMargin(0.)   
ratio_pad.SetGridy()
ratio_pad.SetBottomMargin(0.3)

for ivar in variables:
    
    variable, bins, label, xlabel, ylabel, extra_sel = ivar.var, ivar.bins, ivar.label, ivar.xlabel, ivar.ylabel, ivar.extra_selection
    
    print('plotting', label)
    
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
        histo_tight.fill_array(mc_df_tight[variable], weights=lumi * mc_df_tight.weight * imc.lumi_scaling * mc_df_tight.lhe_weight)

        histo_tight.fillstyle = 'solid'
        histo_tight.fillcolor = 'steelblue'
        histo_tight.linewidth = 0

        stack_prompt.append(histo_tight)

        histo_loose = Hist(bins, title=imc.label, markersize=0, legendstyle='F')
        histo_loose.fill_array(mc_df_loose[variable], weights=-1.* lumi * mc_df_loose.weight * imc.lumi_scaling * mc_df_loose.lhe_weight * mc_df_loose.fr_corr)

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
        histo_tight.fill_array(isig_df_tight[variable], weights=lumi * isig_df_tight.weight * isig.lumi_scaling * isig_df_tight.lhe_weight)
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
    if len(signals_to_plot):
        legend = Legend([all_obs_prompt, stack, hist_error] + signals_to_plot, pad=main_pad, leftmargin=0.28, rightmargin=0.3, topmargin=-0.01, textsize=0.023, textfont=42, entrysep=0.01, entryheight=0.04)
    else:
        legend = Legend([all_obs_prompt, stack, hist_error], pad=main_pad, leftmargin=0.33, rightmargin=0.1, topmargin=-0.01, textsize=0.023, textfont=42, entrysep=0.012, entryheight=0.06)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)

    # plot with ROOT, linear and log scale
    for islogy in [False, True]:
    
        things_to_plot = [stack, hist_error, all_obs_prompt]
        
        # plot signals, as an option
        if plot_signals: 
            things_to_plot += signals_to_plot
        
        # set the y axis range 
        # FIXME! setting it by hand to each object as it doesn't work if passed to draw
        yaxis_max = 1.4 * max([ithing.max() for ithing in things_to_plot])
        for ithing in things_to_plot:
            ithing.SetMaximum(yaxis_max)   
                    
        draw(things_to_plot, xtitle=xlabel, ytitle=ylabel, pad=main_pad, logy=islogy)

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
            
        draw([ratio_exp_error, ratio_data], xtitle=xlabel, ytitle='obs/exp', pad=ratio_pad, logy=False, ylimits=(0.5, 1.5))

        line = ROOT.TLine(min(bins), 1., max(bins), 1.)
        line.SetLineColor(ROOT.kBlack)
        line.SetLineWidth(1)
        ratio_pad.cd()
        line.Draw('same')

        canvas.cd()

        finalstate = ROOT.TLatex(0.7, 0.85, '\mu\mu\mu')
        finalstate.SetTextFont(43)
        finalstate.SetTextSize(25)
        finalstate.SetNDC()
        finalstate.Draw('same')

        legend.Draw('same')
        CMS_lumi(main_pad, 4, 0)
        canvas.Modified()
        canvas.Update()
        canvas.SaveAs('%s%s.pdf' %(label, islogy*'_log'))


    # save a ROOT file with histograms, aka datacard
    outfile = ROOT.TFile.Open('datacard_%s.root' %label, 'recreate')
    outfile.cd()
#     outfile.mkdir(label)
#     outfile.cd(label)
    
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

        # print out the txt datacard
        with open('datacard_%s_%s.txt' %(label, isig.name) , 'w') as card:
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
norm_prompt_{cat}                       lnN             -                              -                                1.15   
norm_nonprompt_{cat}                    lnN             -                              1.20                             -   
norm_sig_{cat}                          lnN             1.2                            -                                -   
--------------------------------------------------------------------------------------------------------------------------------------------
{cat} autoMCStats 0 0 1
'''.format(cat         = label,
           obs         = all_obs_prompt.integral() if blinded==False else -1,
           signal_name = isig.name,
           signal      = isig.integral(),
           prompt      = all_exp_prompt.integral(),
           nonprompt   = all_exp_nonprompt.integral(),
           )
        )

    outfile.Close()
>>>>>>> 74472660f01039d1d9356a633c9d5d21fe4015e4
