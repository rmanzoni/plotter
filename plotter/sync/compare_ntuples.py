import ROOT
import pandas as pd
from collections import OrderedDict
from branches import branches
from root_pandas import read_root

ROOT.gStyle.SetOptStat(0)
ROOT.gROOT.SetBatch(True)

##########################################################################################
##      CONTROL PANEL
##########################################################################################

# FIXME! use an argparser

# base_dir = '/Users/manzoni/Documents/HNL/plotter/plots/2017/mmm/sideband_inverted_m3l_200820_14h_5m/synch_ntuples/'
# base_dir = '/Users/manzoni/Documents/HNL/plotter/plots/2017/mmm/sideband_inverted_m3l_no_bjet_veto_200820_14h_7m/synch_ntuples/'
base_dir = '/Users/manzoni/Documents/HNL/plotter/plots/2017/mmm/sideband_inverted_m3l_no_bjet_veto_200820_17h_56m/synch_ntuples/'

f1_name = '/'.join([base_dir, 'data_mmm.root'])
f2_name = '/'.join([base_dir, 'tree_xcheck_data.root'])
# f2_name = '/'.join([base_dir, 'tree_xcheck_data_17_nobveto.root'])
# f2_name = '/'.join([base_dir, 'tree_xcheck_data_2017.root'])

t1_name = 'tree'
t2_name = 'tree_data'

moniker1 = 'ric' 
moniker2 = 'mar' 

name_of_comparison = 'data_2017_mmm_m3l_sideband_no_b_veto'

##########################################################################################
##########################################################################################

def compare(t1, t2, branches, file_name='comparison'):

    canvas = ROOT.TCanvas('c1', '', 700, 700) ; canvas.Draw()
    canvas.cd() ; main_pad   = ROOT.TPad('main' , '', 0.  , 0.25, 1. , 1.  ) ; main_pad .Draw()
    canvas.cd() ; ratio_pad  = ROOT.TPad('ratio', '', 0.  , 0.  , 1. , 0.25) ; ratio_pad.Draw()

    main_pad.SetTicks(True)
    main_pad.SetBottomMargin(0.)
    main_pad.SetLeftMargin(0.15)
    main_pad.SetRightMargin(0.15)
    ratio_pad.SetLeftMargin(0.15)
    ratio_pad.SetRightMargin(0.15)
    ratio_pad.SetTopMargin(0.)   
    ratio_pad.SetGridy()
    ratio_pad.SetBottomMargin(0.3)

    for ii, ibranch in enumerate(branches):
        if any(ibranch) == None:
            continue
        if len(ibranch)<3: 
            continue

        main_pad.cd()

        h1 = ROOT.TH1F('histo_%d_'%ii + ibranch[0], '', len(ibranch[2])-1, ibranch[2])
        h1.SetLineColor(ROOT.kRed)
        h1.SetLineWidth(2)
        h1.GetXaxis().SetTitle(ibranch[0])
        h1.GetYaxis().SetTitle('events')

        h2 = h1.Clone()
        h2.SetName('histo_' + ibranch[1])
        h2.SetLineColor(ROOT.kBlue)

        t1.Draw('%s >> %s' %(ibranch[0], h1.GetName()))
        t2.Draw('%s >> %s' %(ibranch[1], h2.GetName()))

        h1e = h1.Clone()
        h1e.SetFillStyle(3345)
        h1e.SetFillColor(h1.GetLineColor())

        h2e = h2.Clone()
        h2e.SetFillStyle(3354)
        h2e.SetFillColor(h2.GetLineColor())

        toplot = [h1, h2]
        toplot.sort(key = lambda x : x.GetMaximum(), reverse = True)
        toplot += [h1e, h2e]

        toplot[0].SetMaximum(1.3 * toplot[0].GetMaximum())
        toplot[0].SetMinimum(0.)
        toplot[0].Draw('hist')
        toplot[1].Draw('hist same')
        toplot[2].Draw('e2 same')
        toplot[3].Draw('e2 same')

        leg = ROOT.TLegend(0.12, 0.91, 0.9, 1.)
        leg.SetBorderSize(0)
        leg.SetFillColor(0)
        leg.AddEntry(h1, '%s - %.1f events - mean %.1f - std %.1f' %(moniker1, h1.Integral(), h1.GetMean(), h1.GetRMS()))
        leg.AddEntry(h2, '%s - %.1f events - mean %.1f - std %.1f' %(moniker2, h2.Integral(), h2.GetMean(), h2.GetRMS()))
        leg.Draw('same')

        ratio_pad.cd()

        ratio = h1.Clone()
        ratio.Divide(h2)
        ratio.Draw()

        ratio.GetYaxis().SetTitle('red / blue')

        ratio.GetXaxis().SetLabelSize(ratio.GetXaxis().GetLabelSize() * 3.)
        ratio.GetYaxis().SetLabelSize(ratio.GetYaxis().GetLabelSize() * 3.)
        ratio.GetXaxis().SetTitleSize(ratio.GetXaxis().GetTitleSize() * 3.)
        ratio.GetYaxis().SetTitleSize(ratio.GetYaxis().GetTitleSize() * 3.)
        ratio.GetYaxis().SetNdivisions(405)
        ratio.GetYaxis().SetTitleOffset(0.4)
        ratio.SetMinimum(0.)
        ratio.SetMaximum(2.)

        line = ROOT.TLine(min(ibranch[2]), 1., max(ibranch[2]), 1.)
        line.SetLineColor(ROOT.kBlack)
        line.SetLineWidth(1)
        line.Draw('same')

        canvas.Modified()
        canvas.Update()

        terminator = ''
        if ii == 0: terminator = '('
        if ii == (len(skimmed_branches)-1): terminator = ')'

        canvas.Print('%s.pdf%s' %(file_name, terminator) )

    import pdb ; pdb.set_trace() # otherwise it crashes, root demmerda
    print('%s done' %file_name)
            
##########################################################################################
##########################################################################################

if __name__ == '__main__':

    f1 = ROOT.TFile.Open(f1_name, 'read')
    f1.cd()
    t1 = f1.Get(t1_name)

    f2 = ROOT.TFile.Open(f2_name, 'read')
    f2.cd()
    t2 = f2.Get(t2_name)

    skimmed_branches = []
    for ii, ibranch in enumerate(branches):
        if any(ibranch) == None:
            continue
        if len(ibranch)<3: 
            continue
        skimmed_branches.append(ibranch)

    # compare the ntuples as they are
    compare(t1, t2, skimmed_branches, 'all_events')
    
    # find the events in common
    # convert the trees into pandas dataframes
    df1 = read_root(f1_name, key = t1_name, columns = [ib[0] for ib in skimmed_branches] + ['run', 'lumi', 'event'])
    df2 = read_root(f2_name, key = t2_name, columns = [ib[1] for ib in skimmed_branches] + ['_runNb', '_lumiBlock', '_eventNb'])
        
    new_names = dict( zip([ib[1] for ib in skimmed_branches], [ib[0] for ib in skimmed_branches]) )
    new_names['_runNb'    ] = 'run'  
    new_names['_lumiBlock'] = 'lumi' 
    new_names['_eventNb'  ] = 'event'
    df2 = df2.rename(columns=new_names)
    
    df1_events = df1[['run', 'lumi', 'event']].copy(deep=False)
    df2_events = df2[['run', 'lumi', 'event']].copy(deep=False)

    # events in common between the two dataframes
    common = pd.merge(df1, df2, how='inner', on=['run', 'lumi', 'event'])
    common_events = common[['run', 'lumi', 'event']].copy(deep=False)
    df1_common    = pd.merge(df1, common_events, how='inner', on=['run', 'lumi', 'event'])
    df2_common    = pd.merge(df2, common_events, how='inner', on=['run', 'lumi', 'event'])

    # event IDs in 1 not in 2
    in_1_not_in_2_event_ids = pd.concat([df1_events, df2_events, df2_events]).drop_duplicates(keep=False)
    # full information in 1 not in 2
    in_1_not_in_2 = pd.merge(df1, in_1_not_in_2_event_ids, how='inner', on=['run', 'lumi', 'event'])
    # event IDs in 2 not in 1
    in_2_not_in_1_event_ids = pd.concat([df2_events, df1_events, df1_events]).drop_duplicates(keep=False)
    # full information in 2 not in 1
    in_2_not_in_1 = pd.merge(df2, in_2_not_in_1_event_ids, how='inner', on=['run', 'lumi', 'event'])
    
    # save root trees
    df1          .to_root( '/'.join([base_dir, '%s.root' %name_of_comparison]), key='original_tree_%s'     %moniker1                         )
    df2          .to_root( '/'.join([base_dir, '%s.root' %name_of_comparison]), key='original_tree_%s'     %moniker2            , mode = 'a' )
    df1_common   .to_root( '/'.join([base_dir, '%s.root' %name_of_comparison]), key='common_tree_%s'       %moniker1            , mode = 'a' )
    df2_common   .to_root( '/'.join([base_dir, '%s.root' %name_of_comparison]), key='common_tree_%s'       %moniker2            , mode = 'a' )
    in_1_not_in_2.to_root( '/'.join([base_dir, '%s.root' %name_of_comparison]), key='in_%s_not_in_%s_tree' %(moniker1, moniker2), mode = 'a' )
    in_2_not_in_1.to_root( '/'.join([base_dir, '%s.root' %name_of_comparison]), key='in_%s_not_in_%s_tree' %(moniker2, moniker1), mode = 'a' )


    
