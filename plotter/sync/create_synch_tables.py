import ROOT
from collections import OrderedDict

disp_bins = [
  'lt_0p5',
  '0p5_to_1p5',
  '1p5_to_4p0',
  'mt_4p0',
]

integral_low_data       = OrderedDict()
integral_low_nonprompt  = OrderedDict()
integral_low_prompt     = OrderedDict()
integral_low_signal     = OrderedDict()

integral_high_data      = OrderedDict()
integral_high_nonprompt = OrderedDict()
integral_high_prompt    = OrderedDict()
integral_high_signal    = OrderedDict()

for disp_bin in disp_bins:

    ff = ROOT.TFile.Open('datacard_hnl_m_12_lxy_%s.root' %disp_bin, 'read')
    ff.cd()
    list_of_histos = list(map(ROOT.TKey.GetName, ff.GetListOfKeys()))
    list_of_histos.sort()

    h_data      = ff.Get('data_obs')
    h_nonprompt = ff.Get('nonprompt')
    h_prompt    = ff.Get('prompt') 
#     h_signal    = ff.Get('hnl_m_10_v2_7p0Em07_majorana')
    h_signal    = ff.Get('hnl_m_6_v2_4p1Em06_majorana') # 2017 & 2018
#     h_signal    = ff.Get('hnl_m_10_v2_5p7Em07_majorana') # 2016 mu
#     h_signal    = ff.Get('hnl_m_7_v2_6p0Em05_majorana') # 2016 e
    
    # find bin index corresponding to 4
    thrs = h_signal.FindBin(4)

    # integrals
    integral_low_data      [disp_bin] = h_data     .Integral(0, thrs-1)
    integral_low_nonprompt [disp_bin] = h_nonprompt.Integral(0, thrs-1)
    integral_low_prompt    [disp_bin] = h_prompt   .Integral(0, thrs-1)
    integral_low_signal    [disp_bin] = h_signal   .Integral(0, thrs-1)

    integral_high_data     [disp_bin] = h_data     .Integral(thrs, h_data     .GetNbinsX())
    integral_high_nonprompt[disp_bin] = h_nonprompt.Integral(thrs, h_nonprompt.GetNbinsX())
    integral_high_prompt   [disp_bin] = h_prompt   .Integral(thrs, h_prompt   .GetNbinsX())
    integral_high_signal   [disp_bin] = h_signal   .Integral(thrs, h_signal   .GetNbinsX())

    ff.Close()

for (int_low, int_high) in [ (integral_low_nonprompt, integral_high_nonprompt),
                             (integral_low_prompt   , integral_high_prompt   ),
                             (integral_low_signal   , integral_high_signal   ),]:

    print()
    
    for disp_bin in disp_bins:
        
        print( '%.3f\t%.3f' %(int_low[disp_bin], int_high[disp_bin]), end='\t')



