from CMGTools.HNL.samples.signals_2016 import all_signals as sigs_16
from CMGTools.HNL.samples.signals_2017 import all_signals as sigs_17
from CMGTools.HNL.samples.signals_2018 import all_signals_e as sigs_18
from re import sub
from pdb import set_trace

# for signals in [sigs_16, sigs_17, sigs_18]:
for signals in [sigs_18]:#, sigs_17, sigs_18]:

    if   signals == sigs_16: year = '16'
    elif signals == sigs_17: year = '17'
    elif signals == sigs_18: year = '18'

    with open('plotter_signals_%s.txt' %year, 'w') as f:

        for sig in signals:

            # set_trace()
            v2_label = sub('\.', 'p', '{v2:.1E}'.format(v2=sig.v2))
            v2_label = sub('-', 'm', v2_label)
            if  'Dirac_cc' in sig.name: 
                mode = 'Dirac_cc'
                mode_label = 'dirac_cc' 
            elif 'Dirac' in sig.name:
                mode = 'Dirac'
                mode_label = 'dirac' 
            elif not 'Dirac' in sig.name: 
                mode = 'Majorana'
                mode_label = 'majorana' 
            else:  assert False, 'ERROR: No mode could be determined. Check signal list.'

            label = 'hnl_m_{mass}_v2_{v2_label}_{mode_label}'.format(mass=sig.mass, v2_label=v2_label, mode_label=mode_label)
            if   sig.mass == 1:  color = 'darkorange'
            elif sig.mass == 2:  color = 'forestgreen'
            elif sig.mass == 3:  color = 'firebrick'
            elif sig.mass == 4:  color = 'indigo'
            elif sig.mass == 5:  color = 'chocolate'
            elif sig.mass == 6:  color = 'olive'
            elif sig.mass == 8:  color = 'darkgray'
            elif sig.mass == 10: color = 'teal'
            elif sig.mass == 15: color = 'gold'
            elif sig.mass == 20: color = 'crimson'
            else: assert False, 'ERROR: No mass could be determined. Check signal list.'

            print >> f, "Sample('{name}'\t\t, channel, 'HNL m = {mass}, V^{S} = {v2:.1E}, {mode}' , selection, '{label}',".format(name=sig.name, mass=sig.mass, \
                                                                                                                                 S='{2}', v2=sig.v2, mode=mode, label=label) 
            print >> f, "       '{color}' \t,10, base_dir, post_fix, False, True, True, 1.,  {xs}    , toplot=False),".format(color=color, xs=sig.xs)
            print >> f, ''


    f.close()

