from selections import selections
from plotter_cfgable import Plotter

plotter = Plotter (channel        = 'mmm',
                   basedir        = '/Users/cesareborgia/cernbox/2018_new/mmm/',
                   postfix        = 'HNLTreeProducer/tree.root',
                   lumi           = 59700.,
                   selection_data = selections['baseline'],
                   selection_mc   = '&'.join([selections['baseline'], selections['ispromptlepton']]),
                   model          = 'net_model_weighted.h5', 
                   transformation = 'input_tranformation_weighted.pck',
                   features       = 'input_features.pck',
                   )

plotter.plot()
