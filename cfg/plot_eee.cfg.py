from plotter import Plotter

plotter = Plotter (channel        = 'eee',
                   base_dir        = '/Users/cesareborgia/cernbox/ntuples/2018/',
                   post_fix        = 'HNLTreeProducer_eee/tree.root',
                   lumi           = 59700.,
                   model          = 'NN/eee/net_model_weighted.h5', 
                   transformation = 'NN/eee/input_tranformation_weighted.pck',
                   features       = 'NN/eee/input_features.pck',
                   )

plotter.plot()
