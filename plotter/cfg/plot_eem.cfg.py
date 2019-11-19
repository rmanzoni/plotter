from plotter import Plotter

plotter = Plotter (channel        = 'eem',
                   base_dir        = '/Users/cesareborgia/cernbox/ntuples/2018/eem/',
                   post_fix        = 'HNLTreeProducer/tree.root',
                   lumi           = 59700.,
                   model          = 'NN/mmm/12Nov19_v0/net_model_weighted.h5', 
                   transformation = 'NN/mmm/12Nov19_v0/input_tranformation_weighted.pck',
                   features       = 'NN/mmm/12Nov19_v0/input_features.pck',
                   )

plotter.plot()
