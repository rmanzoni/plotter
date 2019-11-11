
##########################################################################################
# READ NN WEIGHTS
##########################################################################################
import numpy as np
import pickle
from keras.models import load_model

class Evaluator(object):
    def __init__(self, model, transformation, features):
        # load the model
        self.model = load_model(model)
        # load the input feature transformer
        self.transformation = pickle.load(open(transformation, 'r'))

        # define input features
        self.features = pickle.load(open(features, 'r'))

#         
#         self.features = [
#             'l0_pt'              ,
#         #     'l0_eta'             ,
#             'l0_phi'             ,
#         #     'l0_dxy'             ,
#         #     'l0_dz'              ,
# 
#             'l1_pt'              ,
#         #     'l1_eta'             ,
#             'l1_phi'             ,
#         #     'l1_dxy'             ,
#         #     'l1_dz'              ,
# 
#             'l2_pt'              ,
#         #     'l2_eta'             ,
#             'l2_phi'             ,
#         #     'l2_dxy'             ,
#         #     'l2_dz'              ,
# 
#             'hnl_dr_12'          ,
#             'hnl_m_12'           ,
# 
#             'sv_cos'             ,
#             'sv_prob'            ,
#             'hnl_2d_disp'        ,
# 
#         #     'n_vtx'              ,
#             'rho'                ,
#         
#         ############################# 
#         #     'abs_l0_dxy',
#         #     'abs_l0_dz' ,
#             'abs_l0_eta',
#         #     'abs_l1_dxy',
#         #     'abs_l1_dz' ,
#             'abs_l1_eta',
#         #     'abs_l2_dxy',
#         #     'abs_l2_dz' ,
#             'abs_l2_eta',
#             'log_abs_l0_dxy',
#             'log_abs_l0_dz' ,
#             'log_abs_l1_dxy',
#             'log_abs_l1_dz' ,
#             'log_abs_l2_dxy',
#             'log_abs_l2_dz' ,
#         ]
# 
#         self.features = [
#             'l0_pt', 
#             'l1_pt', 
#             'l2_pt', 
#             'hnl_2d_disp', 
#             'hnl_dr_12', 
#             'hnl_m_12', 
#             'sv_prob',
# 
#             'abs_l0_eta',
#             'abs_l1_eta',
#             'abs_l2_eta',
#             
#             'log_abs_l0_dxy',
#             'log_abs_l0_dz' ,
#             'log_abs_l1_dxy',
#             'log_abs_l1_dz' ,
#             'log_abs_l2_dxy',
#             'log_abs_l2_dz' ,
#         ]
# 

    def _prepare_df(self, df):
        df['abs_l0_dxy'] = np.abs(df.l0_dxy)
        df['abs_l0_dz' ] = np.abs(df.l0_dz )
        df['abs_l0_eta'] = np.abs(df.l0_eta)
        df['abs_l1_dxy'] = np.abs(df.l1_dxy)
        df['abs_l1_dz' ] = np.abs(df.l1_dz )
        df['abs_l1_eta'] = np.abs(df.l1_eta)
        df['abs_l2_dxy'] = np.abs(df.l2_dxy)
        df['abs_l2_dz' ] = np.abs(df.l2_dz )
        df['abs_l2_eta'] = np.abs(df.l2_eta)

        df['log_abs_l0_dxy'] = np.log10(np.abs(df.l0_dxy))
        df['log_abs_l0_dz' ] = np.log10(np.abs(df.l0_dz ))
        df['log_abs_l1_dxy'] = np.log10(np.abs(df.l1_dxy))
        df['log_abs_l1_dz' ] = np.log10(np.abs(df.l1_dz ))
        df['log_abs_l2_dxy'] = np.log10(np.abs(df.l2_dxy))
        df['log_abs_l2_dz' ] = np.log10(np.abs(df.l2_dz ))

        return df

    def evaluate(self, df):
        if not len(df):
            print 'empty DartaFrame, returning None'
            return None
        # calculate predictions on the data sample
        print 'predicting on', df.shape[0], 'events'
        # enrich the df with the needed features
        df = self._prepare_df(df)
        # x = pd.DataFrame(data, columns=features)
        x = df[self.features]
        # load the transformation with the correct parameters!
        xx = self.transformation.transform(x[self.features])
        y = self.model.predict(xx)
        return y
