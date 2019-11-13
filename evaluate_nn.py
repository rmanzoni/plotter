
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
        self.transformation = pickle.load(open(transformation, 'rb'))

        # define input features
        self.features = pickle.load(open(features, 'rb'))

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
            print('empty DataFrame, returning None')
            return None
        # calculate predictions on the data sample
        print('predicting on', df.shape[0], 'events')
        # enrich the df with the needed features
        df = self._prepare_df(df)
        # x = pd.DataFrame(data, columns=features)
        x = df[self.features]
        # load the transformation with the correct parameters!
        xx = self.transformation.transform(x[self.features])
        y = self.model.predict(xx)
        return y
