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

    def evaluate(self, df):
        if not len(df):
            print('empty DataFrame, returning None')
            return None
        # calculate predictions on the data sample
        print('predicting on', df.shape[0], 'events')
        # x = pd.DataFrame(data, columns=features)
        x = df[self.features]
        # load the transformation with the correct parameters!
        xx = self.transformation.transform(x[self.features])
        y = self.model.predict(xx)
        return y
