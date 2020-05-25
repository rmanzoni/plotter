'''
Resources:

Build TensorFlow with native CPU instructions (make it faster)
https://gist.github.com/winnerineast/05f63146e4b1e81ae08d14da2b38b11f

https://en.wikipedia.org/wiki/Universal_approximation_theorem
http://neuralnetworksanddeeplearning.com/chap4.html
https://github.com/thomberg1/UniversalFunctionApproximation
https://cms-nanoaod-integration.web.cern.ch/integration/master-102X/mc102X_doc.html
https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
https://stats.stackexchange.com/questions/292278/can-one-theoretically-train-a-neural-network-with-fewer-training-samples-than
'''
import root_pandas

from time import time 
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from collections import OrderedDict

from root_numpy import root2array

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras.activations import softmax
from keras.constraints import unit_norm
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from plotter.sample import get_data_samples, get_mc_samples, get_signal_samples
from plotter.utils import nn_dir

# fix random seed for reproducibility (FIXME! not really used by Keras)
np.random.seed(1986)

class Trainer(object):
    def __init__(
        self               ,
        channel            ,
        features           , 
        composed_features  , 
        base_dir           ,
        post_fix           ,
        selection_data_mmm ,
        selection_mc_mmm   ,
        selection_data_mem ,
        selection_mc_mem   ,
        selection_data_eee ,
        selection_mc_eee   ,
        selection_data_eem ,
        selection_mc_eem   ,
        selection_tight    ,
        lumi               ,
        epochs=1000        ,
        early_stopping=True,
        skip_mc=False,
        val_fraction=0.3,
        scale_mc=1.):

        self.channel            = channel.split('_')[0]
        self.channel_extra      = channel.split('_')[1] if len(channel.split('_'))>1 else ''
        self.features           = features 
        self.composed_features  = composed_features 
        self.base_dir           = base_dir 
        self.post_fix           = post_fix 
        self.selection_data_mmm = ' & '.join(selection_data_mmm)
        self.selection_mc_mmm   = ' & '.join(selection_mc_mmm)
        self.selection_data_mem = ' & '.join(selection_data_mem)
        self.selection_mc_mem   = ' & '.join(selection_mc_mem)  
        self.selection_data_eee = ' & '.join(selection_data_eee)
        self.selection_mc_eee   = ' & '.join(selection_mc_eee)  
        self.selection_data_eem = ' & '.join(selection_data_eem)
        self.selection_mc_eem   = ' & '.join(selection_mc_eem)  
        self.selection_tight   = selection_tight
        self.selection_lnt     = 'not (%s)' %self.selection_tight
        self.lumi              = lumi
        self.epochs            = epochs
        self.early_stopping    = early_stopping
        self.skip_mc           = skip_mc
        self.val_fraction      = val_fraction
        self.scale_mc          = scale_mc

    def train(self):

        net_dir_name = self.channel+'_'+self.channel_extra if len(self.channel_extra) else self.channel
        net_dir = nn_dir(net_dir_name)
        print('============> starting reading the trees')
        print ('Net will be stored in: ', net_dir)
        now = time()
        # FIXME! temporary hack
        data  = get_data_samples('mmm', self.base_dir, self.post_fix.replace('CHANNEL', 'mmm'), self.selection_data_mmm)
        data += get_data_samples('mem', self.base_dir, self.post_fix.replace('CHANNEL', 'mem'), self.selection_data_mem)
        data += get_data_samples('eee', self.base_dir, self.post_fix.replace('CHANNEL', 'eee'), self.selection_data_eee)
        data += get_data_samples('eem', self.base_dir, self.post_fix.replace('CHANNEL', 'eem'), self.selection_data_eem)
        # FIXME! temporary hack
        if self.skip_mc:
            mc = []
        else:
            mc  = get_mc_samples('mmm', self.base_dir, self.post_fix.replace('CHANNEL', 'mmm'), self.selection_mc_mmm)
            mc += get_mc_samples('mem', self.base_dir, self.post_fix.replace('CHANNEL', 'mem'), self.selection_mc_mem)
            mc += get_mc_samples('eee', self.base_dir, self.post_fix.replace('CHANNEL', 'eee'), self.selection_mc_eee)
            mc += get_mc_samples('eem', self.base_dir, self.post_fix.replace('CHANNEL', 'eem'), self.selection_mc_eem)

        print('============> it took %.2f seconds' %(time() - now))

        data_df = pd.concat([idt.df for idt in data], sort=False)
        # mc_df   = pd.concat([imc.df for imc in mc], sort=False)

        # initial weights
        data_df['weight'] = 1.
        data_df['isdata'] = 1
        data_df['ismc'] = 0

        passing_data = data_df.query(self.selection_tight)
        failing_data = data_df.query(self.selection_lnt)

        if self.skip_mc:
            passing = passing_data
            failing = failing_data
        else:
            for i, imc in enumerate(mc):
            
                imc.df['weight'] = -1. * self.lumi * imc.lumi_scaling * imc.df.lhe_weight * self.scale_mc
                imc.df['isdata'] = 0
                imc.df['ismc']   = i+1

                imc.df_tight = imc.df.query(self.selection_tight)
                imc.df_lnt   = imc.df.query(self.selection_lnt)

            passing_mc = pd.concat([imc.df_tight for imc in mc], sort=False)
            failing_mc = pd.concat([imc.df_lnt   for imc in mc], sort=False)

            passing = pd.concat ([passing_data, passing_mc], sort=False)
            failing = pd.concat ([failing_data, failing_mc], sort=False)

        # targets
        passing['target'] = np.ones (passing.shape[0]).astype(np.int)
        failing['target'] = np.zeros(failing.shape[0]).astype(np.int)

        # concatenate the events and shuffle
        main_df = pd.concat([passing, failing], sort=False)
        
        for k, v in self.composed_features.items():
            main_df[k] = v(main_df)
            self.features.append(k)
            
        # reindex to avoid duplicated indices, useful for batches
        # https://stackoverflow.com/questions/27236275/what-does-valueerror-cannot-reindex-from-a-duplicate-axis)%20-mean
        main_df.index = np.array(range(len(main_df)))
        main_df = main_df.sample(frac=1, replace=False, random_state=1986) # shuffle

        # X and Y
        X = pd.DataFrame(main_df, columns=list(set(self.features)))
        # X = pd.DataFrame(main_df, columns=list(set(branches+features+['isnonprompt', 'ismain_df', 'ismc'])))
        Y = pd.DataFrame(main_df, columns=['target'])

        # activation = 'tanh'
        activation = 'selu'
        # activation = 'sigmoid'
        # activation = 'relu'
        # activation = 'LeakyReLU' #??????

        # define the net
        input  = Input((len(self.features),))
        layer  = Dense(128, activation=activation   , name='dense1', kernel_constraint=unit_norm())(input)
        layer  = Dropout(0.5, name='dropout1')(layer)
        layer  = BatchNormalization()(layer)
        layer  = Dense(64, activation=activation   , name='dense2', kernel_constraint=unit_norm())(layer)
        layer  = Dropout(0.4, name='dropout2')(layer)
        # layer  = BatchNormalization()(layer)
        # layer  = Dense(16, activation=activation   , name='dense3', kernel_constraint=unit_norm())(layer)
        # layer  = Dropout(0.4, name='dropout3')(layer)
        # layer  = BatchNormalization()(layer)
        # layer  = Dense(16, activation=activation   , name='dense4', kernel_constraint=unit_norm())(layer)
        # layer  = Dropout(0.4, name='dropout4')(layer)
        # layer  = BatchNormalization()(layer)
        layer  = Dense(16, activation=activation   , name='dense5', kernel_constraint=unit_norm())(layer)
        layer  = Dropout(0.4, name='dropout5')(layer)
        layer  = BatchNormalization()(layer)
        output = Dense(  1, activation='sigmoid', name='output', )(layer)

        # Define outputs of your model
        model = Model(input, output)

        # choose your optimizer
        # opt = SGD(lr=0.0001, momentum=0.8)
        # opt = Adam(lr=0.001, decay=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)
        opt = Adam(lr=0.01, decay=0.05, beta_1=0.9, beta_2=0.999, amsgrad=True)
            # opt = 'Adam'

        # compile and choose your loss function (binary cross entropy for a 1-0 classification problem)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['mae', 'acc'])

        # print net summary
        print(model.summary())

        # plot the models
        # https://keras.io/visualization/
        plot_model(model, show_shapes=True, show_layer_names=True, to_file=net_dir+'model.png')

        # normalize inputs FIXME! do it, but do it wisely
        # https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
        # on QuantileTransformer
        # Note that this transform is non-linear. It may distort linear
        #     correlations between variables measured at the same scale but renders
        #     variables measured at different scales more directly comparable.
        # from sklearn.preprocessing import QuantileTransformer
        # qt = QuantileTransformer(output_distribution='normal', random_state=1986)
        # fit and FREEZE the transformation paramaters. 
        # Need to save these to be applied exactly as are when predicting on a different main_dfset
        # qt.fit(X[features])
        # now transform
        # xx = qt.transform(X[features])

        # alternative way to scale the inputs
        # https://main_dfscienceplus.com/keras-regression-based-neural-networks/

        from sklearn.preprocessing import RobustScaler
        qt = RobustScaler()
        qt.fit(X[self.features])
        xx = qt.transform(X[self.features])

        # save the frozen transformer
        pickle.dump( qt, open( net_dir + 'input_tranformation_weighted.pck', 'wb' ) )

        # save the exact list of features
        pickle.dump( self.features, open( net_dir + 'input_features.pck', 'wb' ) )

        # early stopping
        # monitor = 'val_acc'
        monitor = 'val_loss'
        # monitor = 'val_mae'
        es = EarlyStopping(monitor=monitor, mode='auto', verbose=1, patience=50, restore_best_weights=True)

        # reduce learning rate when at plateau, fine search the minimum
        reduce_lr = ReduceLROnPlateau(monitor=monitor, mode='auto', factor=0.2, patience=5, min_lr=0.00001, cooldown=10, verbose=True)

        # save the model every now and then
        filepath = net_dir + 'saved-model-{epoch:04d}_val_loss_{val_loss:.4f}_val_acc_{val_acc:.4f}.h5'
        save_model = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        # weight the events according to their displacement (favour high displacement)
#         weight = np.array(main_df.weight * np.power(X['hnl_2d_disp'], 0.25))
        weight = np.array(main_df.weight)

        # train only the classifier. beta is set at 0 and the discriminator is not trained
        callbacks = [reduce_lr, save_model]
        if self.early_stopping:
            callbacks.append(es)
        history = model.fit(xx, Y, epochs=self.epochs, validation_split=self.val_fraction, callbacks=callbacks, batch_size=32, verbose=True, sample_weight=weight)  

        # plot loss function trends for train and validation sample
        plt.clf()
        plt.title('loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        # plt.yscale('log')
        center = min(history.history['val_loss'] + history.history['loss'])
        plt.ylim((center*0.98, center*1.5))
        plt.grid(True)
        plt.savefig(net_dir + 'loss_function_history_weighted.pdf')
        plt.clf()

        # plot accuracy trends for train and validation sample
        plt.title('accuracy')
        plt.plot(history.history['acc'], label='train')
        plt.plot(history.history['val_acc'], label='test')
        plt.legend()
        center = max(history.history['val_acc'] +  history.history['acc'])
        plt.ylim((center*0.85, center*1.02))
        # plt.yscale('log')
        plt.grid(True)
        plt.savefig(net_dir + 'accuracy_history_weighted.pdf')
        plt.clf()

        # plot accuracy trends for train and validation sample
        plt.title('mean absolute error')
        plt.plot(history.history['mae'], label='train')
        plt.plot(history.history['val_mae'], label='test')
        plt.legend()
        center = min(history.history['val_mae'] + history.history['mae'])
        plt.ylim((center*0.98, center*1.5))
        # plt.yscale('log')
        plt.grid(True)
        plt.savefig(net_dir + 'mean_absolute_error_history_weighted.pdf')
        plt.clf()

        # calculate predictions on the main_df sample
        print('predicting on', main_df.shape[0], 'events')
        x = pd.DataFrame(main_df, columns=self.features)
        # y = model.predict(x)
        # load the transformation with the correct parameters!
        qt = pickle.load(open( net_dir + 'input_tranformation_weighted.pck', 'rb' ))
        xx = qt.transform(x[self.features])
        y = model.predict(xx)

        # impose norm conservation if you want probabilities
        # compute the overall rescaling factor scale
        scale = 1.
        # scale = np.sum(passing['target']) / np.sum(y)

            # add the score to the main_df sample
        main_df.insert(len(main_df.columns), 'fr', scale * y)

        # let sklearn do the heavy lifting and compute the ROC curves for you
        fpr, tpr, wps = roc_curve(main_df.target, main_df.fr) 
        plt.plot(fpr, tpr)
        xy = [i*j for i,j in product([10.**i for i in range(-8, 0)], [1,2,4,8])]+[1]
        plt.plot(xy, xy, color='grey', linestyle='--')
        plt.yscale('linear')
        plt.savefig(net_dir + 'roc_weighted.pdf')

        # save model and weights
        model.save(net_dir + 'net_model_weighted.h5')
        # model.save_weights('net_model_weights.h5')

        # rename branches, if you want
        # main_df.rename(
        #     index=str, 
        #     columns={'cand_refit_mass12': 'mass12',}, 
        #     inplace=True)

        # save ntuple
        main_df.to_root(net_dir + 'output_ntuple_weighted.root', key='tree', store_index=False)

