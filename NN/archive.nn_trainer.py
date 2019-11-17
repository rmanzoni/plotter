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
       
from getpass import getuser as user

from selections import Selections

# fix random seed for reproducibility (FIXME! not really used by Keras)
np.random.seed(1986)

# luminosity
lumi = 59700.

channel = 'eee'

base_dir = None; out_dir = None
if user() =='manzoni':      
    base_dir = '/Users/manzoni/Documents/efficiencyNN/HNL/%s/ntuples/'%channel
    out_dir  = '' 
if user() =='cesareborgia': 
    base_dir = '/Users/cesareborgia/cernbox/ntuples/2018/%s/'%channel
    out_dir  = '/Users/cesareborgia/Dropbox/HNL/plotter/NN/%s/'%channel
assert base_dir; assert out_dir

# define input features
# TODO make this cfg'able
features = [
    'l0_pt'              ,
#     'l0_eta'             ,
#     'l0_phi'             ,
#     'l0_dxy'             ,
#     'l0_dz'              ,

    'l1_pt'              ,
#     'l1_eta'             ,
#     'l1_phi'             ,
#     'l1_dxy'             ,
#     'l1_dz'              ,

    'l2_pt'              ,
#     'l2_eta'             ,
#     'l2_phi'             ,
#     'l2_dxy'             ,
#     'l2_dz'              ,

    'hnl_dr_12'          ,
    'hnl_m_12'           ,

#     'sv_cos'             ,
    'sv_prob'            ,
    'hnl_2d_disp'        ,

#     'n_vtx'              ,
#     'rho'                ,
]

branches = features + [
    'run'                ,
    'lumi'               ,
    'event'              , 

    'rho'                ,
 
    'l0_pt'              ,
    'l0_eta'             ,
    'l0_phi'             ,
    'l0_dxy'             ,
    'l0_dz'              ,
    'l0_reliso_rho_03'   ,
    # 'l0_id_t'            ,
    # 'l0_id_m'            ,
    'l0_eid_mva_iso_wp90 == 1' ,

    'l1_pt'              ,
    'l1_eta'             ,
    'l1_phi'             ,
    'l1_dxy'             ,
    'l1_dz'              ,
    'l1_reliso_rho_03'   ,
    # 'l1_Medium'          ,
    'l1_MediumNoIso'          ,

    'l2_pt'              ,
    'l2_eta'             ,
    'l2_phi'             ,
    'l2_dxy'             ,
    'l2_dz'              ,
    'l2_reliso_rho_03'   ,
    # 'l2_Medium'          ,
    'l2_MediumNoIso'     ,

    'hnl_q_12'           ,
    'hnl_dr_01'          ,
    'hnl_dr_02'          ,
    'hnl_dr_12'          ,
    'hnl_dphi_01'        ,
    'hnl_dphi_02'        ,
    'hnl_m_12'           ,
    'hnl_m_01'           ,
    'hnl_m_02'           ,
    'hnl_2d_disp'        ,
    'hnl_2d_disp_sig'    ,
    'hnl_dphi_hnvis0'    ,
    'nbj'                ,
    'hnl_w_vis_m'        ,
    'hnl_q_01'           ,
    'sv_cos'             ,
    'sv_prob'            ,
    'weight'             ,
]

branches = list(set(branches))
branches_mc = [
    'l1_gen_match_isPrompt',
    'l1_gen_match_pdgid'   ,
    'l2_gen_match_isPrompt',
    'l2_gen_match_pdgid'   ,
    'lhe_weight'           ,
]

lep = None
if   channel[0] == 'm': lep = 'mu'
elif channel[0] == 'e': lep = 'ele'
assert lep == 'ele' or lep == 'mu', 'ERROR: Lepton flavor.'
filein = [
    base_dir + '/Single_{lep}_2018A/HNLTreeProducer/tree.root'.format(lep=lep),
    base_dir + '/Single_{lep}_2018B/HNLTreeProducer/tree.root'.format(lep=lep),
    base_dir + '/Single_{lep}_2018C/HNLTreeProducer/tree.root'.format(lep=lep),
    base_dir + '/Single_{lep}_2018D/HNLTreeProducer/tree.root'.format(lep=lep),
]

cuts           = Selections(channel)
baseline       = cuts.selections['baseline']
tight          = cuts.selections['tight']
is_prompt_lepton = cuts.selections['is_prompt_lepton']

# load dataset including all event, both passing and failing
passing = pd.DataFrame( root2array(filein, 'tree', branches=branches, selection= baseline + ' &  (%s)' %tight) )
failing = pd.DataFrame( root2array(filein, 'tree', branches=branches, selection= baseline + ' & !(%s)' %tight) )

# initial weights
passing['weight'] = 1.
failing['weight'] = 1.

passing['isdata'] = True
failing['isdata'] = True

passing['ismc'] = 0
failing['ismc'] = 0

# file, xsec, nevents (weighed, including negative weights! FIXME! missing)
mcs = [
#     (base_dir + '/WGamma/HNLTreeProducer/tree.root'            ,  405.271,   6108186        ),
#     (base_dir + '/DYJetsToLL_M50_ext/HNLTreeProducer/tree.root', 6077.22 , 193215674 * 0.678),
#     (base_dir + '/TTJets_ext/HNLTreeProducer/tree.root'        ,  831.76 , 142155064 * 0.373),

    (base_dir + '/DYJetsToLL_M50/HNLTreeProducer/tree.root'        ,  6077.22 , 676319.0   ),
    (base_dir + '/DYJetsToLL_M50_ext/HNLTreeProducer/tree.root'    ,  6077.22 , 130939668.0),
    (base_dir + '/DYJetsToLL_M5to50/HNLTreeProducer/tree.root'     , 81880.0  , 9947344.0  ),
    (base_dir + '/ST_sch_lep/HNLTreeProducer/tree.root'            ,     3.68 , 12447484.0 ),
    (base_dir + '/ST_tW_inc/HNLTreeProducer/tree.root'             ,    35.6  , 9553912.0  ),
    (base_dir + '/ST_tch_inc/HNLTreeProducer/tree.root'            ,    44.07 , 144094782.0),
    (base_dir + '/STbar_tW_inc/HNLTreeProducer/tree.root'          ,    35.6  , 7588180.0  ),
    (base_dir + '/STbar_tch_inc/HNLTreeProducer/tree.root'         ,    26.23 , 74227130.0 ),
    (base_dir + '/TTJets/HNLTreeProducer/tree.root'                ,   831.76 , 10234409.0 ),
    (base_dir + '/TTJets_ext/HNLTreeProducer/tree.root'            ,   831.76 , 53887126.0 ),
    (base_dir + '/WJetsToLNu/HNLTreeProducer/tree.root'            , 59850.76 , 70966439.0 ),
    (base_dir + '/WW/HNLTreeProducer/tree.root'                    ,    75.88 , 7850000.0  ),
    (base_dir + '/WZ/HNLTreeProducer/tree.root'                    ,    27.6  , 3885000.0  ),
    (base_dir + '/ZZ/HNLTreeProducer/tree.root'                    ,    12.14 , 1979000.0  ),
    (base_dir + '/WGamma/HNLTreeProducer/tree.root'                ,  405.271 ,  6108058.0 ),
    (base_dir + '/ZGamma/HNLTreeProducer/tree.root'                ,  123.9   ,  8816038.0 ),
]

passing_mcs = OrderedDict()
failing_mcs = OrderedDict()

for i, imc in enumerate(mcs):
    
    key = imc[0].split('/')[8]
    
    passing_mcs[imc[0].split('/')[8]] = pd.DataFrame( root2array(imc[0], 'tree', branches=branches+branches_mc, selection= baseline + ' &  (%s) & (%s)' %(tight, is_prompt_lepton) ) )
    failing_mcs[imc[0].split('/')[8]] = pd.DataFrame( root2array(imc[0], 'tree', branches=branches+branches_mc, selection= baseline + ' & !(%s) & (%s)' %(tight, is_prompt_lepton) ) )
    
    # weight by luminosityand notice minus sign (prompt MC subtraction)
    passing_mcs[key]['weight'] = -1. * lumi * imc[1] / imc[2] * passing_mcs[key]['lhe_weight']
    failing_mcs[key]['weight'] = -1. * lumi * imc[1] / imc[2] * failing_mcs[key]['lhe_weight']

    passing_mcs[key]['isdata'] = False
    failing_mcs[key]['isdata'] = False

    passing_mcs[key]['ismc'] = i+1
    passing_mcs[key]['ismc'] = i+1

passing = pd.concat([passing] + list(passing_mcs.values()))
failing = pd.concat([failing] + list(failing_mcs.values()))

# targets
passing['target'] = np.ones (passing.shape[0]).astype(np.int)
failing['target'] = np.zeros(failing.shape[0]).astype(np.int)

# concatenate the events and shuffle
data = pd.concat([passing, failing])
# add abs eta, dxy, dz
data['abs_l0_dxy'] = np.abs(data.l0_dxy)
data['abs_l0_dz' ] = np.abs(data.l0_dz )
data['abs_l0_eta'] = np.abs(data.l0_eta)
data['abs_l1_dxy'] = np.abs(data.l1_dxy)
data['abs_l1_dz' ] = np.abs(data.l1_dz )
data['abs_l1_eta'] = np.abs(data.l1_eta)
data['abs_l2_dxy'] = np.abs(data.l2_dxy)
data['abs_l2_dz' ] = np.abs(data.l2_dz )
data['abs_l2_eta'] = np.abs(data.l2_eta)

data['log_abs_l0_dxy'] = np.log10(np.abs(data.l0_dxy))
data['log_abs_l0_dz' ] = np.log10(np.abs(data.l0_dz ))
data['log_abs_l1_dxy'] = np.log10(np.abs(data.l1_dxy))
data['log_abs_l1_dz' ] = np.log10(np.abs(data.l1_dz ))
data['log_abs_l2_dxy'] = np.log10(np.abs(data.l2_dxy))
data['log_abs_l2_dz' ] = np.log10(np.abs(data.l2_dz ))

features += [
#     'abs_l0_dxy',
#     'abs_l0_dz' ,
    'abs_l0_eta',
#     'abs_l1_dxy',
#     'abs_l1_dz' ,
    'abs_l1_eta',
#     'abs_l2_dxy',
#     'abs_l2_dz' ,
    'abs_l2_eta',
    'log_abs_l0_dxy',
    'log_abs_l0_dz' ,
    'log_abs_l1_dxy',
    'log_abs_l1_dz' ,
    'log_abs_l2_dxy',
    'log_abs_l2_dz' ,
]

# reindex to avoid duplicated indices, useful for batches
# https://stackoverflow.com/questions/27236275/what-does-valueerror-cannot-reindex-from-a-duplicate-axis)%20-mean
data.index = np.array(range(len(data)))
data = data.sample(frac=1, replace=False, random_state=1986) # shuffle

# subtract genuine electrons using negative weights
# data['isnonprompt'] =   1. * (data['ele_genPartFlav'] == 0)  \
#                       + 1. * (data['ele_genPartFlav'] == 3)  \
#                       + 1. * (data['ele_genPartFlav'] == 4)  \
#                       + 1. * (data['ele_genPartFlav'] == 5)  \
#                       - 1. * (data['ele_genPartFlav'] == 1)  \
#                       - 1. * (data['ele_genPartFlav'] == 15) \
#                       - 1. * (data['ele_genPartFlav'] == 22)

# X and Y
X = pd.DataFrame(data, columns=list(set(branches+features+['isnonprompt', 'isdata', 'ismc'])))
Y = pd.DataFrame(data, columns=['target'])

# activation = 'tanh'
activation = 'selu'
# activation = 'sigmoid'
# activation = 'relu'
# activation = 'LeakyReLU' #??????

# define the net
input  = Input((len(features),))
layer  = Dense(1024, activation=activation   , name='dense1', kernel_constraint=unit_norm())(input)
layer  = Dropout(0.4, name='dropout1')(layer)
layer  = BatchNormalization()(layer)
# layer  = Dense(256, activation=activation   , name='dense2', kernel_constraint=unit_norm())(layer)
# layer  = Dropout(0.4, name='dropout2')(layer)
# layer  = BatchNormalization()(layer)
# layer  = Dense(16, activation=activation   , name='dense3', kernel_constraint=unit_norm())(layer)
# layer  = Dropout(0.4, name='dropout3')(layer)
# layer  = BatchNormalization()(layer)
# layer  = Dense(16, activation=activation   , name='dense4', kernel_constraint=unit_norm())(layer)
# layer  = Dropout(0.4, name='dropout4')(layer)
# layer  = BatchNormalization()(layer)
layer  = Dense(32, activation=activation   , name='dense5', kernel_constraint=unit_norm())(layer)
layer  = Dropout(0.2, name='dropout5')(layer)
layer  = BatchNormalization()(layer)
output = Dense(  1, activation='sigmoid', name='output', )(layer)

# Define outputs of your model
model = Model(input, output)

# choose your optimizer
# opt = SGD(lr=0.0001, momentum=0.8)
# opt = Adam(lr=0.001, decay=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)
opt = Adam(lr=0.02, decay=0.05, beta_1=0.9, beta_2=0.999, amsgrad=True)
# opt = 'Adam'

# compile and choose your loss function (binary cross entropy for a 1-0 classification problem)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['mae', 'acc'])

# print net summary
print(model.summary())

# plot the models
# https://keras.io/visualization/
plot_model(model, show_shapes=True, show_layer_names=True, to_file=out_dir+'model.png')

# normalize inputs FIXME! do it, but do it wisely
# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
# on QuantileTransformer
# Note that this transform is non-linear. It may distort linear
#     correlations between variables measured at the same scale but renders
#     variables measured at different scales more directly comparable.
# from sklearn.preprocessing import QuantileTransformer
# qt = QuantileTransformer(output_distribution='normal', random_state=1986)
# fit and FREEZE the transformation paramaters. 
# Need to save these to be applied exactly as are when predicting on a different dataset
# qt.fit(X[features])
# now transform
# xx = qt.transform(X[features])

# alternative way to scale the inputs
# https://datascienceplus.com/keras-regression-based-neural-networks/

from sklearn.preprocessing import RobustScaler
qt = RobustScaler()
qt.fit(X[features])
xx = qt.transform(X[features])

# save the frozen transformer
pickle.dump( qt, open( out_dir + 'input_tranformation_weighted.pck', 'wb' ) )

# save the exact list of features
pickle.dump( features, open( out_dir + 'input_features.pck', 'wb' ) )

# early stopping
# monitor = 'val_acc'
monitor = 'val_loss'
# monitor = 'val_mae'
es = EarlyStopping(monitor=monitor, mode='auto', verbose=1, patience=50, restore_best_weights=True)

# reduce learning rate when at plateau, fine search the minimum
reduce_lr = ReduceLROnPlateau(monitor=monitor, mode='auto', factor=0.2, patience=5, min_lr=0.00001, cooldown=10, verbose=True)

# save the model every now and then
filepath = out_dir + 'saved-model-{epoch:04d}_val_loss_{val_loss:.4f}_val_acc_{val_acc:.4f}.h5'
save_model = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# weight the events according to their displacement (favour high displacement)
weight = np.array(data.weight * np.power(X['hnl_2d_disp'], 0.25))

# train only the classifier. beta is set at 0 and the discriminator is not trained
# history = model.fit(xx, Y, epochs=2000, validation_split=0.5, callbacks=[es, reduce_lr], batch_size=32, sample_weight=np.array(X['isnonprompt']))  
# history = model.fit(xx, Y, epochs=100, validation_split=0.5, callbacks=[es, reduce_lr], batch_size=32, verbose=True)  
history = model.fit(xx, Y, epochs=1000, validation_split=0.5, callbacks=[es, reduce_lr, save_model], batch_size=32, verbose=True, sample_weight=weight)  

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
plt.savefig(out_dir + 'loss_function_history_weighted.pdf')
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
plt.savefig(out_dir + 'accuracy_history_weighted.pdf')
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
plt.savefig(out_dir + 'mean_absolute_error_history_weighted.pdf')
plt.clf()

# calculate predictions on the data sample
print('predicting on', data.shape[0], 'events')
x = pd.DataFrame(data, columns=features)
# y = model.predict(x)
# load the transformation with the correct parameters!
qt = pickle.load(open( out_dir + 'input_tranformation_weighted.pck', 'rb' ))
xx = qt.transform(x[features])
y = model.predict(xx)

# impose norm conservation if you want probabilities
# compute the overall rescaling factor scale
scale = 1.
# scale = np.sum(passing['target']) / np.sum(y)

# add the score to the data sample
data.insert(len(data.columns), 'fr', scale * y)

# let sklearn do the heavy lifting and compute the ROC curves for you
fpr, tpr, wps = roc_curve(data.target, data.fr) 
plt.plot(fpr, tpr)
xy = [i*j for i,j in product([10.**i for i in range(-8, 0)], [1,2,4,8])]+[1]
plt.plot(xy, xy, color='grey', linestyle='--')
plt.yscale('linear')
plt.savefig(out_dir + 'roc_weighted.pdf')

# save model and weights
model.save(out_dir + 'net_model_weighted.h5')
# model.save_weights('net_model_weights.h5')

# rename branches, if you want
# data.rename(
#     index=str, 
#     columns={'cand_refit_mass12': 'mass12',}, 
#     inplace=True)

# save ntuple
data.to_root(out_dir + 'output_ntuple_weighted.root', key='tree', store_index=False)

