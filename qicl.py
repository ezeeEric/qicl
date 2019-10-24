#!/usr/bin/env python
# coding: utf-8

# ## VQC implementation of a HEP classification problem.
# 
# @author: Eric Drechsler (dr.eric.drechsler@gmail.com)

import pandas
import pickle
import time

time0=time.time()
#from datasets import *
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.input import ClassificationInput
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
import numpy as np

from utils.plotter import plotTruth,plotVars,plotVarsInd

from vqcore.vqc import VQC
from vqcore.vqc_utils import getSimulationInstance,getIBMQInstance,trainVQC,predictVQC
from vqcore.featureMaps import getFeatureMap
from vqcore.optimisers import getOptimiser

from utils.tools import timelogDict,chronomat
from argparse import ArgumentParser

argParser = ArgumentParser(add_help=False)
argParser.add_argument( '-t', '--steerTestRun', action="store_true")
argParser.add_argument( '-od', '--steerOutDir', help='Output Directory', default=".")
argParser.add_argument( '-nevt', '--numberEvents', help='Number of events', default=100)
argParser.add_argument( '-mbs', '--minibatchsize', help='Number of events', default=-1)
argParser.add_argument( '-sh',   '--numberShots', help='Number of shots', default=1024)
#this is mainly driving runtime
argParser.add_argument( '-mi',   '--maxiter', help='Max trials', default=100)
argParser.add_argument( '-ss',   '--saveSteps', help='SPSA save steps', default=5)
argParser.add_argument( '-vfd',  '--varFormDepth', help='variational form depth', default=2)
argParser.add_argument( '-fd',  '--feature_dimension', help='Feature Map dimension', default=3)
argParser.add_argument( '-fmd',  '--featureMapDepth', help='Feature Map depth', default=1)
argParser.add_argument( '-nqb',  '--numberQbits', help='Number of qbits', default=3)
argParser.add_argument( '-opt',  '--optimiser', help='Choose optimiser [SPSA,COBYLA,L_BFGS_B,NELDER_MEAD,P_BFGS,SLSQP]', type=str, default="COBYLA")
argParser.add_argument( '-qu',  '--quantum', help='Run on IBMQ', action="store_true")
args = argParser.parse_args()

in_df = pandas.read_pickle("./input/MixData_PD.pkl")

nevt = int(args.numberEvents)
var1 = 'lep1_pt'
var2 = 'lep2_pt'
var3 = 'reco_zv_mass'
nvars = 3

y_train_label=in_df['isSignal'].values[:nevt] #[0 1 0 ...]
y_test_label=in_df['isSignal'].values[nevt:2*nevt] #same
x_train=in_df.loc[:nevt-1,[var1, var2, var3]].values #[[var1 var2 var3], [var1 var2 var3],...]
x_test=in_df.loc[nevt:2*nevt-1,[var1, var2, var3]].values
y_train=np.eye(2)[y_train_label]
y_test=np.eye(2)[y_test_label]

trainDict={"signal": [], "background": []}
testDict ={"signal": [], "background": []}

label_names = ['background','signal']
plotVarsInd(x_train, in_df, [var1,var2,var3], label_names)
plotTruth(x_train,in_df,label_names)
import sys
sys.exit()

#TODO better way of dealing with this?
for i in range(0,nevt):
    if (y_train_label[i]==1):
        trainDict["signal"].append(x_train[i].tolist())
    else:
         trainDict["background"].append(x_train[i].tolist())
trainDict={"signal": np.array(trainDict["signal"]), "background":  np.array(trainDict["background"])}
            
for i in range(0,nevt):
    if (y_test_label[i]==1):
        testDict["signal"].append(x_test[i].tolist())
    else:
         testDict["background"].append(x_test[i].tolist())
testDict={"signal": np.array(testDict["signal"]), "background":  np.array(testDict["background"])}

datapoints, class_to_label = split_dataset_to_data_and_labels(testDict)
print("Test and Training data ready")

@chronomat
def getDefaultVQC(optimiser,feature_map):
    var_form = RYRZ(
            num_qubits=int(args.numberQbits),
            depth=int(args.varFormDepth),
            entanglement='full',
            entanglement_gate='cx',
    )
    print(trainDict,testDict)
    vqc = VQC(optimiser, feature_map, var_form, trainDict, testDict, max_evals_grouped=1,minibatch_size=int(args.minibatchsize))
    return vqc


parameters={
"max_trials":int(args.maxiter),
"save_steps":int(args.saveSteps),
"maxiter":int(args.maxiter),
"maxfun":int(args.maxiter),#BFGS
}
optimiser=getOptimiser(name=str(args.optimiser),params=parameters)

parameters={
"feature_dimension":int(args.feature_dimension),
"depth":						int(args.featureMapDepth),
"entanglement":		'full',
}
featureMap=getFeatureMap(name="SecondOrderExpansion",params=parameters)

quInstance=None
if args.quantum:
	quInstance=getIBMQInstance(int(args.numberShots))
else:
	quInstance=getSimulationInstance(int(args.numberShots))

defVQC=getDefaultVQC(optimiser,featureMap)
result=trainVQC(defVQC,quInstance)

#prediction=predictVQC(defVQC, datapoints[0])

#time or tag setting in name
outtag="_".join([str(vars(args)[i]) if not "steer" in str(i) else "" for i in vars(args)])
outtag+="_%s"%(int(time.time()))
pklFile=open("{0}/output/qicl{1}.pkl".format(args.steerOutDir,outtag),'wb')
pickle.dump( result , pklFile)
pickle.dump( vars(args) , pklFile)

#kept for compatibility with batch scripts $
duration=time.time()-time0
m, s = divmod(duration, 60)
h, m = divmod(m, 60)
print("Execution time: %d " % (duration))
print("Execution time: %d:%02d:%02d" % (h,m,s))
for key,val in timelogDict.items():
  print(key,val)
