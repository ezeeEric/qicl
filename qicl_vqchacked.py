#!/usr/bin/env python
# coding: utf-8

# ## VQC implementation of a HEP classification problem.
# 
# @author: Eric Drechsler (dr.eric.drechsler@gmail.com)

# In[7]:


import pandas
import pickle
import time
time0=time.time()

from plotter import plotTruth,plotVars
#from datasets import *
from qiskit import BasicAer
from qiskit.aqua import run_algorithm, QuantumInstance
# from qiskit.aqua.algorithms import VQC
from vqc import VQC
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.input import ClassificationInput
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
import numpy as np


# In[8]:


from argparse import ArgumentParser

argParser = ArgumentParser(add_help=False)
argParser.add_argument( '-t', '--steerTestRun', action="store_true")
argParser.add_argument( '-od', '--steerOutDir', help='Output Directory', default=".")
argParser.add_argument( '-nevt', '--numberEvents', help='Number of events', default=50)
argParser.add_argument( '-sh',   '--numberShots', help='Number of shots', default=100)
#this is mainly driving runtime
argParser.add_argument( '-mt',   '--maxTrials', help='Max trials SPSA', default=1)
argParser.add_argument( '-ss',   '--saveSteps', help='SPSA save steps', default=3)
argParser.add_argument( '-vfd',  '--varFormDepth', help='variational form depth', default=2)
argParser.add_argument( '-fmd',  '--featMapDepth', help='Feature Map depth', default=3)
argParser.add_argument( '-nqb',  '--numberQbits', help='Number of qbits', default=3)
#default
#argParser.add_argument( '-spsa',  '--spsaoptim', help='Use SPSA optimiser', action='store_true')
# argParser.add_argument( '-cob',  '--steerCobylaOptim', help='Use COBYLA optimiser', action='store_true')
argParser.add_argument( '-mop',  '--steerManualOptim', help='Use manual optimisation strategy', action='store_true')
args = argParser.parse_args()


# ## Prepare dataset

# In[3]:


in_df = pandas.read_pickle("MixData_PD.pkl")

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
# plotVars(x_train, in_df, [var1,var2,var3], label_names)
# plotTruth(x_train,in_df,label_names)

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


def getDefaultVQC():
    optimizer = SPSA(
                     max_trials=int(args.maxTrials),
                     c0=4.0,
                     save_steps=int(args.saveSteps),
                     skip_calibration=True
    )
    feature_map = SecondOrderExpansion(
            feature_dimension=int(args.featMapDepth), 
            depth=int(args.varFormDepth),
            entanglement = 'full'
    )


    var_form = RYRZ(
            num_qubits=int(args.numberQbits),
            depth=int(args.varFormDepth),
            entanglement='full',
            entanglement_gate='cx',
    )


    vqc = VQC(optimizer, feature_map, var_form, trainDict, testDict, manualOptimisation=args.steerManualOptim)

    return vqc

def defaultVQC(vqc):
    backend = BasicAer.get_backend('qasm_simulator')

    quantum_instance = QuantumInstance(
            backend = backend,
            shots=int(args.numberShots),
            seed_simulator=420,
            seed_transpiler=420,
    )

    result=None
    print("Running Algorithm")
    if not args.steerTestRun:
        result = vqc.run(quantum_instance)
        print("testing success ratio: ", result['testing_accuracy'])
    return result

def predictVQC(vqc,testData):
    predicted_probs, predicted_labels = vqc.predict(testData)
    predicted_classes = map_label_to_class_name(predicted_labels, vqc.label_to_class)
    print("prediction:   {}".format(predicted_labels))
    return predicted_classes

defVQC=getDefaultVQC()
result=defaultVQC(defVQC)

prediction=predictVQC(defVQC, datapoints[0])
