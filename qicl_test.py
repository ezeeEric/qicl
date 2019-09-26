#!/usr/bin/env python
# codiVng: utf-8

# This is a test file for the VQC implementation of a HEP classification problem.
# @author: Eric Drechsler (dr.eric.drechsler@gmail.com)

# Copied from qiskit-community-tutorials

import pandas
import pickle
import time
time0=time.time()

from datasets import *
from qiskit import BasicAer
from qiskit.aqua import run_algorithm, QuantumInstance
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua.input import ClassificationInput

from argparse import ArgumentParser

argParser = ArgumentParser(add_help=False)
steerArgs = argParser.add_argument_group('parameters')
paraArgs = argParser.add_argument_group('parameters')

argParser.add_argument( '-nevt', '--numberEvents', help='Number of events', default=10)
argParser.add_argument( '-sh', '--numberShots', help='Number of shots', default=1024)
argParser.add_argument( '-mt', '--maxTrials', help='Max trials SPSA', default=20)
argParser.add_argument( '-ss', '--saveSteps', help='SPSA save steps', default=5)
argParser.add_argument( '-vfd', '--varFormDepth', help='variational form depth', default=2)
argParser.add_argument( '-fmd', '--featMapDepth', help='Feature Map depth', default=1)
args = argParser.parse_args()

in_df = pandas.read_pickle("../MixData_PD.pkl")

nevt = args.numberEvents
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

params = {
    'problem': {'name': 'classification', 'random_seed': 420 },
    'algorithm': {'name': 'VQC', 'override_SPSA_params': True},
    'backend': {'shots': args.numberShots},
    'optimizer': {'name': 'SPSA', 'max_trials': args.maxTrials, 'save_steps': args.saveSteps},
    'variational_form': {'name': 'RYRZ', 'depth': args.varFormDepth},
    'feature_map': {'name': 'SecondOrderExpansion', 'depth': args.featMapDepth}
}

classification_input = ClassificationInput(trainDict, testDict, x_test)
backend = BasicAer.get_backend('qasm_simulator')

result={'testing_accuracy':None, 'predicted_classes':None}
#result = run_algorithm(params, classification_input, backend=backend)
print("testing success ratio: ", result['testing_accuracy'])
print("predicted classes:", result['predicted_classes'])

#time or tag setting in name
outtag="_".join([str(i) for i in vars(args).values()])
outtag+="_%s"%(int(time.time()))
pklFile=open("./qicl_test_%s.pkl"%outtag,'wb')
pickle.dump( result , pklFile)

m, s = divmod(time.time()-time0, 60)
h, m = divmod(m, 60)
print("Execution time: %d:%02d:%02d" % (h,m,s))
print("Success!")
