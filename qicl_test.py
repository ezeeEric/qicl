#!/usr/bin/env python
# coding: utf-8

# This is a test file for the VQC implementation of a HEP classification problem.
# 
# @author: Eric Drechsler (dr.eric.drechsler@gmail.com)

# Copied from qiskit-community-tutorials

# Copied from qiskit-community-tutorials

# In[1]:


from datasets import *
from qiskit import BasicAer
from qiskit.aqua import run_algorithm, QuantumInstance
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua.input import ClassificationInput
import pandas


# In[2]:


in_df = pandas.read_pickle("../MixData_PD.pkl")

nevt = 1000
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


# In[3]:


params = {
    'problem': {'name': 'classification', 'random_seed': 420 },
    'algorithm': {'name': 'VQC', 'override_SPSA_params': True},
    'backend': {'shots': 100},
    'optimizer': {'name': 'SPSA', 'max_trials': 20, 'save_steps': 5},
    'variational_form': {'name': 'RYRZ', 'depth': 2},
    'feature_map': {'name': 'SecondOrderExpansion', 'depth': 1}
}

classification_input = ClassificationInput(trainDict, testDict, x_test)
backend = BasicAer.get_backend('qasm_simulator')

#result = run_algorithm(params, classification_input, backend=backend)
#print("testing success ratio: ", result['testing_accuracy'])
#print("predicted classes:", result['predicted_classes'])


# In[ ]:




