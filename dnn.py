# Binary Classification with Sonar Dataset: Standardized Smaller
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

var1 = 'lep1_pt'
var2 = 'lep2_pt'
var3 = 'reco_zv_mass'
nevt=100000

# load dataset
in_df = pd.read_pickle("./input/MixData_PD.pkl")

# split into input (X) and output (Y) variables
X = in_df.loc[:nevt-1,[var1, var2, var3]].values
Y = in_df['isSignal'].values[:nevt] 

# smaller model
def create_model():
  # create model
  model = Sequential()
  model.add(Dense(32, input_dim=3, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(32, input_dim=3, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=10, batch_size=2048, verbose=0)))

pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, Y, cv=kfold)

print("Training on {0} events".format(nevt))
print("Efficiency: %.2f (%.2f)" % (results.mean(), results.std()))
