""" Definition of analysis independent, package wide tools.

@author Eric "Dr. Dre" Drechsler (eric.drechsler@cern.ch)
"""

import sys
from functools import wraps
import time
import numpy as np

#create global dictionary to store execution times of various methods
from collections import OrderedDict
timelogDict=OrderedDict()

def chronomat(method):
    """
    Method to measure execution time of single methods.
    Simply add the decorator @chronomat to your method to print methods execution time at the 
    end of the code run. Stores values in a global dictionary.
    """
    @wraps(method)
    def timeThis(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        timelogDict[method.__module__+'.'+method.__name__]='%2.2f ms' % ( (te - ts) * 1000)
        return result
    return timeThis

def accumulativeChronomat(method):
    """
    Method to measure execution time of single methods.
    Simply add the decorator @chronomat to your method to print methods execution time at the 
    end of the code run. Stores values in a global dictionary.
    """
    @wraps(method)
    def timeThis(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'acc_'+method.__module__+'.'+method.__name__ in timelogDict:
            timelogDict['acc_'+method.__module__+'.'+method.__name__]+=(te - ts) * 1000
        else:
            timelogDict['acc_'+method.__module__+'.'+method.__name__]=(te - ts) * 1000
        return result
    return timeThis


def min_max_scaling(x, axis=None):
     """Normalized to [-1, 1]"""
     min = x.min(axis=axis, keepdims=True)
     max = x.max(axis=axis, keepdims=True)
     result = (x-min)/(max-min)
     result = 2.*result-1.
     return result

def getEfficiency(trueLabel,predLabel):
  if abs(len(trueLabel)-len(predLabel))>0:
    print("Error labellist length")
  print(trueLabel)
  print(predLabel)
  nSig=np.count_nonzero(trueLabel==1)
  nBg=np.count_nonzero(trueLabel==0)
  nSigTP=nSigFP=0.
  nBgTP=nBgFP=0.
  for lab in range(len(trueLabel)):
    if trueLabel[lab]==predLabel[lab]:
      if trueLabel[lab]==1: nSigTP+=1. 
      if trueLabel[lab]==0: nBgTP+=1.
    #FP bg
    if trueLabel[lab]==0 and predLabel[lab]==1:
      nSigFP+=1
    if trueLabel[lab]==1 and predLabel[lab]==0:
      nBgFP+=1
   
  sigEff=nSigTP/nSig
  bgEff=nBgTP/nBg
  bgFPrate=nSigFP/nBg

  effDict={
    "nSig":   nSig,
    "nBg":    nBg,
    "nSigTP": nSigTP,
    "nSigFP": nSigFP,
    "nBgTP":  nBgTP,
    "nBgFP":  nBgFP,
    "sigEff": sigEff,
    "bgEff":  bgEff,
    "bgFPrate":  bgFPrate,
  }
  print("SigEff:",sigEff)
  print("BgEff:",bgEff)
  print("FPrate:",bgFPrate)
  return sigEff
