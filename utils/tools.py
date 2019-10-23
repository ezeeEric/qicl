""" Definition of analysis independent, package wide tools.

@author Eric "Dr. Dre" Drechsler (eric.drechsler@cern.ch)
"""

import sys
from functools import wraps
import time

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
