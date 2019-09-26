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
    end of the plotalyzer run. Stores values in a global dictionary.
    """
    @wraps(method)
    def timeThis(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        timelogDict[method.__module__+'.'+method.__name__]='%2.2f ms' % ( (te - ts) * 1000)
        return result
    return timeThis

#        logger=logging.getLogger("Elapsed time for method")
#        logger.setLevel(1)
#        for key, item in timelogDict.items():
#            logger.log(1,'%s(): %s' % (key, item))

