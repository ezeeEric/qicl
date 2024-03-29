""" Package wide definition of logging style

@author Eric "Dr. Dre" Drechsler (eric.drechsler@cern.ch)
"""

import logging

logging.addLevelName( logging.INFO,    "\033[1;95m{0}\033[1;0m".format('INFO '))
logging.addLevelName( logging.DEBUG,   "\033[1;96m{0}\033[1;0m".format('DEBUG'))
logging.addLevelName( logging.WARNING, "\033[1;93m{0}\033[1;0m".format('WARN '))
logging.addLevelName( logging.ERROR,   "\033[1;91m{0}\033[1;0m".format('ERROR'))

#add user defined level
logging.addLevelName( 1 ,              "\033[1;92m%s\033[1;0m" %    'TIME')

BOLD = "\033[1m"
RESET = "\033[0m"
logging.basicConfig( level=logging.INFO, format='{0}[%(asctime)s.%(msecs)03d]{1} %(levelname)8s  {0}%(name)-50s{1}%(message)s'.format(BOLD,RESET),datefmt="%H:%M:%S")
