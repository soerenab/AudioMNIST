#Making root folder available to notebook
import os
import sys 

if os.path.split(os.getcwd())[0] not in sys.path : sys.path.append(os.path.split(os.getcwd())[0])