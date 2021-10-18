import os
import random as rn 
import numpy as np
from tensorflow import random 

def fix_seed(seed, fix_it= True):
    if(fix_it) : 
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(seed)
        rn.seed(seed)
        random.set_seed(seed)
    else : 
        pass 