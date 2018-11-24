'''
Load saved CIFAR10 example using Keras
'''

import os
import shutil
import numpy as np

from keras.models import load_model


##############################################
# Set up directories
##############################################

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))
TB_LOG_DIR = os.path.join(SCRIPT_DIR, 'tb_logs')
CHKPT_DIR = os.path.join(SCRIPT_DIR, 'ckpts')
CHKPT_FILE = os.path.join(CHKPT_DIR,'model.ckpt')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')




##############################################
# Load saved Keras model
##############################################
model = load_model(os.path.join(MODEL_DIR, 'model.h5'))

print(model.inputs)
print(model.outputs)


# print a summary of the model
print(model.summary())
