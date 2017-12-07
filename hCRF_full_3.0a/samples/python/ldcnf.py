#!/usr/bin/python
#ldcnf.py file.
#Julien-Charles Levesque
#8/19/2011

#System libs.
import sys

#3rd party libs.
import numpy as np

#Home-made scripts
import file_util as fu
import data_util as du

#This will only work as long as the script is invoked directly
#from command line or from a bash script.
results_dir = fu.backup_scriptfile(sys.argv[0])

#For this to work, you need a compiled version of hCRF lib in the same folder
#as this script or in a folder located in the PYTHONPATH.
import PyHCRF as pyhcrf

data_dir = '../../data/idiap/'

train_data_file = data_dir + 'dataTrain.csv'
train_labels_file = data_dir + 'labelsTrain.csv'
#val_data_file = data_dir + 'val_data_w_bias.csv'
#val_labels_file = data_dir + 'val_labels.csv'
#test_data_file = data_dir + 'dataTest.csv'
#test_labels_file = data_dir + 'labelsTest.csv'


toolbox = crf.ToolboxLDCRF(2)
toolbox.addFeatureFunction(crf.START_FEATURE_ID)
toolbox.addFeatureFunction(crf.EDGE_FEATURE_ID)
toolbox.addFeatureFunction(crf.GATE_NODE_FEATURE_ID,40,1)
toolbox.setOptimizer(crf.OPTIMIZER_LBFGS)
toolbox.setGradient(crf.GRADIENT_LDCNF)

train_data = crf.DataSet()
train_data.load(train_data_file,train_labels_file)

val_data = crf.DataSet()
val_data.load(val_data_file,val_labels_file)

toolbox.setDebugLevel(1)
toolbox.setMaxNbIteration(100)
toolbox.train(train_data,True)

toolbox.save( results_dir + 'model_cnf', results_dir + 'features_cnf')
toolbox.test(train_data, results_dir + 'train', results_dir + 'train_stats')
toolbox.test(val_data, results_dir + 'val', results_dir + 'val_stats')