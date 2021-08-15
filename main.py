#!/usr/bin/env python
#coding=utf-8
import pickle
import sys, os, re, subprocess, math
reload(sys)
sys.setdefaultencoding("utf-8")
from os.path import abspath, dirname, join
whereami = abspath(dirname(__file__))
sys.path.append(whereami)

from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from scipy import *
import itertools

from utility import getWfixingA, getSparseWeight, appendDFToCSV_void, stop_critier
from solver import optimize 
from data import load_dataset, loadSeer

def main(datasetName):
	if datasetName == 'seer':
		numberOfClass = 2
		X, y = loadSeer(numberOfClass)
	else:
		X, y = load_dataset(name=datasetName)

	from sklearn.model_selection import train_test_split
	#P_train, P_test, y_train, y_test = train_test_split(X, y, random_state=42)
	# split: 60-20-20
	P_train, P_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	P_train, P_val, y_train, y_val = train_test_split(P_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

	input_size, input_dimension = P_train.shape
	numberOfClass = np.unique(y).shape[0]
	
	P_train = np.transpose(P_train); P_val	= np.transpose(P_val); P_test	= np.transpose(P_test); 
	if numberOfClass > 2:	
		from sklearn.preprocessing import LabelBinarizer, StandardScaler
		lb = LabelBinarizer()
		y_train = lb.fit_transform(y_train); y_train = np.transpose(y_train);
		y_val = lb.fit_transform(y_val); y_val = np.transpose(y_val);
		y_test = lb.transform(y_test); y_test = np.transpose(y_test);
	elif numberOfClass == 2:	
		y_train = y_train.reshape((y_train.shape[0], 1)); y_train = np.transpose(y_train);
		y_val = y_val.reshape((y_val.shape[0], 1)); y_val = np.transpose(y_val);
		y_test = y_test.reshape((y_test.shape[0], 1)); y_test = np.transpose(y_test);

	print datasetName, input_size, input_dimension, numberOfClass
	rank = P_train.shape[1] 
	
	#grid search or pre-defined hyper-parameters
	lambda_X_list = [0.1]#[math.pow(10, x) for x in range(-4, 3)]; 
	lambda_W_list = [10.0]#[math.pow(10, x) for x in range(-4, 3)]; 
	lambda_D_list = [0.01]#[math.pow(10, x) for x in range(-4, 3)];	
	valid_list = []
	
	for (lambda_X, lambda_W, lambda_D) in itertools.product(lambda_X_list, lambda_W_list, lambda_D_list):
		try:
			###############################################
			# 0-th iteration
			##############################################
			X_ini = np.random.rand(P_train.shape[0], rank)
			W_ini = np.random.rand(rank, P_train.shape[1])
			D_ini = np.random.rand(y_train.shape[0], rank)			
			
			X = X_ini; D = D_ini; W = W_ini;			
			
			w_val = getSparseWeight(X, P_val, choice = 0, alpha = 0.3)
			w = getSparseWeight(X, P_test, choice = 0, alpha = 0.3)
			print 'iter', 'training', 'validation', 'testing'
			print 	0, roc_auc_score(y_train.T, np.dot(D, W).T), \
					roc_auc_score(y_val.T, np.dot(D, w_val.T).T), \
					roc_auc_score(y_test.T, np.dot(D, w.T).T)
			appendDFToCSV_void(pd.DataFrame([{"train":roc_auc_score(y_train.T, np.dot(D, W).T), "validation":roc_auc_score(y_val.T, np.dot(D, w_val.T).T), "test":roc_auc_score(y_test.T, np.dot(D, w.T).T)}]), join(whereami+'/res', datasetName+'.log'))

					
			from time import time
			start = time()
			###############################################
			# loop iteration
			##############################################
			for iter in range(1, 50):
				#update X
				X_new = optimize.min_rank_dict(P_train, W, lambda_X, X)
				
				#update W
				Z = np.concatenate((P_train,y_train),axis=0)
				A = np.concatenate((X_new,D),axis=0)
				W_new = getWfixingA(Z, W, A, lambda_W)
				
				#update D
				D_new = optimize.min_rank_dict(y_train, W_new, lambda_D, D)
				#D_new = np.dot(y_train,  np.linalg.pinv(W_new))
				#from sklearn.decomposition.dict_learning import _update_dict
				#D_new = _update_dict(D, y_train, W_new)
				#E = np.dot(y_train, W_new.T)
				#F = np.dot(W_new, W_new.T)
				#D_new = optimize.ODL_updateD(D, E, F, iterations = 1)
				

				w = getSparseWeight(X_new, P_test, choice = 0, alpha = 0.3)				
				w_val = getSparseWeight(X, P_val, choice = 0, alpha = 0.3)
				print 	iter, roc_auc_score(y_train.T, np.dot(D_new, W_new).T), \
						roc_auc_score(y_val.T, np.dot(D_new, w_val.T).T), \
						roc_auc_score(y_test.T, np.dot(D_new, w.T).T)

				appendDFToCSV_void(pd.DataFrame([{"train":roc_auc_score(y_train.T, np.dot(D_new, W_new).T),"validation":roc_auc_score(y_val.T, np.dot(D_new, w_val.T).T), "test":roc_auc_score(y_test.T, np.dot(D_new, w.T).T) }]), join(whereami+'/res', datasetName+'.log'))

				D = D_new; W = W_new; X = X_new; 
				valid_list.append( roc_auc_score(y_val.T, np.dot(D_new, w_val.T).T) )
				
				if stop_critier(valid_list):
					break

		except Exception as err:
			print( err ) 
			


if __name__ == '__main__':
	
	datasetName = sys.argv[1]
	main(datasetName)
	