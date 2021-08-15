#!/usr/bin/env python
#coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os
import types, random, time, re
from os.path import abspath, dirname, join
whereami = abspath(dirname(__file__))
sys.path.append(whereami)

from collections import defaultdict
import pandas as pd
import numpy as np


def appendDFToCSV_void(df, csvFilePath, sep=","):
    import os
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False) 

def getWfixingA(Z, W_k, A_k, lambda_W):
	beforeVal = np.linalg.norm( (Z - np.dot(A_k, W_k)), 2 ) + lambda_W * np.sum(np.sqrt(np.sum(W_k ** 2, 1)))
	alpha = 1 / np.linalg.norm(A_k, 2)
	
	W_new = np.zeros((W_k.shape[0], W_k.shape[1]))
	for i in range(0, W_k.shape[0]):		
		norm2 = np.linalg.norm( W_k[i,:], 2 )
		if norm2 != 0:
			constant = 1 / norm2 * np.maximum(0, norm2 - lambda_W * alpha)
			W_new[i,:] = constant * W_k[i,:]
		elif norm2 == 0:
			W_new[i,:] = 1e3
			
	afterVal = np.linalg.norm( (Z - np.dot(A_k, W_new)), 2 ) + lambda_W * np.sum(np.sqrt(np.sum(W_new ** 2, 1)))

	return W_new

def getSparseWeight(D, train_output, **kwargs):
	coef = 0; model = None
	#print kwargs
	if kwargs['choice'] == 0:
		# OMP
		from sklearn.linear_model import OrthogonalMatchingPursuit
		n_nonzero_coefs = int(D.shape[1] * kwargs['alpha'])
		model = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)		
	elif kwargs['choice'] == 1:
		#l_1	
		from sklearn.linear_model import LassoLars
		model = LassoLars(alpha = kwargs['alpha'])	
	else:
		print 'unknown input'
		exit()
	
	model.fit(D, train_output)
	coef = model.coef_  
	return coef


def stop_critier(valid_list):
	res = False
	if len(valid_list)>5 and (valid_list[-1] < valid_list[-2]) and (valid_list[-2] < valid_list[-3]) and (valid_list[-3] < valid_list[-4]) and (valid_list[-4] < valid_list[-5]):
		res = True
	
	return res