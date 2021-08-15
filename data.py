#!/usr/bin/env python
#coding=utf-8
import pickle
import sys, os, re, subprocess, math
from os.path import abspath, dirname, join
whereami = abspath(dirname(__file__))
sys.path.append(whereami)
import pandas as pd

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_dataset(name):

	if name == 'hepatitis':
		fileName = 'hepatitis.csv'
		df = pd.read_csv( join(whereami+'/dataset', fileName ), header = None, sep = ',')
		df = df.applymap(lambda x: np.nan if '?' in str(x) else x)
		df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
		tempX = df.values[:,0:-2]
		y = LabelEncoder().fit_transform(df.values[:,-1])

	if name == 'liver': 
		fileName = 'Liver.csv'
		df = pd.read_csv( join(whereami+'/dataset', fileName ), header = 0, sep = ',')		
		tempX = df.values[:,0:-1]
		y = LabelEncoder().fit_transform(df.values[:,-1])
	
	if name == 'blood': 
		fileName = 'Blood.csv'
		df = pd.read_csv( join(whereami+'/dataset', fileName ), header = 0, sep = ',')
		df = df.applymap(lambda x: np.nan if '?' in str(x) else x)
		df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))		
		tempX = df.values[:,1:-1]
		y = LabelEncoder().fit_transform(df.values[:,-1])

	if name == 'mammographic':  
		fileName = 'mammographic_masses_data.csv'
		df = pd.read_csv( join(whereami+'/dataset', fileName ), header = 0, sep = ',')
		df = df.applymap(lambda x: np.nan if '?' in str(x) else x)
		df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
		tempX = df.values[:,1:-1]
		y = LabelEncoder().fit_transform(df.values[:,-1])

	if name == 'loan':
		fileName = 'loan.csv'
		df = pd.read_csv( join(whereami+'/dataset', fileName ), header = None, sep = ',')
		df = df.applymap(lambda x: np.nan if '?' in str(x) else x)
		df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
		tempX = df.values[:,0:-1]
		y = LabelEncoder().fit_transform(df.values[:,-1])
		
	if name == 'tae': 
		fileName = 'tae.csv'
		df = pd.read_csv( join(whereami+'/dataset', fileName ), header = None, sep = ',')
		df = df.applymap(lambda x: np.nan if '?' in str(x) else x)
		df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
		tempX = df.values[:,:-1]
		y = LabelEncoder().fit_transform(df.values[:,-1])


	if name == 'cleveland':
		fileName = 'cleveland.data'
		df = pd.read_csv( join(whereami+'/dataset', fileName ), header = None, sep = ',')
		df = df.applymap(lambda x: np.nan if '?' in str(x) else x)
		df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
		tempX = df.values[:,:-1]
		y = LabelEncoder().fit_transform(df.values[:,-1])

	if name == 'cmc':
		fileName = 'cmc.data'
		df = pd.read_csv( join(whereami+'/dataset', fileName ), header = None, sep = ',')
		df = df.applymap(lambda x: np.nan if '?' in str(x) else x)
		df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
		df_sub_copy = df[ list(df.columns)[2:-1] ].copy()
		tempX1 = pd.get_dummies(df_sub_copy).values
		tempX2 = df.values[:,0:2]
		tempX = np.concatenate((tempX1, tempX2), axis=1)
		y = LabelEncoder().fit_transform(df.values[:,-1])
		

	if name == 'wine':
		fileName = 'wine_quality_red.csv'
		df = pd.read_csv( join(whereami+'/dataset', fileName ), header = 0, sep = ',')
		tempX = df.values[:,1:-2]
		y = LabelEncoder().fit_transform(df.values[:,-1])
		print(tempX[0], y[0], tempX.shape)


	if name == 'abalone':
		fileName = 'abalone.csv'
		df = pd.read_csv( join(whereami+'/dataset', fileName ), header = 0, sep = ',')
		tempX = df.values[:,2:-2]
		y = LabelEncoder().fit_transform(df.values[:,1])
		print(tempX[0], y[0], tempX.shape)


	X = tempX.astype(float)  
	return X,y




def loadSeer(numberOfClass):
	
	def decide_range(time_range, target):
		if target == -1:
			import random
			print "target <= 0, please double check"
			raw_input()
			return random.randint(1,3)
		for i in range(len(time_range)):
			if time_range[i]<=target<=time_range[i+1]:
				return i

	records = pd.read_csv( join(whereami+'/dataset', 'seer.log' ), header = 0, sep = '\t')	
	label = "SRV_TIME_MON"					
	result = records.sort([label], ascending=True, axis = 0)[label]
	
	list_range = [0]
	for temp in range(1, numberOfClass):
		temp_p = result.values[ int(result.count()/numberOfClass * temp) ]
		list_range.append( temp_p )
	list_range.append( records[label].max(axis=0) )		
	
	records[label + u'.outcome'] = records.apply(lambda x: decide_range( list_range , x[label]), axis=1)
	records.drop("SRV_TIME_MON", axis=1, inplace=True)
	
	tag = "SRV_TIME_MON.outcome"		
	y = records[tag].values
	X = records[ list(set(records.columns) - set(tag)) ].values
	
	from sklearn.preprocessing import normalize
	X = normalize(X, axis=0, norm='l2')
	
	return X, y


if __name__ == '__main__':
	loadSeer(numberOfClass = 3)