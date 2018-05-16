#!/usr/bin/env python

# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn import preprocessing
import numpy as np

def SVM(features,labels,test):
	clf = svm.SVC(C=1,kernel='rbf',gamma=0.002,tol=0.001)
	clf.fit(features, labels) 
	
	return clf.predict(test)
