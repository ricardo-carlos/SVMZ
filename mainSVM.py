#!/usr/bin/env python

# -*- coding: utf-8 -*-

from itertools import islice
from collections import deque
from pylab import *
from math import sqrt
import numpy as np
from tqdm import tqdm
import sys
import json
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
import scipy, scipy.signal
import pywt
import pandas as pd
from scipy.signal import savgol_filter

import SVM


def sliding_window(iterable, size=25, step=1, fillvalue=9.81): #list, size of window, step, fill empty value
    if size < 0 or step < 1: raise ValueError
    it = iter(iterable)
    q = deque(islice(it, size), maxlen=size)
    if not q: return  # empty iterable or size == 0
    q.extend(fillvalue for _ in range(size - len(q)))  # pad to size
    while True:
        yield iter(q)  # iter() to avoid accidental outside modifications
        q.append(next(it))
        q.extend(next(it, fillvalue) for _ in range(step - 1))

def compare(anomalies,zlist):	
	tp = 0
	fn = 0
	fp = 0
	fp1 = 0
	y = 0
	fant = zlist[0]
	for x in anomalies:
		x['count'] = set()

	for i in zlist:
		for j, a in enumerate(anomalies):
			if ( (a['start']-15) <= i <= (a['end']+15)):
				a['count'].add(i)

	for a in anomalies:
		if (len(a['count']) >= 1):
			tp+=1		
	if len(anomalies) > tp: 
		fn = len(anomalies) - tp
	
	for f in zlist:
		for j, a in enumerate(anomalies):
			aux = f - fant
			if (aux < 100 ):
				if ( ( (a['start']-15) > f) or  (f > (a['end']+15)) ):
					y+=1 #revisar todos los segmentos posibles
			if(y == len(anomalies)): 
				fp+=1
			fant = f
		y = 0	

	return {
		'true_positives'    : tp,
		'false_negatives'   : fn,
		'false_positives'  : fp,
	}

def label(anomalies,wp):	
	l = 0
	y = 0
	for j, a in enumerate(anomalies):
		if ((a['start']-15) <= wp <= (a['end']+15)):
			l = 1

	for j, a in enumerate(anomalies):
		if ( ((a['start']-15) > wp) or  (wp > (a['end']+15)) ):
			y+=1 #revisar todos los segmentos posibles
	if(y == len(anomalies)): 
		l = 0
	y = 0					
	return l

def test(ltp,lfp,lfn):
	try:
		sensivity = float(ltp/float(ltp+lfn))
	except:
		pass

	try:
		precision = float(ltp)/(float(ltp)+float(lfp))
	except:
		precision = 1.0

	try:
		f1 = (2*(sensivity*precision))/(sensivity+precision)
	except:
		pass

	return sensivity,precision,f1

def extract_features(zlist,anomalies):
	sum1 = 0
	sumf = 0
	mean = 0
	meanc = 0
	stdev = 0
	var = 0
	pos = 0
	aux = 0
	cont = 0
	cv = 0
	g = 9.81
	u = 0.15*g
	u1 = 0.015*g
	ua = g + (g*0.3)
	ub = g - (g*0.3)
	cp = 0
	zdiff = 0
	ud = 0.2*g
	threshold = 0
	diff = 0
	X = []
	y = []
	z = []
	features = []
	varfl = []
	FMLP = []
	FMLP2 = []
	features2 = []
	labels = []
	c = 0
	varc = 0
	cvc = 0
	sc = 0
	meanf = 0
	varf = 0
	stdevf = 0
	cvf = 0

	window = sliding_window(zlist,size=30,step=15)

	for i, w in tqdm(enumerate(window)):
		l = list(w)

		jant = l[0]
		mean = sum(l) / len(l) #Mean

		for j in l:                         
			sum1 += (float(j) - mean)**2  
			pos+=1      
		var = float(sum1) / (len(l) - 1) # Variance
		stdev = sqrt(var) # Standard deviation
		cv = stdev / mean # Coefficient of variation
		pos2 = pos / 2
		diff = l[len(l)-1] - l[0]
		
		if (var > float(u)):
			varc = 0.8 # Threshold
			aux+=1
		else: varc = 0.2

		if (stdev > float(u)):
			threshold = 0.9 # Threshold
			aux+=1
		else: threshold = 0.2

		if (cv > float(u1)):
			cvc = 0.8 # Threshold
			aux+=1
		else: cvc = 0.2

		for j in l:
			if(float(j) > float(ua)) or (float(j) < float(ub)):
				cp = 0.6
				aux+=1
				break
			else: cp = 0.3
		
			if ( ((float(j) - float(jant)) > float(ud)) or ((float(j) - float(jant)) < -float(ud)) ):
				zdiff+=1
			jant = j

		if (aux >= 3): 
			cont = 0.8
		else: cont = 0.2

		sc = varc + threshold + cvc + cp + cont 
		if (sc >= 2):
			cs = 0.8
		else: cs = 0.2

		#Features

		lab = label(anomalies, pos2)
		features.append([mean,var,stdev,cv,cp,diff,threshold,varc,cvc,cont,sc*0.1,cs])
		labels.append([lab])

		X = features
		XX = FMLP
		y = labels

		aux = 0
		sum1 = 0
		stdev = 0
		cv = 0
		threshold = 0
		diff = 0
		c+=1
		cp = 0
		
	return X, y 

'''
	Main
'''

with open('Caminos/Training/P10-E.json') as f:
	datatraining = json.load(f)

with open('Caminos/Test/P10-P.json') as f:
	datatest = json.load(f)

zlist = datatraining['rot_acc_z']
#xlist = datatraining['rot_acc_x']
#ylist = datatraining['rot_acc_y']
anomalies = datatraining['anomalies']

zlist2 = datatest['rot_acc_z']
#xlist2 = datatest['rot_acc_x']
#ylist2 = datatest['rot_acc_y']
anomalies2 = datatest['anomalies']

# Get features and labels from training and test
X, y = extract_features(zlist,anomalies)
z, zz = extract_features(zlist2,anomalies2)


#window = sliding_window(zlist2,size=30,step=15)
#windowx = sliding_window(xlist2,size=30,step= 15) 
#windowy = sliding_window(ylist2,size=30,step=15)

'''
	Clasificadores segmentacion
'''
svm = []
# ------ SVM
#print "SVM: "
ressvm = SVM.SVM(X,y,z)
for i,p in enumerate(ressvm):
	if p == 1:
		svm.append(i*15)

resultsvm = compare(anomalies2,svm)
ltp = resultsvm['true_positives']
lfp = resultsvm['false_positives']
lfn = resultsvm['false_negatives']
s,p,f1 = test(ltp,lfp,lfn)

with open('SVM.txt','a') as f:
	f.write(str(svm))
	f.write(",")
	f.write(str(s))
	f.write(",")
	f.write(str(p))
	f.write(",")
	f.write(str(f1))
	f.write("\n")

nlist = []
nlist1 = []
svmlist = []

for a in datatest['anomalies']:
	print( a['start'], a['end'], a['type'])
	nlist.append( ((a['end'] + a['start'])/2) )
print("\n")

h = int(nlist[-1] + 1)

for e in range(0, h):
	for a in svm:
		if (e == a):
			svmlist.append(15)
	svmlist.append(0)

for i, z in enumerate(datatest['rot_acc_z']):
	for j, a in enumerate(anomalies2):
		if ( (a['start']) <= i <= (a['end'])):
			nlist1.append(15)
			break
	if ( ( (a['start']) > i) or  (i > (a['end'])) ):
		nlist1.append(0)

plt.grid(True)
plt.plot(datatest['rot_acc_z'], label = "Eje Z")
plt.plot(svmlist, 'o', label = "SVM")
plt.plot(nlist1, '-', color= 'g', label = "Event")
plt.legend(loc = "upper right")
plt.show()