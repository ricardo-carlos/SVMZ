#!/usr/bin/env python

# -*- coding: utf-8 -*-

import sys
import json

import matplotlib.pyplot as plt

with open(sys.argv[1]) as f:
    data = json.load(f)

nlist = []
nlist1 = []
print "\n"
for a in data['anomalies']:
	print a['start'], a['end'], a['type']
	nlist.append( ((a['end'] + a['start'])/2) )
print "\n"


h = int(nlist[-1] + 1)


for e in range(0, h):
	for a in nlist:
		if (e == a):
			nlist1.append(15)
	nlist1.append(0)
	


plt.grid(True)
plt.plot(data['rot_acc_x'], label = "Eje X")
plt.plot(data['rot_acc_y'], label = "Eje Y")
plt.plot(data['rot_acc_z'], label = "Eje Z")
plt.plot(nlist1, 'o', label = "Event")
plt.legend(loc = "upper right")

plt.show()