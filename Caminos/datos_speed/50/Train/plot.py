#!/usr/bin/env python

# -*- coding: utf-8 -*-

import sys
import json

import matplotlib.pyplot as plt

with open(sys.argv[1]) as f:
    data = json.load(f)

print "\n"
for a in data['anomalies']:
	print a['start'], a['end'], a['type']
print "\n"

plt.grid(True)
plt.plot(data['rot_acc_z'], label = "Z axis")
plt.plot(data['rot_acc_x'], label = "X axis")
plt.plot(data['rot_acc_y'], label = "Y axis")

plt.legend(loc = "upper rigth")
plt.show()