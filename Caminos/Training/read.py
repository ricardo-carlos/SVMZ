#!/usr/bin/env python

# -*- coding: utf-8 -*-

import sys
import json

with open(sys.argv[1]) as f:
    data = json.load(f)


"""rot_acc_z = data['rot_acc_z']
rot_acc_x = data['rot_acc_x']
rot_acc_y = data['rot_acc_y']"""

speed = data['speed']
sp = []
for s in speed:
	sp.append(s['speed'])
print sp



# print '/', data.keys()
# print '/metadata/', data['metadata'].keys()
# print '/anomalies/', data['metadata'].keys()

"""for a in data['anomalies']:
	print a['start'], a['end'], a['type']"""