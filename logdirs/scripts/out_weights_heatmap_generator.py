# Code exported from a REPL session after an exploration tturned out very 
# fruitful. Please don't judge!!


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

with open('../logdir_008/runlog_0.pkl', 'rb') as pkl:
	good = pickle.load(pkl)

labels = sorted(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'over', 'under'])
labelsalt = sorted(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'over', 'under'])
ws = {}
w = good['test_data'][-1]['out']['weights']
for i, label in enumerate(labelsalt):
	ws[label] = w[i+12]
d = {
	'zero': '00',
	'one': '01',
	'two': '02',
	'three': '03',
	'four': '04',
	'five': '05',
	'six': '06',
	'seven': '07',
	'eight': '08',
	'nine': '09',
	'ten': '10',
	'over': 'over',
	'under': 'under'}

dalt = {
	'zero': '00',
	'one': '01',
	'two': '02',
	'three': '03',
	'four': '04',
	'five': '05',
	'six': '06',
	'seven': '07',
	'eight': '08',
	'nine': '09',
	'over': 'over',
	'under': 'under'}

rws = {}
for k, v in ws.items():
	new_val = {}
	for i, label in enumerate(labels):
		new_val[d[label]] = v[i+13]
	rws[d[k]] = new_val
df = pd.DataFrame(rws)
df = df.reindex(sorted(df.columns), axis=1)
new_df = df.reindex(index=df.index[::-1])
sns.heatmap(new_df, linewidth=0.3, center=0, cmap='coolwarm_r')
ax = plt.title('Weight heatmap b\w output of P-Module and 1s digit of output')
fig = ax.get_figure()
fig.savefig('out_heatmap_2.png')
