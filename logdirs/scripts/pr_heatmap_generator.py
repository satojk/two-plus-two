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
ws = {}
w = good['test_data'][-1]['a1r']['weights']
for i, label in enumerate(labels):
	ws[label] = w[i]
d = {
	'zero': '0',
	'one': '1',
	'two': '2',
	'three': '3',
	'four': '4',
	'five': '5',
	'six': '6',
	'seven': '7',
	'eight': '8',
	'nine': '9',
	'ten': '10',
	'over': 'over',
	'under': 'under'}
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

rws = {}
for k, v in ws.items():
	new_val = {}
	for i, label in enumerate(labels):
		new_val[d[label]] = v[i]
	rws[d[k]] = new_val
df = pd.DataFrame(rws)
df = df.reindex(sorted(df.columns), axis=1)
new_df = df.reindex(index=df.index[::-1])
sns.heatmap(new_df, linewidth=0.3, center=0, cmap='coolwarm_r')
ax = plt.title('Weight heatmap between output of P-Module and Pr layer')
fig = ax.get_figure()
fig.savefig('pr_heatmap_2.png')
