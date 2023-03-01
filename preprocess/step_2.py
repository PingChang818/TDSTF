import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
pd.set_option('mode.chained_assignment', None)

# Read extracted time series data.
events = pd.read_csv('data/mimic_iii_events.csv', low_memory = False, usecols=['HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'VALUENUM', 'TABLE', 'NAME'])
icu = pd.read_csv('data/mimic_iii_icu.csv')
# Convert times to type datetime.
events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
icu.INTIME = pd.to_datetime(icu.INTIME)
icu.OUTTIME = pd.to_datetime(icu.OUTTIME)

# Assign ICUSTAY_ID to rows without it. Remove rows that can't be assigned one.
icu['icustay_times'] = icu.apply(lambda x:[x.ICUSTAY_ID, x.INTIME, x.OUTTIME], axis=1)
adm_icu_times = icu.groupby('HADM_ID').agg({'icustay_times':list}).reset_index()
icu.drop(columns=['icustay_times'], inplace=True)
events = events.merge(adm_icu_times, on=['HADM_ID'], how='left')
idx = events.ICUSTAY_ID.isna()
tqdm.pandas()
def f(x):
    chart_time = x.CHARTTIME
    for icu_times in x.icustay_times:
        if icu_times[1]<=chart_time<=icu_times[2]:
            return icu_times[0]
events.loc[idx, 'ICUSTAY_ID'] = (events.loc[idx]).progress_apply(f, axis=1)
events.drop(columns=['icustay_times'], inplace=True)
events = events.loc[events.ICUSTAY_ID.notna()]
events.drop(columns=['HADM_ID'], inplace=True)

# Filter icu table.
icu = icu.loc[icu.ICUSTAY_ID.isin(events.ICUSTAY_ID)]

# Get rel_charttime in minutes.
events = events.merge(icu[['ICUSTAY_ID', 'INTIME']], on='ICUSTAY_ID', how='left')
events['rel_charttime'] = events.CHARTTIME-events.INTIME
events.drop(columns=['INTIME', 'CHARTTIME'], inplace=True)
events.rel_charttime = events.rel_charttime.dt.total_seconds()//60

all_icustays = np.array(icu.ICUSTAY_ID)

# Get ts_ind.
def inv_list(x):
    d = {}
    for i in range(len(x)):
        d[x[i]] = i
    return d
icustay_to_ind = inv_list(all_icustays)
events['ts_ind'] = events.ICUSTAY_ID.map(icustay_to_ind)

# Rename some columns.
events.rename(columns={'rel_charttime':'minute', 'NAME':'variable', 'VALUENUM':'value'}, inplace=True)

# Add gender and age.
icu['ts_ind'] = icu.ICUSTAY_ID.map(icustay_to_ind)
data_age = icu[['ts_ind', 'AGE']]
data_age['variable'] = 'Age'
data_age.rename(columns={'AGE':'value'}, inplace=True)
data_gen = icu[['ts_ind', 'GENDER']]
data_gen.loc[data_gen.GENDER=='M', 'GENDER'] = 0
data_gen.loc[data_gen.GENDER=='F', 'GENDER'] = 1
data_gen['variable'] = 'Gender'
data_gen.rename(columns={'GENDER':'value'}, inplace=True)
data = pd.concat((data_age, data_gen), ignore_index=True)
data['minute'] = 0
events = pd.concat((data, events), ignore_index=True)

# Drop duplicate events.
events.drop_duplicates(inplace=True)

events = events.merge(icu[['ts_ind', 'HADM_ID', 'SUBJECT_ID']], on='ts_ind', how='left')
events.rename(columns={'HADM_ID':'hadm_id', 'SUBJECT_ID':'sub_id'}, inplace=True)

# Filter columns.
events = events[['ts_ind', 'minute', 'variable', 'value', 'hadm_id', 'sub_id']]

# Aggregate data.
events['value'] = events['value'].astype(float)
events = events.groupby(['ts_ind', 'minute', 'variable']).agg({'value':'mean', 'hadm_id':'first', 'sub_id':'first'}).reset_index()

# Get variable indices.
static_varis = ['Age', 'Gender']
ii = events.variable.isin(static_varis)
events = events.loc[~ii]
var = sorted(list(set(events.variable)))
def inv_list(l):
    d = {}
    for i in range(len(l)):
        d[l[i]] = i
    return d
var_to_ind = inv_list(var)

# target variables
target_var = []
target_var.append(var_to_ind['HR'])
target_var.append(var_to_ind['SBP'])
target_var.append(var_to_ind['DBP'])
target_var = np.array(target_var)
events['vind'] = events.variable.map(var_to_ind)
pickle.dump([var, target_var], open('data/var.pkl','wb'))

# 20 threads split
sets = np.array_split(events, 20)
pickle.dump(sets, open('data/sets.pkl','wb'))
