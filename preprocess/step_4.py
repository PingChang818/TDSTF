import os
import pickle
import pandas as pd
import numpy as np

# target_var   HR:49 SBP:111 DBP:26
var, target_var = pickle.load(open('data/var.pkl', 'rb'))
samples = []
info = []

for file in os.listdir('data/first'):
    sr, ir = pickle.load(open('data/first/' + file, 'rb'))
    samples += sr
    info += ir
info = pd.DataFrame(info, columns = ['ts_ind', 'sub_id', 'x_len', 'y_len'])

# suject-wise data split
subjects = np.unique(np.array(info['sub_id']))
np.random.shuffle(subjects)
len_sub = len(subjects)
bp1, bp2 = int(0.64 * len_sub), int(0.8 * len_sub)
train_sub = subjects[: bp1]
valid_sub = subjects[bp1 : bp2]
test_sub = subjects[bp2 :]

# normalize
rec = []
for i in range(len(var)):
    rec.append([])
    
for sub_id in np.concatenate((train_sub, valid_sub)):
    index = np.array(info.loc[info.sub_id == sub_id].index)
    for i in index:
        stay = samples[i]
        for j in range(info.iloc[i]['x_len']):
            rec[stay[0][j]].append(stay[2][j])
            
mean = np.full(len(var), np.nan)
std = np.full(len(var), np.nan)
for i in range(len(var)):
    if len(rec[i]) > 0:
        mean[i] = np.array(rec[i]).mean()
        std[i] = np.array(rec[i]).std()

sub_sets = [train_sub, valid_sub, test_sub]
data_sets = [[], [], []]
info_sets = [[], [], []]
for si in range(len(sub_sets)):
    for sub_id in sub_sets[si]:
        index = np.array(info.loc[info.sub_id == sub_id].index)
        for i in index:
            stay = samples[i]
            for j in range(len(samples[i][0])):
                if np.isnan(mean[stay[0][j]]):
                    stay[2][j] = 0
                else:
                    s = 1 if std[stay[0][j]] == 0 else std[stay[0][j]]
                    stay[2][j] = (stay[2][j] - mean[stay[0][j]]) / s
            data_sets[si].append(stay)
            info_sets[si].append(info.iloc[i])

pickle.dump([mean, std], open('data/mean_std.pkl','wb'))
pickle.dump([data_sets[0], pd.DataFrame(info_sets[0]), data_sets[1], pd.DataFrame(info_sets[1]), data_sets[2], pd.DataFrame(info_sets[2])], open('data/dataset.pkl','wb'))
