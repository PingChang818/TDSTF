import pickle
import numpy as np
import concurrent.futures

def sample(data, thread):
    _, target_var = pickle.load(open('data/var.pkl', 'rb'))
    icu_id = data.ts_ind.unique()
    samples = []
    info = []
    for i in icu_id:
        t = 0
        icu_data = data.loc[data.ts_ind==i]
        max_t = icu_data.minute.max()
        while (t + 40) < max_t:
            y = icu_data.loc[(icu_data.minute>=(t+30))&(icu_data.minute<(t+40))&(icu_data.vind.isin(target_var))]
            pick = False
            for j in target_var:
                if len(y.loc[y.vind==j]) > 1:
                    pick = True
                    break
            if pick:
                x = icu_data.loc[(icu_data.minute>=t)&(icu_data.minute<(t+30))]
                x = x.groupby('ts_ind').agg({'vind':list, 'minute':list, 'value':list})
                if len(x) > 0:
                    y = y.loc[y.vind.isin(target_var)].groupby('ts_ind').agg({'vind':list, 'minute':list, 'value':list})
                    x_vind = x['vind'].iloc[0]
                    y_vind = y['vind'].iloc[0]
                    lx = len(x_vind)
                    ly = len(y_vind)
                    x_minute = x['minute'].iloc[0]
                    x_minute = [m-t for m in x_minute]
                    y_minute = y['minute'].iloc[0]
                    y_minute = [m-t for m in y_minute]
                    x_value = x['value'].iloc[0]
                    y_value = y['value'].iloc[0]
                    vind = x_vind + y_vind
                    minute = x_minute + y_minute
                    value = x_value + y_value
                    mask = np.ones(lx + ly)
                    ymask = np.copy(mask)
                    ymask[: lx] = 0
                    samples.append([np.array(vind), np.array(minute), np.array(value), ymask])
                    info.append([x.index[0], icu_data.iloc[0].sub_id, lx, ly])
                    break
            t += 10
    pickle.dump([samples, info], open('data/first/samples_{}.pkl'.format(thread+1),'wb'))
    print('Thread_{} finished'.format(thread))

sets = pickle.load(open('data/sets.pkl', 'rb'))
threads = list(np.arange(len(sets)))

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(sample, sets, threads)