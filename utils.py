import pandas as pd
import numpy as np
import os

PATH = 'data/movielens/'
TRAINFILE = PATH + 'train.csv'
TESTFILE = PATH + 'test.csv'
VALIDFILE = PATH + 'val.csv'
MAPFILE=PATH+'item2id.map'

def get_item():
    train = pd.read_csv(TRAINFILE, sep='\t')
    valid = pd.read_csv(VALIDFILE, sep='\t')
    test = pd.read_csv(TESTFILE, sep='\t')
    data = pd.concat([train, valid, test])
    return data.item.unique()


def load_data(f, max_len):
    if os.path.exists(MAPFILE):
        item2idmap = {}
        with open(MAPFILE,'r') as fi:
            lines=fi.readlines()
            for line in lines:
                k, v = line.strip().split('\t')
                item2idmap[int(k)] = int(v)
    else:
        items = get_item()
        item2idmap = dict(zip(items, range(items.size)))
        with open(MAPFILE, 'w') as fo:
            for k, v in item2idmap.items():
                fo.write(str(k) + '\t' + str(v) + '\n')
    n_items = len(item2idmap)
    data = pd.read_csv(f, sep='\t')
    data['item'] = data['item'].map(item2idmap)
    data = data.sort_values(by=['Time']).groupby('user')['item'].apply(list).to_dict()
    new_x = []
    new_y = []
    for k, v in data.items():
        if len(v) < max_len + 1:
            continue
        x = v[:max_len]
        y = [0] * n_items
        y[v[max_len]] = 1
        new_x.append(x)
        new_y.append(y)
    return new_x, new_y, n_items

def load_train(max_len):
    return load_data(TRAINFILE, max_len)

def load_valid(max_len):
    return load_data(VALIDFILE, max_len)

def load_test(max_len):
    return load_data(TESTFILE, max_len)

def cal_eval(top_k_index,labels):
    users = np.shape(top_k_index)[0]
    top_k = np.shape(top_k_index)[1]
    hits, ndcg, mrr = 0, 0.0, 0.0
    for i in range(users):
        cur_user = top_k_index[i]
        for j in range(top_k):
            if labels[i] == cur_user[j]:
                hits += 1
                mrr += 1 / (1 + j)
                dcg = 1 / np.log2(1 + 1 + j)
                idcg = 1 / np.log2(1 + 1)
                ndcg += dcg / idcg
                break
    return hits / users,ndcg / users,mrr / users

if __name__=="__main__":
    load_train(16)