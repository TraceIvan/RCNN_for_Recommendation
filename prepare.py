import pandas as pd
import numpy as np
def movielens_pre(maxlen=16):
    file="data/movielens/ratings.dat"
    data = pd.read_csv(file, sep='::', header=None)
    data.columns = ['user', 'item', 'rating', 'Time']
    data = data[data.rating >= 4]
    data.drop(['rating'], axis=1, inplace=True)
    event_lengths = data.groupby('user').size()
    print('Average check-ins per user: {}'.format(event_lengths.mean()))
    item_supports = data.groupby('item').size()
    data = data[np.in1d(data.item, item_supports[item_supports >= maxlen].index)]
    print('Unique items: {}'.format(data.item.nunique()))
    event_lengths = data.groupby('user').size()
    data = data[np.in1d(data.user, event_lengths[event_lengths >= maxlen].index)]
    tmin = data.Time.min()
    tmax = data.Time.max()
    train_end=(tmax - tmin) * 0.8 + tmin
    val_end=(tmax - tmin) * 0.9 + tmin
    train_data = data.loc[(data['Time'] >= tmin) & (data['Time'] <= train_end)]
    train_data = train_data.groupby("user").filter(lambda x: len(x) > maxlen)
    print('Train size: {}'.format(len(train_data)))
    print('Unique users: {}'.format(train_data.user.nunique()))
    print('Unique items: {}'.format(train_data.item.nunique()))
    val_data = data.loc[(data['Time'] >train_end) & (data['Time'] <= val_end)]
    val_data = val_data.groupby("user").filter(lambda x: len(x) > maxlen)
    print('val size: {}'.format(len(val_data)))
    print('Unique users: {}'.format(val_data.user.nunique()))
    print('Unique items: {}'.format(val_data.item.nunique()))
    test_data = data.loc[(data['Time'] > val_end) & (data['Time'] <= tmax)]
    test_data = test_data.groupby("user").filter(lambda x: len(x) > maxlen)
    print('val size: {}'.format(len(test_data)))
    print('Unique users: {}'.format(test_data.user.nunique()))
    print('Unique items: {}'.format(test_data.item.nunique()))
    train_data.to_csv('data/movielens1M/train.csv', sep='\t', index=False)
    val_data.to_csv('data/movielens1M/val.csv', sep='\t', index=False)
    test_data.to_csv('data/movielens1M/test.csv', sep='\t', index=False)

if __name__=="__main__":
    movielens_pre(maxlen=16)