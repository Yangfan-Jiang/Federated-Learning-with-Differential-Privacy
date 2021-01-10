"""
Implementation of several useful tool
Please copy this script to target path
"""
import numpy as np
import random


def bank_non_iid():
    # split bank data into non-iid data
    d = np.load("bank-full.npy")
    # pth = "./non-iid/3clients/"
    tot = d.shape[0]  # total number of data
    train_id = np.array(random.sample(range(tot), int(tot*0.8)))
    test_id = np.array(list(set(range(tot)) - set(train_id)))

    train_set = d[train_id]
    test_set = d[test_id]
    tot = train_id.shape[0]
    loc_size = int(tot / 3)
    d1 = train_set[:loc_size]
    d2 = train_set[loc_size:   2*loc_size]
    d3 = train_set[2*loc_size:]# 3*loc_size]
    #d4 = train_set[3*loc_size: 4*loc_size]
    #d5 = train_set[4*loc_size:]
    np.save('./non-iid/3clients/bank1.npy', d1)
    np.save('./non-iid/3clients/bank2.npy', d2)
    np.save('./non-iid/3clients/bank3.npy', d3)
    #np.save('bank4.npy', d4)
    #np.save('bank5.npy', d5)
    np.save('./non-iid/3clients/bank4.npy', test_set)


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 100, 600
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, int(num_shards/num_users), replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


if __name__ == '__main__':
    bank_non_iid()
