import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from .augmentations import DataTransform

from sklearn import preprocessing

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]


        X_train = X_train.permute(0,2,1)


        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]

        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index], index
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index], index

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode, args):
    if configs.data_name in [
        'FingerMovements',
        'LSST',
        'UWaveGestureLibrary',
        'InsectWingbeat',
    ]:
        train_dataset = torch.load(os.path.join(data_path,configs.data_name, "train.pt"))
        valid_dataset = torch.load(os.path.join(data_path,configs.data_name, "val.pt"))
        test_dataset = torch.load(os.path.join(data_path, configs.data_name,"test.pt"))

        X_train, y_train = np.array(train_dataset['samples']).astype(np.float32), np.array(train_dataset['labels']).astype(np.float32)
        X_test, y_test = np.array(valid_dataset['samples']).astype(np.float32), np.array(valid_dataset['labels']).astype(np.float32)
        X_val, y_val = np.array(test_dataset['samples']).astype(np.float32), np.array(test_dataset['labels']).astype(np.float32)

        x_all = np.concatenate((X_train, X_test, X_val), axis=0)
        y_all = np.concatenate((y_train, y_test, y_val), axis=0)

        if not np.isfinite(x_all).all():
            x_all[np.isnan(x_all)] = 0
            x_all[np.isinf(x_all)] = 0

    elif configs.data_name in [
        'Plane',
        'Wafer',

    ]:
        x_train, y_train, x_test, y_test = TSC_data_loader(data_path+'/', configs.data_name)
        x_all = np.concatenate((x_train, x_test), axis=0)
        y_all = np.concatenate((y_train, y_test), axis=0)
        x_all = x_all.reshape(x_all.shape[0],x_all.shape[1],1)



    ts_idx = list(range(x_all.shape[0]))
    np.random.shuffle(ts_idx)
    x_all = x_all[ts_idx]
    y_all = y_all[ts_idx]

    train_set, val_set,test_set = {},{},{}

    label_idxs = np.unique(y_all)
    class_stat_all = {}
    for idx in label_idxs:
        class_stat_all[idx] = len(np.where(y_all == idx)[0])
    print("[Stat] All class: {}".format(class_stat_all))

    train_idx = []
    val_idx = []
    test_idx = []

    for idx in label_idxs:
        target = list(np.where(y_all == idx)[0])
        nb_samp = int(len(target))
        train_idx += target[:int(nb_samp * 0.6)]
        val_idx += target[int(nb_samp * 0.6):int(nb_samp * 0.8)]
        test_idx += target[int(nb_samp * 0.8):]

    x_train = torch.tensor(x_all[train_idx])
    y_train = torch.tensor(y_all[train_idx])
    x_val = torch.tensor(x_all[val_idx])
    y_val = torch.tensor(y_all[val_idx])
    x_test = torch.tensor(x_all[test_idx])
    y_test = torch.tensor(y_all[test_idx])

    train_set['samples'] = x_train
    train_set['labels'] = y_train
    val_set['samples'] = x_val
    val_set['labels'] = y_val
    test_set['samples'] = x_test
    test_set['labels'] = y_test


    train_dataset = Load_Dataset(train_set, configs, training_mode)
    valid_dataset = Load_Dataset(val_set, configs, training_mode)
    test_dataset = Load_Dataset(test_set, configs, training_mode)

    if(training_mode == "self_supervised"):
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                                   shuffle=True, drop_last=configs.drop_last,
                                                   )
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                                   shuffle=False, drop_last=configs.drop_last,
                                                   )

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                                  shuffle=False, drop_last=False,
                                                  )
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size_finetune,
                                                   shuffle=True, drop_last=configs.drop_last,
                                                   )
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size_finetune,
                                                   shuffle=False, drop_last=configs.drop_last,
                                                   )

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size_finetune,
                                                  shuffle=False, drop_last=False,
                                                  )




    return train_loader, valid_loader, test_loader
def TSC_data_loader(dataset_path,dataset_name):
    print("[INFO] {}".format(dataset_name))

    #ScreenType:  Train size: 375  Time series length: 720  Train_dataset:(375,721)
    Train_dataset = np.loadtxt(
        dataset_path + dataset_name + '/' + dataset_name + '_TRAIN.tsv')
    Test_dataset = np.loadtxt(
        dataset_path + dataset_name + '/' + dataset_name + '_TEST.tsv')
    Train_dataset = Train_dataset.astype(np.float32)
    Test_dataset = Test_dataset.astype(np.float32)

    X_train = Train_dataset[:, 1:]
    y_train = Train_dataset[:, 0:1]

    X_test = Test_dataset[:, 1:]
    y_test = Test_dataset[:, 0:1]
    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(y_train, axis=1))
    y_train = le.transform(np.squeeze(y_train, axis=1))
    y_test = le.transform(np.squeeze(y_test, axis=1))
    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test

def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))



def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a
