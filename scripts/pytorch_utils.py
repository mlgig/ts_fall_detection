import numpy as np
from scripts.utils import one_hot_encoding, gen_cube, MyDataset
from torch.utils.data import DataLoader
import torch
from scripts.ConvTran.utils import dataset_class
from copy import deepcopy
from scripts.ConvTran.hyper_parameters import params as transform_params
from scripts.ConvTran.utils import Initialization
from scripts.ConvTran.Models.model import ConvTran
from scripts.ConvTran.Models.utils import load_model as load_transformer

def transform_data4ResNet( X_train,y_train,X_test,y_test,batch_s=(64,64)):

    # first transform labels
    y_train,y_test,enc = one_hot_encoding( y_train,y_test )

    # then transform train and test set
    train_set_cube = np.array([gen_cube(acl) for acl in X_train.tolist()])
    X_train = torch.tensor(train_set_cube).type(torch.float)
    y_train = torch.tensor(y_train).type(torch.int64)

    test_set_cube = np.array( [gen_cube(acl) for acl in X_test.tolist()] )
    X_test = torch.tensor(test_set_cube).type(torch.float)
    y_test =  torch.tensor(y_test).type(torch.int64)

    # return loader or single tensors accordingly to batch_s
    if batch_s is None:
        return  X_train, y_train,X_test,y_test, enc
    else:
        # otherwise return the loaders
        train_loader = DataLoader(MyDataset(X_train,y_train), batch_size=batch_s[0],shuffle=True)
        test_loader = DataLoader(MyDataset(X_test,y_test), batch_size=batch_s[1],shuffle=False)
        return train_loader, test_loader, enc


def transform2tensors(X_train, y_train, X_test,y_test,batch_size=None,device="cpu"):
    X_train = torch.tensor( X_train  ).type(torch.float).to(device)
    X_test = torch.tensor( X_test  ).type(torch.float).to(device)


    y_train,y_test,enc = one_hot_encoding( y_train,y_test )
    y_train = torch.Tensor( y_train ).type(torch.int64).to(device)
    y_test = torch.Tensor( y_test ).type(torch.int64).to(device)

    if batch_size==None:

        return X_train, y_train, X_test, y_test, enc

    else:
        train_loader = DataLoader(MyDataset(X_train,y_train), batch_size=batch_size[0],shuffle=True)
        test_loader = DataLoader(MyDataset(X_test,y_test), batch_size=batch_size[1],shuffle=False)
        return train_loader, test_loader, enc

########################   functions for ConvTran ###################################################
def transform4ConvTran(config, n_classes, test_X, test_y, train_X, train_y):

    train_y,test_y,enc = one_hot_encoding(train_y,test_y)

    if np.any(train_X!=None):
        train_dataset = dataset_class(train_X, train_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    else:
        train_loader=None

    test_dataset = dataset_class(test_X, test_y)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    config['num_labels'] = n_classes
    config['Data_shape'] = test_dataset.feature.shape
    config['loss_module'] = torch.nn.CrossEntropyLoss(reduction='none')
    return train_loader, test_loader, enc

def load_ConvTran(test_X, test_y, train_X, train_y, n_classes, path) :
    config = deepcopy( transform_params )
    device = Initialization(config)
    config['batch_size'] = 32
    _, test_loader, enc = transform4ConvTran(config, n_classes, test_X, test_y, train_X, train_y)

    model = ConvTran(config, num_classes=config['num_labels']).to(device)
    loaed_model = load_transformer(model,path)
    loaed_model.eval()

    return loaed_model, test_loader, enc ,device
