import argparse
from scripts.dCAM.src.models.CNN_models import ResNetBaseline,dResNetBaseline, ModelCNN
from scripts.pytorch_utils import transform_data4ResNet, transform2tensors, transform4ConvTran
from scripts.MyModels.MyMiniRocket import MyMiniRocket
# from load_data import load_data
import torch
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from scripts.ConvTran.utils import Setup, Initialization
from scripts.ConvTran.hyper_parameters import params as transform_params
from scripts.ConvTran.Models.model import ConvTran
from scripts.ConvTran.Training import SupervisedTrainer, train_runner
from scripts import farseeing
from copy import deepcopy

def train_eval(dataset_name, n_classes, X_train, y_train, X_test, y_test):
     # load ConvTran parameters dictionary
    config = deepcopy(transform_params)
    device = Initialization(config)

    X_train, X_test = farseeing.expand_for_ts(X_train, X_test)
    # transform data to pytorch tensor and initializing the model
    train_loader, test_loader, enc = transform4ConvTran(config, n_classes, X_train, y_train, X_test, y_test)
    model = ConvTran(config, num_classes=n_classes).to(device)

    # set optimizer and instantiate train and val loader
    config['optimizer'] = torch.optim.Adam(model.parameters())

    trainer = SupervisedTrainer(model, train_loader, device, config['loss_module'],
                                config['optimizer'], l2_reg=0, print_interval=config['print_interval'], console=config['console'], print_conf_mat=False)
    val_evaluator = SupervisedTrainer(model, test_loader, device, config['loss_module'],
                                      print_interval=config['print_interval'], console=config['console'],
                                      print_conf_mat=False)
    # define save path and train
    file_path = "./saved_models/" + dataset_name + "/ConvTran_" + ".pt"
    acc = train_runner(config, model, trainer, val_evaluator, file_path)
    print("ConvTran accuracy was", acc)