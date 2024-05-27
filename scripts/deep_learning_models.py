import argparse
from models.dCAM.src.models.CNN_models import ResNetBaseline,dResNetBaseline, ModelCNN
from pytorch_utils import transform_data4ResNet, transform2tensors, transform4ConvTran
from models.MyModels.MyMiniRocket import MyMiniRocket
# from load_data import load_data
import torch
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from models.ConvTran.utils import Setup, Initialization
from models.ConvTran.hyper_parameters import params as transform_params
from models.ConvTran.Models.model import ConvTran
from models.ConvTran.Training import SupervisedTrainer, train_runner
from copy import deepcopy


""" functions to train classifiers """

def train_ConvTran(dataset_name, n_classes, X_train, y_train, X_test, y_test):
    """
    function to train ConvTran
    """

    # load ConvTran parameters dictionary
    config = deepcopy(transform_params)
    device = Initialization(config)

    # transform data to pytorch tensor and initializing the model
    train_loader, test_loader, enc = transform4ConvTran(config, n_classes, X_train, y_train, X_test, y_test)
    model = ConvTran(config, num_classes=n_classes).to(device)

    # set optimizer and instantiate train and val loader
    config['optimizer'] = torch.optim.Adam(model.parameters())

    trainer = SupervisedTrainer(model, train_loader, device, config['loss_module'], config['optimizer'], l2_reg=0,
                                print_interval=config['print_interval'], console=config['console'],
                                print_conf_mat=False)
    val_evaluator = SupervisedTrainer(model, test_loader, device, config['loss_module'],
                                      print_interval=config['print_interval'], console=config['console'],
                                      print_conf_mat=True)

    # define save path and train
    file_path = "./saved_models/" + dataset_name + "/ConvTran_" + ".pt"
    acc = train_runner(config, model, trainer, val_evaluator, file_path)
    print("ConvTran accuracy was", acc)


def train_ResNet(X_train, y_train, X_test, y_test, dataset_name,
                device, n_channels, n_classes, n_filters=128):
    """
    function to train dResNet and ResNet
    """

    train_loader, test_loader, enc = transform_data4ResNet(X_train, y_train, X_test, y_test)
    resNet = ResNetBaseline(n_channels, mid_channels=n_filters, num_pred_classes=n_classes).to(device)
    # load data and instantiate classifier accordingly
    # if model_n == "ResNet":
    #     train_loader, test_loader, enc = transform_data_resnet()
    #     resNet = ResNetBaseline(n_channels, mid_channels=n_filters, num_pred_classes=n_classes).to(device)
    # else:
    #     train_loader, test_loader, enc = transform_data_dResnet()
    #     resNet = dResNetBaseline(n_channels, mid_channels=n_filters, num_pred_classes=n_classes).to(device)

    # define ModelCNN wrapper and train
    model = ModelCNN(model=resNet, n_epochs_stop=50, device=device, save_path="saved_models/tmp.pt")
    acc = model.train(num_epochs=300, train_loader=train_loader, test_loader=test_loader)

    # save the model
    model_name = "_".join((model_n, str(n_filters)))
    file_path = "/".join(("saved_models", dataset_name, model_name))
    # os.rename("saved_models/tmp.pt",file_path+".pt")
    torch.save(resNet, file_path+".pt")
    torch.cuda.empty_cache()
    print(model_n, " accuracy was ", acc)


# def main(args):

#     # get arguments and device to be used
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     dataset_name = args.dataset
#     classifier_name = args.classifier

#     # load data
#     train_X, train_y, test_X, test_y, seq_len, n_channels, n_classes = load_data(dataset_name, concat=False)

#     # train accordingly to selected classifier
#     if classifier_name.lower().count("rocket")>0:
#         train_mini_rocket(dataset_name, device, n_channels, n_classes, seq_len, test_X, test_y, train_X,
#                   train_y, classifier_name)

#     elif classifier_name.lower().count("resnet")>0:

#         transform_data_dResnet = lambda : transform_data4ResNet(X_train= train_X, y_train= train_y,
#                          X_test= test_X, y_test= test_y, batch_s=(32,32))
#         transform_data_resnet = lambda : transform2tensors(train_X,train_y,test_X,test_y, batch_size=(32,32),
#                                                            device=device )

#         train_d_ResNet(dataset_name, device, classifier_name, n_channels, n_classes,
#                        transform_data_dResnet, transform_data_resnet)
#     elif classifier_name.lower()=="convtran":
#         train_ConvTran(dataset_name, n_classes, test_X, test_y, train_X, train_y)


# if __name__ == "__main__" :
#     parser = argparse.ArgumentParser()
#     parser.add_argument("dataset", type=str, help="which dataset to be used")
#     parser.add_argument("classifier", type=str, help="which classifier to be trained")
#     parser.add_argument("--concat", type=bool, default=False, help="whether or not to concatenate the dataset")
#     args = parser.parse_args()
#     main(args)