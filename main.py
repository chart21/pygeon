#main function
import cnns
import data_load
import train
import parameter_export
import torch
import numpy as np
import os
import argparse
from torch import nn
from torch import optim
# from deepreduce_models.resnet import *

def save_model(model,filepath):
    parameter_export.save_weights_compatible_with_cpp(model, filepath+'.bin')

def export_pth_model(model,filepath):
    torch.save(model.state_dict(), filepath+'.pth')

def train_model(model,dataset_name,num_epochs=80,lr=0.001, transform = "standard", criterion=nn.CrossEntropyLoss, optimizer=optim.Adam):
    train_loader, test_loader,num_classes = data_load.load_dataset(dataset_name, transform)
    train.train_and_evaluate(model, train_loader, test_loader, num_epochs, optimizer, lr, criterion)

def train_test_model(model,dataset_name,num_epochs=80,lr=0.001, transform = "standard", model_name="LeNet", criterion=nn.CrossEntropyLoss, optimizier=optim.Adam):
    train_loader, test_loader,num_classes = data_load.load_dataset(dataset_name, transform)
    train.train_test(model, train_loader, test_loader, num_epochs, optimizier, lr, criterion,  model_name, dataset_name, transform)

def load_model(model,filepath):
    model.load_state_dict(torch.load(filepath+'.pth', map_location=torch.device('cpu')))

def load_checkpoint(model,filepath):
    checkpoint = torch.load(filepath+'.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict({k.replace('module.',''): v for k, v in checkpoint['snet'].items()})


def test_model(model,dataset_name):
    train_loader, test_loader,num_classes = data_load.load_dataset(dataset_name)
    train.evaluate(model, test_loader)

def export_test_dataset(dataset_name):
    train_set, test_set,num_classes = data_load.load_dataset(dataset_name)
    data_load.export_dataset(test_set,'./data/'+dataset_name+'_test_images.bin','./data/'+dataset_name+'_test_labels.bin')

def train_all(num_epochs=80,lr=0.001,criterion=nn.CrossEntropyLoss,optimizer=optim.Adam):
    models_mnist = [(cnns.LeNet(num_classes=10),"LeNet")]
    models_cifar10 = [(cnns.AlexNet(num_classes=10),"AlexNet_CryptGPU"),(cnns.AlexNet_32(num_classes=10),"AlexNet_32"),(cnns.VGG16(num_classes=10),"VGG16"),(cnns.ResNet18_avg(num_classes=10),"ResNet18_avg"),(cnns.ResNet50_avg(num_classes=10),"ResNet50_avg"),(cnns.ResNet101_avg(num_classes=10),"ResNet101_avg"),(cnns.ResNet152_avg(num_classes=10),"ResNet152_avg"),(cnns.ResNet50(num_classes=10),"ResNet50"),(cnns.ResNet101(num_classes=10),"ResNet101"),(cnns.ResNet152(num_classes=10),"ResNet152")]   
    models_cifar100 = [(cnns.AlexNet(num_classes=100),"AlexNet_CryptGPU"),(cnns.AlexNet_32(num_classes=100),"AlexNet_32"),(cnns.VGG16(num_classes=100),"VGG16"),(cnns.ResNet18_avg(num_classes=100),"ResNet18_avg"),(cnns.ResNet50_avg(num_classes=100),"ResNet50_avg"),(cnns.ResNet101_avg(num_classes=100),"ResNet101_avg"),(cnns.ResNet152_avg(num_classes=100),"ResNet152_avg"),(cnns.ResNet50(num_classes=100),"ResNet50"),(cnns.ResNet101(num_classes=100),"ResNet101"),(cnns.ResNet152(num_classes=100),"ResNet152")]
    for model, model_name in models_mnist:
        #get model name
        transform = "standard"
        train_test_model(model,"MNIST", num_epochs, lr, transform, model_name, criterion, optimizer)
        transform = "custom"
        train_test_model(model,"MNIST", num_epochs, lr, transform, model_name, criterion, optimizer)
    for model, model_name in models_cifar10:
        #get model name
        transform = "standard"
        train_test_model(model,"CIFAR-10", num_epochs, lr, transform, model_name, criterion, optimizer)
        transform = "custom"
        train_test_model(model,"CIFAR-10", num_epochs, lr, transform, model_name, criterion, optimizer)
    for model, model_name in models_cifar100:
        #get model name
        model_name = model.__class__.__name__
        transform = "standard"
        train_test_model(model,"CIFAR-100", num_epochs, lr, transform, model_name, criterion, optimizer)
        transform = "custom"
        train_test_model(model,"CIFAR-100", num_epochs, lr, transform, model_name, criterion, optimizer)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a model and export it for PIGEON')
    parser.add_argument('--model', type=str, default='LeNet', help='Model to use')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--dataset_name', type=str, default='MNIST', help='Dataset name')
    parser.add_argument('--modelpath', type=str, default='./models/lenet5_mnist', help='Path to save/load the model')
    parser.add_argument('--num_epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help='Loss function')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer')

    # Add arguments for actions
    parser.add_argument('--action', type=str, choices=['train', 'import', 'train_all', 'none'], default='none', help='Action to perform on the model')
    parser.add_argument('--export_model', action='store_true', help='Export the model as a .bin file for PIGEON')
    parser.add_argument('--export_dataset', action='store_true', help='Export the test dataset as a .bin file for PIGEON')
    
    # Add argument for transformation
    parser.add_argument('--transform', type=str, choices=['custom', 'standard'], default='standard', help='Type of transformation to apply')

    args = parser.parse_args()

    # Create directories if they do not exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Dynamically create model, criterion, and optimizer
    model_class = getattr(cnns, args.model)
    model = model_class(num_classes=args.num_classes)
    
    criterion = getattr(nn, args.criterion)
    
    optimizer = getattr(optim, args.optimizer)


    # Perform the specified action
    if args.action == 'train':
        train_model(model, args.dataset_name, args.num_epochs, args.lr, args.transform, criterion, optimizer)
    elif args.action == 'import':
        load_model(model, args.modelpath)
    elif args.action == 'train_all':
        train_all(args.num_epochs, args.lr, criterion, optimizer)
    elif args.action == 'none':
        print("No action specified for the model.")

    # Export the test dataset if specified
    if args.export_dataset:
        export_test_dataset(args.dataset_name)
        print("Exported the test dataset for PIGEON.")
    
    # Export the model if specified
    if args.export_model:
        parameter_export.save_weights_compatible_with_cpp(model, args.modelpath + '.bin')
        print("Exported the model for PIGEON.")

if __name__ == '__main__':
    main()
