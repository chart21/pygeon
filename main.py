#main function
import cnns
import data_load
import misc
import train
import parameter_export
import torch
import numpy as np
import os
from torch import nn
from torch import optim
# from deepreduce_models.resnet import *

def save_model(model,filepath):
    parameter_export.save_weights_compatible_with_cpp(model, filepath+'.bin')

def export_pth_model(model,filepath):
    torch.save(model.state_dict(), filepath+'.pth')

def train_model(model,dataset_name,num_epochs,lr, transform = "standard"):
    train_loader, test_loader,num_classes = data_load.load_dataset(dataset_name, transform)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train.train_and_evaluate(model, train_loader, test_loader, num_epochs)

def train_test_model(model,dataset_name,num_epochs,lr, transform = "standard", model_name="LeNet"):
    train_loader, test_loader,num_classes = data_load.load_dataset(dataset_name, transform)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train.train_test(model, train_loader, test_loader, num_epochs, model_name, dataset_name, transform)

def load_model(model,filepath):
    model.load_state_dict(torch.load(filepath+'.pth', map_location=torch.device('cpu')))

def load_checkpoint(model,filepath):
    checkpoint = torch.load(filepath+'.pth.tar', map_location=torch.device('cpu'))
    # Assuming the model is wrapped in DataParallel and saved in 'snet' key
    model.load_state_dict({k.replace('module.',''): v for k, v in checkpoint['snet'].items()})


def test_model(model,dataset_name):
    train_loader, test_loader,num_classes = data_load.load_dataset(dataset_name)
    train.evaluate(model, test_loader)

def export_test_dataset(dataset_name):
    train_set, test_set,num_classes = data_load.load_dataset(dataset_name)
    data_load.export_dataset(test_set,'./data/'+dataset_name+'_test_images.bin','./data/'+dataset_name+'_test_labels.bin')

def train_all(num_epochs,lr):
    models_mnist = [(cnns.LeNet(num_classes=10),"LeNet")]
    models_cifar10 = [(cnns.AlexNet(num_classes=10),"AlexNet_CryptGPU"),(cnns.AlexNet_32(num_classes=10),"AlexNet_32"),(cnns.VGG16(num_classes=10),"VGG16"),(cnns.ResNet18_avg(num_classes=10),"ResNet18_avg"),(cnns.ResNet50_avg(num_classes=10),"ResNet50_avg"),(cnns.ResNet101_avg(num_classes=10),"ResNet101_avg"),(cnns.ResNet152_avg(num_classes=10),"ResNet152_avg"),(cnns.ResNet50(num_classes=10),"ResNet50"),(cnns.ResNet101(num_classes=10),"ResNet101"),(cnns.ResNet152(num_classes=10),"ResNet152")]   
    models_cifar100 = [(cnns.AlexNet(num_classes=100),"AlexNet_CryptGPU"),(cnns.AlexNet_32(num_classes=100),"AlexNet_32"),(cnns.VGG16(num_classes=100),"VGG16"),(cnns.ResNet18_avg(num_classes=100),"ResNet18_avg"),(cnns.ResNet50_avg(num_classes=100),"ResNet50_avg"),(cnns.ResNet101_avg(num_classes=100),"ResNet101_avg"),(cnns.ResNet152_avg(num_classes=100),"ResNet152_avg"),(cnns.ResNet50(num_classes=100),"ResNet50"),(cnns.ResNet101(num_classes=100),"ResNet101"),(cnns.ResNet152(num_classes=100),"ResNet152")]
    # models_mnist = [cnns.LeNet(num_classes=10)]
    # models_cifar10 = [cnns.AlexNet(num_classes=10),cnns.AlexNet_32(num_classes=10),cnns.VGG16(num_classes=10),cnns.ResNet18_avg(num_classes=10),cnns.ResNet50_avg(num_classes=10),cnns.ResNet101_avg(num_classes=10),cnns.ResNet152_avg(num_classes=10)]
    # models_cifar100 = [cnns.AlexNet(num_classes=100),cnns.AlexNet_32(num_classes=100),cnns.VGG16(num_classes=100),cnns.ResNet18_avg(num_classes=100),cnns.ResNet50_avg(num_classes=100),cnns.ResNet101_avg(num_classes=100),cnns.ResNet152_avg(num_classes=100)]
    for model, model_name in models_mnist:
        #get model name
        transform = "standard"
        train_test_model(model,"MNIST", num_epochs, lr, transform, model_name)
        transform = "custom"
        train_test_model(model,"MNIST", num_epochs, lr, transform, model_name)
    for model, model_name in models_cifar10:
        #get model name
        transform = "standard"
        train_test_model(model,"CIFAR-10", num_epochs, lr, transform, model_name)
        transform = "custom"
        train_test_model(model,"CIFAR-10", num_epochs, lr, transform, model_name)
    for model, model_name in models_cifar100:
        #get model name
        model_name = model.__class__.__name__
        transform = "standard"
        train_test_model(model,"CIFAR-100", num_epochs, lr, transform, model_name)
        transform = "custom"
        train_test_model(model,"CIFAR-100", num_epochs, lr, transform, model_name)
    
def main():
    #if directory models does not exist, create it
    os.makedirs('models', exist_ok=True)
    
    train_all(2,0.01)
    #model = LeNet5(num_classes=10) # replace with Qunatized LeNet
    # dataset_name = 'MNIST'
    # modelpath = './models/lenet5_mnist'
    # num_epochs = 80
    # lr = 0.01
    # train_model(model,dataset_name,num_epochs,lr) 
    # parameter_export.save_weights_compatible_with_cpp(model, modelpath+'.bin')
    #export_test_dataset(dataset_name) 
    # parameter_export.save_quantized_weights_compatible_with_cpp(model, modelpath+'.bin') 
    # parameter_export.save_quantization_params(model, modelpath+'bin_quant')
    # train_loader, test_loader,num_classes = data_load.load_dataset(dataset_name) # replace with qunatized dataset loader
    # data_load.export_quantized_dataset(test_loader,'./data/'+dataset_name+'_test_images.bin','./data/'+dataset_name+'_test_labels.bin') # export qunatized dataset



    # model = DRD_C100_230K(num_classes=100)  # or any other model you have
    # model = DRD_C100_115K(num_classes=100)  # or any other model you have
    # model = cnns.ResNet50(num_classes=100)
    # dataset_name = 'CIFAR-100'
    # modelpath = './deepreduce_models/CIFAR100_models/model_DRD_C100_115K'
    # num_epochs = 20
    # lr = 0.001

    # load_model(model,modelpath)
    # load_model(model,'./models/resnet50_cifar100')
    # save_model(model,'./deepreduce_models/CIFAR100_models/model_DRD_C100_230K_mod')
    # load_checkpoint(model,modelpath)
    # export_pth_model(model,modelpath)

    # parameter_export.write_params(model,'./deepreduce_models/CIFAR100_models/model_DRD_C100_230K.bin')
    # print(model)

    # parameter_export.read_params(model,'./deepreduce_models/CIFAR100_models/model_DRD_C100_230K.bin')
    # parameter_export.import_weights_compatible_with_cpp(model,'./deepreduce_models/CIFAR100_models/model_DRD_C100_230K.bin')
    # test_model(model,dataset_name)
    # misc.print_layers_and_params(model)
    # print(model)
    # save_model(model,modelpath)
    # test_model(model,dataset_name)
    # export_test_dataset(dataset_name)


if __name__ == '__main__':
    main()

