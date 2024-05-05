import kagglehub
import torch
import logging
from data_setup import setup_data,tirads
import datetime
import sys
from torch import nn
from torchvision.models import resnet18
sys.path.insert(1, 'C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/scripts')
from engine import create_experiments_dir, train
from utils import save_results,save_model_weights

logging.basicConfig(level=logging.INFO,filename='trained_models.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')
BASE_DIR='C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/Deep Neural Networks'

# kagglehub.login() 
# # Download latest version
# path = kagglehub.model_download("wilsonbpegros/inception-based-tumor-classifier/pyTorch/testv19")

# print("Path to model files:", path)

path='C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/Deep Neural Networks/scripts/kaggle models/InceptionV4_model_19.pt'


def train_model(path:str,epochs:int,lr:float):
    target_names=list(tirads.keys())
    model=load_model(path)
    output_size=len(tirads.keys())
    dir_name=f'experiments_trained_models/{ datetime.datetime.now().strftime("%H-%M-%d-%m-%Y")}'
    adaptive_lr=True
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    train_dataloader,test_dataloader=setup_data(BATCH_SIZE=32)
    path_experiments=create_experiments_dir(dir_name=dir_name,BASE_DIR=BASE_DIR)
    result_train,result_test=train(model=model,
          train_dataloader=train_dataloader,
          test_dataloader=test_dataloader,
          epochs=epochs,
          optimizer=optimizer,
          adaptive_lr=adaptive_lr,
          model_name=f'resnet18',
          dir_name=dir_name,
          class_num=output_size,
          target_name=target_names,
          BASE_DIR=BASE_DIR,
          
          )

    save_results(train_result=result_train,
                 test_result=result_test,
                 model_name='resnet18',
                 path=path_experiments
                 )

def load_model(path:str):
    model=resnet18(weights='DEFAULT')
    # model=torch.load(path,map_location='cpu')
    # print(type(model))

    for name,param in model.named_parameters():
        if 'fc' not in name or 'layer4' not in name or 'layer3' not in name:
            param.requires_grad = False
    model.fc=nn.Linear(512, len(tirads.keys()))
    # for name, param in model.named_parameters():
    #     print(name)
    #     logging.info(f'Name layer default save {name},params {param.shape}')
    # for name,param in model.named_parameters():
    #     if 'fc' not in name or 'layer4' not in name:
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True
    return model


# load_model(path=path)
# print( len(tirads))

train_model(path=path,epochs=200,lr=0.01)