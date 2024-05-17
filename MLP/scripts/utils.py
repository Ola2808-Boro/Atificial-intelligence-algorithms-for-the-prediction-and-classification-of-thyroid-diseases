import matplotlib.pyplot as plt
from torch import nn
import torch
import seaborn as sns
import os
import logging
import json
from torch.optim.lr_scheduler import LambdaLR,StepLR,ConstantLR,LinearLR,ExponentialLR,ReduceLROnPlateau
import cv2
import numpy as np
from scipy.ndimage import convolve
from pydicom import dcmread
import sys


logging.basicConfig(level=logging.INFO,filename='MLP.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')



class Sin(nn.Module):
    def __init__(self):
        pass
    def forward(self,x):
        return torch.sin(x)
    

class Logarithmic(nn.Module):
    def __init__(self):
        pass
    def forward(self,x):
        if x>=0:
            return torch.log(x+1)
        else:
            return -torch.log(-x+1)
    
#TODO:
#check Neural func 
class Neural(nn.Module):
    def __init__(self):
        pass
    def forward(self,x):
        return (1/1+torch.exp(-torch.sin(x)))




def define_activation_function(activation_function:str)->nn.Module:
    logging.info(f'Passed activation function {activation_function.lower()}')
    if activation_function.lower()=='sigmoid':#sigmoidalna
        activation_function=nn.Sigmoid()
    elif activation_function.lower()=='tanh': #tangens hiperboliczny
        activation_function=nn.Tanh() 
    elif activation_function.lower()=='neural': #neuronalna
        activation_function=Neural()
    elif activation_function.lower()=='exponential': #wykładnicza
        activation_function=nn.ELU()
    elif activation_function.lower()=='log':#logarytmiczna
        activation_function=Logarithmic()
    elif activation_function.lower()=='sin': #sinusoidalna
        activation_function=Sin()
    else:
         logging.warning('Default activation function')
         activation_function=nn.Tanh()
    logging.info(f'Returned activation function {activation_function}')
    return activation_function


def define_scheduler(scheduler:str,optimizer:torch.optim)->nn.Module:
    logging.info(f'Passed activation function {activation_function.lower()}')
    if activation_function.lower()=='lambdalr':#sigmoidalna
        lambda1 = lambda epoch: 0.95 ** epoch
        scheduler=LambdaLR(optimizer=optimizer,lr_lambda=lambda1)
    elif activation_function.lower()=='steplr': #tangens hiperboliczny
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif activation_function.lower()=='constantlr': #neuronalna
        activation_function=ConstantLR(optimizer=optimizer,factor=0.5)
    elif activation_function.lower()=='exponential': #wykładnicza
        activation_function=nn.LinearLR(optimizer=optimizer)
    elif activation_function.lower()=='chainedscheduler':#logarytmiczna
        activation_function=ExponentialLR(optimizer=optimizer, gamma=0.1)
    elif activation_function.lower()=='reducelronplateau': #sinusoidalna
        activation_function=ReduceLROnPlateau(optimizer=optimizer,mode='min')#TODO scheduler.step(metrics)
    else:
         logging.warning('Default activation function')
         activation_function=nn.Tanh()
    logging.info(f'Returned scheduler {scheduler}')
    return scheduler




def plot_data_distribution(df_list:list,title:str):
    """
    Plots the distribution of classes in each DataFrame in the list.

    Parameters:
        df_list (list): A list of DataFrame objects.

    This function iterates through each DataFrame in df_list, calculates the class distribution,
    and plots it as a bar chart.
    """
    for df in df_list:
        logging.info(f'Train data distribution: {df["class"].value_counts()}')
        df_disctribution=df['class'].value_counts().sort_values(ascending=True)
        data={}
        for idx in df_disctribution.index:
            data.update({f'{idx}':df_disctribution[idx]})
        labels=list(data.keys())
        values=list(data.values())
        fig = plt.figure(figsize = (10, 5))
        plt.bar(labels,values,color ='maroon', 
        width = 0.4)
        plt.title(title)
        plt.show()


def plot_correlation(df):
    """
    Plots a heatmap showing the correlation matrix of the given DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.

    Returns:
    - None
    ```
    """
    Corr_Matrix = round(df.corr(),2)
    axis_corr = sns.heatmap(
    Corr_Matrix,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(50, 500, n=500),
    square=True
    )

    plt.show()
        
def save_model_weights(model:nn.Module,optimizer:torch.optim,epoch:int,loss:int,dir_name:str,model_name:str,BASE_DIR:str):
    path_dir=create_experiments_dir(dir_name=dir_name,BASE_DIR=BASE_DIR)
    path=path_dir+'/'+model_name+'.pth'
    logging.info(f'Path to save model {path}, model {model_name}')
    for name, param in model.named_parameters():
            logging.info(f'Name layer default save {name}')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)

def plot_charts(train_result:dict, test_result:dict, model_name:str, dir_name:str,BASE_DIR:str):
    
    epoch=train_result['epoch']
    for idx,data in enumerate(train_result):
        if idx!=0:
            x=epoch
            y=train_result[idx]
            plt.plot(x,y)
            plt.title(f'Model {model_name} metrics: {train_result.keys()[idx]}')
            plt.xlabel('Epochs')
            plt.xlabel(f'{train_result.keys()[idx].capitalize()}')
            path=BASE_DIR+'/'+str(dir_name).replace("\\","/")
            plt.savefig(f'{path}/{model_name}_{train_result.keys()[idx]}.png')
    

def save_results(train_result:dict, test_result:dict, model_name:str, path:str):
    logging.info(f'Path to save results {path}')
    train_data=json.dumps(train_result,indent=6)
    test_data=json.dumps(test_result,indent=6)
    data=[{
        'name':'train',
        'data':train_data
    },
    {
        'name':'test',
        'data':test_data
    }]
    logging.info(f'Results train: {train_data}')
    logging.info(f'Results test: {test_data}')

    for itm in data:
        with open(f"{path}/{model_name}_{itm['name']}.json", "w") as file:
            file.write(itm['data'])


def save_model(model:nn.Module,path:str):
    torch.save(model,path)



def load_model(path:str):
    model=torch.load(path)
    return model


def load_model_weights(model:nn.Module,path:str,optimizer:torch.optim):
    checkpoint=torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logging.info(f'Loading weights')
    
    return model,optimizer,epoch,loss

def create_experiments_dir(dir_name:str,BASE_DIR:str):
    
    path=BASE_DIR+'/'+str(dir_name).replace("\\","/")
    print('Path',path)
    try:
        os.mkdir(path)
        
    except OSError as error:
        print(error)

    return path


