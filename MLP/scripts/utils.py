import matplotlib.pyplot as plt
from torch import nn
import torch
import seaborn as sns
import os

def plot_data_distribution(df_list:list,title:str):
    """
    Plots the distribution of classes in each DataFrame in the list.

    Parameters:
        df_list (list): A list of DataFrame objects.

    This function iterates through each DataFrame in df_list, calculates the class distribution,
    and plots it as a bar chart.
    """
    for df in df_list:
        print('Train data distribution',df['class'].value_counts())
        df_disctribution=df['class'].value_counts().sort_values(ascending=True)
        data={}
        for idx in df_disctribution.index:
            data.update({f'{idx}':df_disctribution[idx]})
        print(data)
        labels=list(data.keys())
        values=list(data.values())
        print(type(labels),type(values),labels,values)
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
        
def save_model_weights(model:nn.Module,optimizer:torch.optim,epoch:int,loss:int,dir_name:str,model_name:str):
    path_dir=create_experiments_dir(dir_name=dir_name)
    path=path_dir+'/'+model_name+'.pth'
    print(f'Path to save model {path}, model {model_name}')
    for name, param in model.named_parameters():
            print('Name layer default save',name)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)
    


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
    
    return model,optimizer,epoch,loss

def create_experiments_dir(dir_name:str):
    
    BASE_DIR='C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP'
    path=BASE_DIR+'/'+str(dir_name).replace("\\","/")
    print('Path',path)
    try:
        os.mkdir(path)
        
    except OSError as error:
        print(error)

    return path