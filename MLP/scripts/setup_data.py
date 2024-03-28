import os
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler 
import torch
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,auc,RocCurveDisplay,classification_report
from torchmetrics.classification import Accuracy, Precision, F1Score, Recall,ROC
from pathlib import Path
from torch.optim.lr_scheduler import ExponentialLR
import random
from torch import nn
from torch.nn import Linear, Sequential,CrossEntropyLoss,Tanh
import datetime
import wandb
from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN,SMOTETomek
from torchsummary import summary
import datetime
import glob


def read_csv_data(paths:list):
    """
    Reads CSV files located at the given paths and returns a list of DataFrame objects.

    Parameters:
        paths (list): A list of file paths to the CSV files.

    Returns:
        list: A list containing DataFrame objects, each representing data from a CSV file.

    Each CSV file is assumed to have 24 columns. The last two columns are dropped,
    and the second-to-last column is renamed to 'class'. Columns are named as 'column 1',
    'column 2', and so on.
    """
    
    df_list=[]
    for path in paths:
        df=pd.read_csv(path, sep=' ')
        columns_names=[f'column {id+1}' for id in range(24)]
        df.columns=columns_names
        df=df.drop(['column 23', 'column 24'], axis=1)
        df=df.rename(columns={'column 22':'class'})
        print('DataFrame', df.head())
        df_list.append(df)
    return df_list


def balancing_dataset(option:str,df:pd.DataFrame):
    resampled_data=[]
    algorithm=''
    #One way to fight this issue is to generate new samples in the classes which are under-represented. 
    #The most naive strategy is to generate new samples by randomly sampling with replacement the current available samples. 
    if option.lower()=='naive random over-sampling':
        algorithm= RandomOverSampler(random_state=0)
    elif option.lower()=='smote':
        algorithm=SMOTE(random_state=0)
    elif option.lower()=='adasyn':
        algorithm=ADASYN(random_state=0)
    elif option.lower()=='random under sampling':
        algorithm= RandomUnderSampler(random_state=0)
    elif option.lower()=='smotetomek':
        algorithm = SMOTETomek(random_state=0)
    elif option.lower()=='smoteenn':
        algorithm = SMOTETomek(random_state=0)
    for data in df:
            X_resampled, y_resampled = algorithm.fit_resample(data.drop('class',axis=1), data['class'])
            df_resampled=pd.concat([X_resampled, y_resampled],axis=1)
            print('===============================================================================================')
            print(data['class'].value_counts())
            print(df_resampled['class'].value_counts())
            print('===============================================================================================')
            resampled_data.append(df_resampled)
    return resampled_data

def select_features(num:int,df_list:list):
    """
    Selects the top 'num' features based on correlation with the target variable from each DataFrame in the list.

    Parameters:
        num (int): The number of features to select.
        df_list (list): A list of DataFrame objects.

    Returns:
        list: A list containing DataFrame objects, each with the selected features.

    For each DataFrame in df_list, calculates the correlation matrix and selects the top 'num' features
    based on their correlation with the target variable ('class'). Drops the remaining features.
    """
        
    df_corr_list=[]
    for df in df_list:
        print('-------------------------------------------------')
        print(f'Df before selecting features {df.head()}')
        df_corr=df.corr()
        df_corr['class'].sort_values(ascending=False)[:num]
        dropped_labels=df_corr['class'].sort_values(ascending=False)[num:].index
        print('Dropped lables ', dropped_labels)
        df=df.drop(dropped_labels,axis=1)
        df_corr_list.append(df)
        print(f'Df after selecting features {df.head()}')
        print('-------------------------------------------------')
    return df_corr_list



def preprocessing_data(df_list:list):
    """
    Preprocesses the data in each DataFrame in the list by standardizing features and transforming the target variable.

    Parameters:
        df_list (list): A list of DataFrame objects.

    Returns:
        list: A list containing preprocessed DataFrame objects.

    For each DataFrame in df_list, standardizes the features using StandardScaler and transforms the target variable.
    """
    df_preprocessed_list=[]
    for df in df_list:
        print(f'Df before processing {df.head()}')
        scaler = StandardScaler() 
        standard = scaler.fit_transform(df.drop('class',axis=1)) 
        columns_names=[f'column {id+1}' for id in range(12)]
        df_standard=pd.DataFrame(standard,columns=[f'column {id+1}' for id in range(12)])
        class_df=pd.DataFrame(df['class'].values,columns=['class']).map(lambda x: x-1)
        print('Class ',class_df)
        df=pd.concat([df_standard, class_df],axis=1)
        print('Finally ',df.head())
        df_preprocessed_list.append(df)
    return df_preprocessed_list

def convert_to_tensors(df_list):
    """
    Converts each DataFrame in the list to PyTorch tensors.

    Parameters:
        df_list (list): A list of DataFrame objects.

    Returns:
        list: A list containing tuples of PyTorch tensors (X, y).

    For each DataFrame in df_list, converts features and target variable to PyTorch tensors.
    """
    tensors=[]
    for df in df_list:
        X=torch.tensor(df.drop('class',axis=1).values,dtype=torch.double)
        y=torch.tensor(df['class'].values)
        tensors.append([X,y])
    return tensors


class ThyroidGarvanDataset(Dataset):
    """
    PyTorch dataset class for Thyroid Garvan dataset.

    Parameters:
        X (Tensor): Input features.
        y (Tensor): Target labels.

    Returns:
        Tuple: A tuple containing input features and target labels.

    This class is used to create a PyTorch dataset for the Thyroid Garvan dataset.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx],self.y[idx]
    
def create_dataset(data:list):
    """
    Creates a list of ThyroidGarvanDataset objects from a list of data.

    Parameters:
        data (list): A list of tuples, where each tuple contains input features and target labels.

    Returns:
        list: A list containing ThyroidGarvanDataset objects.

    This function creates a list of PyTorch datasets from a list of data, where each dataset is created using the
    ThyroidGarvanDataset class.
    """
    datasets=[]
    for itm in data:
        dataset = ThyroidGarvanDataset(itm[0],itm[1])
        datasets.append(dataset)
    return datasets

def create_dataloder(batch_size:int,datasets:list):
    """
    Creates a list of PyTorch DataLoader objects from a list of datasets.

    Parameters:
        batch_size (int): The batch size for each DataLoader.
        datasets (list): A list of PyTorch Dataset objects.

    Returns:
        list: A list containing PyTorch DataLoader objects.

    This function creates a list of PyTorch DataLoader objects from a list of datasets.
    """
    dataloaders=[]
    shuffle=True
    for idx,dataset in enumerate(datasets):
        if idx==1:
            shuffle=False
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)    
        dataloaders.append(dataloader)
    return dataloaders