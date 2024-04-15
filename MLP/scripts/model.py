
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn import Linear, Sequential,CrossEntropyLoss,Tanh
import glob
from .utils import load_model_weights
import logging

logging.basicConfig(level=logging.INFO,filename='MLP.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')


class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) neural network model.

    Parameters:
        input_size (int): The size of the input features.
        hidden_layers (int): The number of hidden layers in the MLP.
        hidden_units (int): The number of units in each hidden layer.
        output_size (int): The size of the output layer.

    Returns:
        Tensor: The output tensor from the MLP model.

    This class defines a simple MLP neural network model with customizable input size,
    number of hidden layers, number of units in each hidden layer, and output size.
    """
    def __init__(self,input_size:int, hidden_layers:int,hidden_units:int,output_size:int,activation_function:torch.nn.Module,remove_output_layer:bool):
        super(MLP, self).__init__()
        layers = nn.ModuleList()
        layers.append(
             Linear(input_size, hidden_units)
        )
        layers.append(
                activation_function
                )
        for num_layer in range(hidden_layers):
            layers.append(
                Linear(hidden_units, hidden_units)
                )
            layers.append(
                activation_function
                )
        if remove_output_layer==False:
            layers.append(
                    Linear(hidden_units,output_size)
                    )
        # else:
        #     self.layers.pop(key=-1)
            #TODO: usunac chyba Tanh() 
        self.enc_red = nn.Sequential(*layers)  
    def forward(self, x):
        """
        Initializes the MLP neural network model.

        Args:
            input_size (int): The size of the input features.
            hidden_layers (int): The number of hidden layers in the MLP.
            hidden_units (int): The number of units in each hidden layer.
            output_size (int): The size of the output layer.
        """
            
        return self.enc_red(x)



class Parallel_Concatenation_MLP(nn.Module):
   
    def __init__(self,models_list:list,size:int,output_size:int):
        super(Parallel_Concatenation_MLP, self).__init__()
        self.MLP_list=models_list
        self.outputs=[]
        self.size=size
        self.output_size=output_size
        self.output=Linear(self.output_size*self.size, self.output_size)
    def forward(self, x):
        logging.info(f'Params input size {self.output_size* self.size}, num of MLP: {len(self.MLP_list)}')
        for idx,model in enumerate(self.MLP_list):
            logging.info(f'Model MLP {idx} architecture {model}, output shape: {model(x).shape}')
            for param in model.parameters():
                param.requires_grad = False
            self.outputs.append(model(x))
        logging.info(f'Shapes from MLP {[output.shape for output in self.outputs]}',)
        concatenated_output = torch.cat(self.outputs, dim=1)
        self.outputs=[]
        logging.info(f'Shape after concatenating {concatenated_output.shape}')
        return self.output(concatenated_output)
    

# class Stacking_Concatenation_MLP(nn.Module):
   
#     def __init__(self,models_list:list,size:int,output_size:int):
#         super(Parallel_Concatenation_MLP, self).__init__()
#         self.MLP_list=models_list
#         self.outputs=[]
#         self.size=size
#         self.output_size=output_size
#         self.output=Linear(self.output_size*self.size, self.output_size)
#     def forward(self, x):
#         logging.info(f'Params input size {self.output_size* self.size}, num of MLP: {len(self.MLP_list)}')
#         for idx,model in enumerate(self.MLP_list):
#             logging.info(f'Model MLP {idx} architecture {model}, output shape: {model(x).shape}')
#             for param in model.parameters():
#                 param.requires_grad = False
#             self.outputs.append(model(x))
#         logging.info(f'Shapes from MLP {[output.shape for output in self.outputs]}',)
#         concatenated_output = torch.cat(self.outputs, dim=1)
#         self.outputs=[]
#         logging.info(f'Shape after concatenating {concatenated_output.shape}')
#         return self.output(concatenated_output)


def merged_models(dir_name,model:nn.Module,optimizer:torch.optim):
    BASE_DIR='C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/'
    files_pt=glob.glob(f'{BASE_DIR}/{dir_name}/*.pt')
    files_pth=(glob.glob(f'{BASE_DIR}/{dir_name}/*.pth'))
    files=files_pt+files_pth
    files_clean=[file_itm.replace('\\','/').replace('//','/') for file_itm in files]
    logging.info(f'Files with weights to merge :{files_clean}')
    
    models=[]
    for file_itm in files_clean:
        for name, param in model.named_parameters():
            logging.info(f'Model Name layer default {name}')
        model_merged,optimizer,epoch,loss=load_model_weights(model,file_itm,optimizer)
        for param in model_merged.parameters():
            param.requires_grad = False
        # for name, param in model_merged.named_parameters():
        #     print('Name layer',name)
        models.append(model_merged)  
    logging.info(f'Models {models}')       
    mmlp_model=Parallel_Concatenation_MLP(models_list=models,size=len(models),output_size=3)
    logging.info(f'Model architecture after concatenating {mmlp_model}')
    return mmlp_model,len(models)


