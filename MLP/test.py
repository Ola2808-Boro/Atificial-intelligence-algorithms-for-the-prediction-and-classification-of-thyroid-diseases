import torch
from scripts.utils import load_model_weights
from scripts.model import MLP
import glob

# path='C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/experiments/26-28-23-25-03-2024/MLP_1.pth'

# # input_size,hidden_units,hidden_size,output_size,remove_output_layer=False
# # input_size=12
# # hidden_size=8
# # hidden_units=1
# # output_size=3
# #  (12, 1, 8, 3)
# model=MLP(12,1,8,3,False)
# optimizer = torch.optim.Adam(model.parameters())
# model,optimizer,epoch,loss=load_model_weights(model=model,path=path,optimizer=optimizer)
# print(f'MOdel {model} optim {optimizer} epoch {epoch} loss {loss}')
print(glob.glob('C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/MLP/experiments/12-06-26-03-2024/*.pth'))