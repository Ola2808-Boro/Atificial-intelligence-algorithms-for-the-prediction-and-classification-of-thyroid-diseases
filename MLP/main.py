from scripts.engine import train_MLP
import datetime


#Project
project_name="MLP"

#Model architecture
input_size=12
hidden_size=8
hidden_units=1
output_size=3
activation="ReLU"

#Model architecture MMLP 
mmlp_option={
    'concatenation_option':'parallel',  #Sequential Concatenation,Parallel Concatenation,Model Stacking, Single MLP
    #'concatenation_option':'single',  #Sequential Concatenation,Parallel Concatenation,Model Stacking, Single MLP
    'num_of_MLP':2,
    'size':hidden_size,
    'output_size':output_size
}

#Training
optimizer='adam'
adaptive_lr= 0.01
epochs=5

#Datasets
balanced_database=True
batch_size=512
num_features=13
option='SMOTETomek'
#Oversample methods: [Naive random over-sampling,SMOTE,ADASYN]
#Undersample methods: [Random under-sampling]
#Combination of over- and under-sampling [SMOTETomek,SMOTEENN]

#Logs
dir_name=f'experiments/{ datetime.datetime.now().strftime("%H-%M-%d-%m-%Y")}'

# wandb.init(
#     # set the wandb project where this run will be logged
#     project=project_name,
#     name='run 1',
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate":adaptive_lr,
#     "architecture": f"MLP({input_size},{hidden_size},{hidden_units},{output_size})",
#     "dataset": f"Garvan Institute - features {num_features-1}",
#     "epochs": epochs,
#     "activation": activation,
#     "num_classes": output_size,
#     }
# )


train_MLP(
    input_size=input_size,
    hidden_size=hidden_size,
    hidden_units=hidden_units,
    output_size=output_size,
    optimizer_name=optimizer,
    adaptive_lr= adaptive_lr,
    epochs=epochs,
    balanced_database=balanced_database,
    batch_size=batch_size,
    num_features=num_features,
    option=option,
    mmlp_option=mmlp_option,
    dir_name=dir_name
    )

# wandb.finish()