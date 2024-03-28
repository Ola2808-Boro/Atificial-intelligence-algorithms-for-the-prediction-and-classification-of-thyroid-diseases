from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,auc,RocCurveDisplay,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.classification import Accuracy, Precision, F1Score, Recall,ROC
from .utils import save_model_weights, create_experiments_dir,plot_data_distribution
from torch.optim.lr_scheduler import ExponentialLR
from .setup_data import read_csv_data,balancing_dataset,select_features,preprocessing_data,convert_to_tensors,create_dataset,create_dataloder
from .model import merged_models,MLP
import pandas as pd


def train_step(model:nn.Module,epoch:int, dataloader:DataLoader,optimizer:torch.optim.Optimizer,scheduler,loss_fn:nn.Module,accuracy,precision, recall, f1_score,roc,device:torch.device,model_name:str):
    """
    Performs one training step on the model.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        scheduler: The learning rate scheduler.
        loss_fn (nn.Module): The loss function.
        accuracy: The accuracy metric function.
        precision: The precision metric function.
        recall: The recall metric function.
        f1_score: The F1 score metric function.
        roc: The ROC curve metric function.
        device (torch.device): The device to perform operations on.
        model_name (str): The name of the model.

    Returns:
        Tuple: A tuple containing the average loss, accuracy, precision, F1 score, and recall for the current epoch.
    """
    print(f'Model train {model_name} epoch {epoch}')
    model.train()
    loss_avg=0
    acc_avg=0
    prec_avg=0
    recall_avg=0
    f1_score_avg=0
    for batch,(x,y) in enumerate(dataloader):
        #print(f'Batch {batch}')
        x,y=x.to(torch.float32),y.to(torch.long)
        y_pred=model(x).squeeze()
        y = y.squeeze()
        y_pred_class=torch.softmax(y_pred, dim=1).argmax(dim=1)
        #print(f'Pred logits {y_pred[0][0]}')
        #print(f'y:{y}, y_pred_logits: { y_pred}')
        print(f'Shape y {y.shape},y_pred:{y_pred.shape}')
        #print(f'y_pred_class {y_pred_class} y_pred_class shape:{y_pred_class.shape}')
        #print(f'To loss_fn y_pred_logits {y_pred.shape} y: {y.shape}')
        #print(f'Data type x:{x.shape}, y: {y.shape}')
        loss=loss_fn(y_pred,y)
        loss_avg=loss_avg+loss.item()
        acc=accuracy(y_pred,y)
        recall_result=recall(y_pred,y)
        prec=precision(y_pred,y)
        f1_score_result=f1_score(y_pred,y)
        acc_avg=acc_avg+acc
        prec_avg=prec_avg+prec
        recall_avg=recall_avg+recall_result
        f1_score_avg=f1_score_avg+f1_score_result
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
        #print(f'Acc: {acc} prec: {prec} recall {recall_result} f1_score_result {f1_score_result}')
    loss_avg=loss_avg/len(dataloader)
    acc_avg=acc_avg/len(dataloader)
    prec_avg=prec_avg/len(dataloader)
    f1_score_avg=f1_score_avg/len(dataloader)
    recall_avg=recall_avg/len(dataloader)
    
    
    target_names = ['normal (not hypothyroid)','hyperfunction',' subnormal functioning']
    report=classification_report(y, y_pred_class, target_names=target_names)
    #report=classification_report(y, y_pred_class, target_names=target_names,output_dict=True)
    print(report)
#     wandb.log(report)
#     wandb.log({f"conf_mat_{epoch}" : wandb.plot.confusion_matrix(
#                          y_true=y, preds=y_pred_class,
#                          class_names=target_names)})


        
#     cm=confusion_matrix(y,y_pred_class)
#     cm_df = pd.DataFrame(cm,
#                      index = ['normal (not hypothyroid)','hyperfunction',' subnormal functioning'], 
#                      columns = ['normal (not hypothyroid)','hyperfunction',' subnormal functioning'])
#     sns.heatmap(cm_df, annot=True)
#     plt.title('Confusion Matrix - test')
#     plt.ylabel('Actal Values')
#     plt.xlabel('Predicted Values')
    #print('aaa',loss_avg,acc_avg.item(),prec_avg.item(),f1_score_avg.item(),recall_avg.item())
    return loss_avg,acc_avg.item(),prec_avg.item(),f1_score_avg.item(),recall_avg.item()

def test_step(model:nn.Module,epoch:int, dataloader:DataLoader,loss_fn:nn.Module,accuracy,precision, recall, f1_score,roc,device:torch.device,model_name:str):
    """
        Performs one testing step on the model.

        Args:
            model (nn.Module): The neural network model.
            dataloader (DataLoader): The testing data loader.
            loss_fn (nn.Module): The loss function.
            accuracy: The accuracy metric function.
            precision: The precision metric function.
            recall: The recall metric function.
            f1_score: The F1 score metric function.
            roc: The ROC curve metric function.
            device (torch.device): The device to perform operations on.
            model_name (str): The name of the model.

        Returns:
            Tuple: A tuple containing the average loss, accuracy, precision, F1 score, and recall for the test set.
        """
    model.eval()
    with torch.inference_mode():
        loss_avg=0
        acc_avg=0
        prec_avg=0
        recall_avg=0
        f1_score_avg=0
        fpr_avg, tpr_avg, thresholds_avg=0,0,0
        for batch,(x,y) in enumerate(dataloader):
            #print(f'Batch {batch}')
            x,y=x.to(torch.float32),y.to(torch.long).squeeze()
            #print(f'Shape y {type(y)},y_pred:{type(y_pred)}')
            y_pred=model(x).squeeze()
            y_pred_class=torch.softmax(y_pred, dim=1).argmax(dim=1)
            #y_pred=model(x).squeeze()
            print(f'Shape y {y.shape},y_pred:{y_pred.shape}')
            #y = y.squeeze()
            #y_pred_class=torch.softmax(y_pred, dim=1).argmax(dim=1)
            loss=loss_fn(y_pred,y)
            loss_avg=loss_avg+loss.item()
            acc=accuracy(y_pred,y)
            recall_result=recall(y_pred,y)
            prec=precision(y_pred,y)
            f1_score_result=f1_score(y_pred,y)
            acc_avg=acc_avg+acc
            prec_avg=prec_avg+prec
            recall_avg=recall_avg+recall_result
            f1_score_avg=f1_score_avg+f1_score_result
            fpr, tpr, thresholds = roc(torch.softmax(y_pred,dim=1), y)
            #print(f'Roc: {fpr, tpr, thresholds}')
            #fpr_avg=fpr+fpr_avg
            #tpr_avg=tpr+tpr_avg
            #thresholds_avg=thresholds+thresholds_avg
            #print(f'Roc: {fpr, tpr, thresholds}')
#         roc_auc = auc(fpr, tpr)
#         display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
#                                  estimator_name='example estimator')
#         display.plot()
#         plt.show()
       
        target_names = ['normal (not hypothyroid)','hyperfunction',' subnormal functioning']
        print(classification_report(y, y_pred_class, target_names=target_names))
        #print(classification_report(y, y_pred_class, target_names=target_names,output_dict=True))
        cm=confusion_matrix(y,y_pred_class)
        cm_df = pd.DataFrame(cm,
                     index = ['normal (not hypothyroid)','hyperfunction',' subnormal functioning'], 
                     columns = ['normal (not hypothyroid)','hyperfunction',' subnormal functioning'])
        sns.heatmap(cm_df, annot=True)
        plt.title('Confusion Matrix - test')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')
#         wandb.log({f"conf_mat_{epoch}" : wandb.plot.confusion_matrix(
#                         y_true=y, preds=y_pred_class,
#                         class_names=target_names)})
        loss_avg=loss_avg/len(dataloader)
        acc_avg=acc_avg/len(dataloader)
        prec_avg=prec_avg/len(dataloader)
        f1_score_avg=f1_score_avg/len(dataloader)
        recall_avg=recall_avg/len(dataloader)
        #fpr_avg/=len(dataloader)
        #tpr_avh/=len(dataloader)
        #thresholds_avg/=len(dataloader)
        return loss_avg,acc_avg.item(),prec_avg.item(),f1_score_avg.item(),recall_avg.item()


def train(model:nn.Module, 
          train_dataloader:DataLoader,
          test_dataloader:DataLoader,
          epochs:int,
          optimizer:torch.optim,
          lr:float,
          model_name:str,
          dir_name:str,
         ):
    """
    Trains the model.

    Args:
        model (nn.Module): The neural network model.
        train_dataloader (DataLoader): The training data loader.
        test_dataloader (DataLoader): The testing data loader.
        epochs (int): The number of epochs to train the model for.
        optimizer (str): The name of the optimizer.
        lr (float): The learning rate.
        model_name (str): The name of the model.

    Returns:
        Tuple: A tuple containing dictionaries with training and testing results.
    """
    loss_fn = nn.CrossEntropyLoss()
    accuracy=Accuracy(task="multiclass", num_classes=3,average='weighted')
    precision= Precision(task="multiclass", num_classes=3,average='weighted')
    f1_score=F1Score(task="multiclass", num_classes=3,average='weighted')
    recall=Recall(task="multiclass", num_classes=3,average='weighted')
    roc = ROC(task="multiclass", num_classes=3)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    result_train={
      'epoch':[],
      'train_loss':[],
      'train_acc':[],
      'train_precision':[],
      'train_recall':[],
      'train_f1_score':[],

  }

    result_test={
      'test_loss':0,
      'test_acc':0,
      'test_precision':0, 
      'test_recall':0,
      'test_f1_score':0,
  }

  
    for epoch in range(int(epochs)):
#         train_precission,train_recall,train_f1_score
        train_loss, train_acc,train_precision,train_f1_score,train_recall= train_step(model=model,
                                                                                      epoch=epoch,
                                                                                      dataloader=train_dataloader,
                                                                                      loss_fn=loss_fn,
                                                                                      accuracy=accuracy,
                                                                                      precision=precision, 
                                                                                      recall=recall, 
                                                                                      f1_score=f1_score,
                                                                                      roc=roc,
                                                                                      optimizer=optimizer,
                                                                                      scheduler=scheduler, 
                                                                                      device=device,
                                                                                      model_name=model_name)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} |"
            f"train_precision: {train_precision:.4f} |"
            f"train_f1_score: {train_f1_score:.4f} |"
            f"train_recall: {train_recall:.4f} "

        )
        result_train['epoch'].append(epoch)
        result_train['train_loss'].append(round(train_loss,5))
        result_train['train_acc'].append(round(train_acc,5))
        result_train['train_recall'].append(round(train_recall,5))
        result_train['train_f1_score'].append(round(train_f1_score,5))
#test_precission,test_recall,test_f1_score
    test_loss,test_acc,test_precision,test_f1_score,test_recall=test_step(model=model,
                                                                          epoch=epoch,
                                                                          dataloader=test_dataloader,
                                                                          loss_fn=loss_fn,
                                                                          accuracy=accuracy,
                                                                          precision=precision, 
                                                                          recall=recall, 
                                                                          f1_score=f1_score,
                                                                          roc=roc, 
                                                                          device=device,
                                                                          model_name=model_name)

    print(f"test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}, test_prec: {test_precision:.4f}, test_f1_score: {test_f1_score:.4f} test_recall: {test_recall:.4f}")
    result_test['test_loss']=round(test_loss,5)
    result_test['test_acc']=round(test_acc,5)
    result_test['test_precision']=round(test_precision,5)
    result_test['test_recall']=round(test_recall,5)
    result_test['test_f1_score']=round(test_f1_score,5)
    
    for name, param in model.named_parameters():
            print('Name layer default pass to save',name)
    save_model_weights(model=model,optimizer=optimizer,epoch=epoch,loss=test_loss,dir_name=dir_name,model_name=model_name)
    
    return result_train,result_test


#Training configurations

# 1. Model architecture
# input_size int
# hidden_size int
# hidden_units int
# outputs_size int
# activation_function string 
# adaptive_lr float

# 2. Training
# epochs int

# 3. Data preprocessing
# number of features int (nie)
# balanced database bool


def train_MLP(
    input_size:int,
    hidden_size:int,
    hidden_units:int,
    output_size: int,
    optimizer_name: str,
    adaptive_lr: float,
    epochs:int,
    balanced_database:bool,
    batch_size:int,
    num_features:int,
    option:str,
    mmlp_option:dict,
    dir_name:str
    ):
    
    
    create_experiments_dir(dir_name)
    models_list=[]
    optimizer=''
    if mmlp_option['concatenation_option'].lower()=='single':
        print(f'Model architecture {input_size,hidden_units,hidden_size,output_size}')
        model = MLP(input_size,hidden_units,hidden_size,output_size,remove_output_layer=False)
        models_list.append(model)
    else:
        if mmlp_option['concatenation_option'].lower()=='parallel':
            for num in range(mmlp_option['num_of_MLP']):
                #models_list.append(MLP(input_size,hidden_units,hidden_size,output_size,remove_output_layer=True))
                models_list.append(MLP(input_size,hidden_units,hidden_size,output_size,remove_output_layer=False))
            #model=Parallel_Concatenation_MLP(models_list,mmlp_option['size'],mmlp_option['output_size'])
            #print(model)
    for idx,model in enumerate(models_list):
        if  optimizer_name.lower()=='adam':
            optimizer = torch.optim.Adam(model.parameters())
        elif  optimizer_name.lower()=='SGD':
            optimizer = torch.optim.SGD(model.parameters())
        else:
            print(f'Enter the correct name of the optimizer')


        path_train='C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/ann-train.data'
        path_test='C:/Users/olkab/Desktop/Magisterka\Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/ann-test.data'
        paths=[path_train,path_test]
        print('Read data')
        df=read_csv_data(paths)
        #plot_data_distribution(df,'Original data')
        if option!='None':
            print(f'Resampled data, method: {option}')
            resampled_data=balancing_dataset(option,df)
    #         X_train,y_train=resampled_data[0]
    #         X_test,y_test=resampled_data[1]
            df=[resampled_data[0],df[1]]
            #plot_data_distribution(df,f'Resampled data, method: {option}') 
        print('Select data')
        df_corr=select_features(num=num_features,df_list=df)
        print('Processing data')
        preprocessed_df=preprocessing_data(df_corr)
        print('Convers data to tensors')
        tensors=convert_to_tensors(preprocessed_df)#[[X,y],[X,y]]
        print('Creating datasets')
        datasets=create_dataset(tensors)
        print('Creating dataloaders')
        train_dataloader,test_dataloader=create_dataloder(batch_size=batch_size, datasets=datasets)
        result_train,result_test=train(
            model=model, 
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            epochs=epochs,
            optimizer=optimizer,
            lr=0.001,
            model_name=f'MLP_{str(idx+1)}',
            dir_name=dir_name
        )
    if mmlp_option['concatenation_option'].lower()=='parallel':
        merged_model,MLP_num=merged_models(dir_name=dir_name,model= MLP(input_size,hidden_units,hidden_size,output_size,remove_output_layer=False),optimizer=optimizer)
        #merged_model=merged_models(dir_name=dir_name,model= MLP(input_size,hidden_units,hidden_size,output_size,remove_output_layer=True),optimizer=optimizer)
        #train_dataloader,test_dataloader=create_dataloder(batch_size=batch_size, datasets=datasets)
        result_train,result_test=train(
            model=merged_model, 
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            epochs=epochs,
            optimizer=optimizer,
            lr=0.001,
            model_name=f'MLP_merged',
            dir_name=dir_name
        )