import pandas as pd
import os
import glob
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image, ImageDraw
import re
import xml.etree.ElementTree as ET
import logging
import json
import codecs
import torch
from torch.utils.data import Dataset, DataLoader,Subset
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import DataLoader,random_split
from torchvision.transforms import Compose,ToTensor,Resize,Normalize

logging.basicConfig(level=logging.INFO,filename='data_setup.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
PATH_XML_FILES='C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/DDTI Thyroid Ultrasound Images/*.xml'
PATH_IMAGES='C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/DDTI Thyroid Ultrasound Images/*.jpg'

class DDTIThyroidUltrasoundImagesDataset(Dataset):
    def __init__(self, X, y,transform):
        self.X = X
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return self.transform(self.X[idx]),self.y[idx]


tirads={
    '5' :5,
    '4a':4,
    '4b':3,   
    '4c':2, 
    '3' :1,   
    '2' :0        
}
data={
    'xml':{
        'num':0,
        'paths':[]
    },
    'img':{
        'num':0,
        'paths':[]
    }
}

# patient_data={
#     'number':0,
#     'age':0,
#     'sex':0,
#     'composition':0,
#     'echogenicity':0,
#     'margins':0,
#     'calcifications':0,
#     'tirads':0,
#     'reportbacaf':0,
#     'reporteco':0,
#     'mark':[],
#     'images_path':[],
#     'masks_path':[],
    
    
# }
def count_data(paths:str|list[str])->dict:
    for path in paths:
        files_path=glob.glob(path)
        if 'xml' in path:
            data['xml']['num']=len(files_path)
            data['xml']['paths']=files_path
        elif 'jpg' in path:
            data['img']['num']=len(files_path)
            data['img']['paths']=files_path
        else:
            logging.warning(f'Not expected data type {path}')

    logging.info(f"Num of xml files : {data['xml']['num']}, num of jpg files: {data['img']['num']}")
    return data


def generate_dataset(data):


    xml_files=data['xml']['paths']
    img_files=data['img']['paths']
    for xml_file in xml_files:
        patient_data={}
        #logging.info(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        data_svg=[]
        for child in root:
            logging.info(f'{child.tag}:{child.text}')
            if child.tag=='mark':
                images_num=[]
                svg=[]
                #data=[]
                for itm in child:
                    if itm.tag=='image':
                        images_num.append(itm.text)
                        #print(xml_file,images_num)
                    else:
                        svg.append(itm.text)
                data_svg.append([{'num':images_num[idx],'svg':svg[idx]} for idx in range(len(images_num))])
                #print(data)
                #print('----------------------------------------------')
                #patient_data.update({child.tag:data})
            else:      
                patient_data.update({child.tag:child.text})
        patient_data.update({child.tag:data_svg})
        img_path_regexp = re.findall(r'\b\d+',xml_file)[0]
        logging.info(f"Regex {img_path_regexp}, patient number{patient_data['number']}_")
        #img_paths_patient=[ codecs.decode(img_file,'unicode_escape')for img_file in img_files if patient_data['number'] == re.findall(r'\b\d+',img_file)[0]]
        img_paths_patient=[img_file.replace('/','\\') for img_file in img_files if patient_data['number'] == re.findall(r'\b\d+',img_file)[0]]
        mask_paths_patient=[img_file.split('.jpg')[0]+'_masks.jpg' for img_file in img_paths_patient]
        patient_data.update({'masks_path':mask_paths_patient})
        patient_data.update({'images_path':img_paths_patient})
        logging.info(f'Adding data {patient_data}')
        json_object = json.dumps(patient_data)
        with open('patient_data.json','a') as f:
                f.write(f'{json_object} \n')


def removal_artefacts(img_path:str)->Image:
    img=Image.open(fp=img_path)
    print(f'Image size {img.size}')
    (left, upper, right, lower) = (120, 8, 430, 320)
    img_crop = img.crop((left, upper, right, lower))
    img_crop.show()



def create_masks(data_frame:pd.DataFrame):

    for idx in data_frame.index:
        images_path=data_frame['images_path'][idx]
        masks_path=data_frame['masks_path'][idx]
        logging.info(f'Index {idx}')
        for index,img_path in enumerate(images_path):
            logging.info(img_path)
            for mask in data_frame['mark'][idx]:
                regex=re.findall(r'_\d+',img_path)[0]
                logging.info(f'Tag {mask}')
                if mask[0]['num']==regex.replace('_',''):
                    print(f"img {img_path}, regex {regex}")
                    img=Image.open(img_path)
                    img_size=img.size
                    new_img=Image.new(mode='1',size=img_size)
                    #new_img.show()
                    draw = ImageDraw.Draw(new_img)
                    points=mask[0]['svg']
                    logging.info(points)
                    x_coordinates=[int(itm) for itm in re.findall(r'"x":\s*(\d+)',points)]
                    y_coordinates=[int(itm) for itm in re.findall(r'"y":\s*(\d+)',points)]
                    coordinates=list(zip(x_coordinates,y_coordinates))
                    logging.info(f"Coordinates num :{mask[0]['num']} {coordinates}")
                    draw.polygon(coordinates, fill='white',outline='white')
                    #new_img.save(masks_path[idx])
                    new_img.show()
           



def analyze_data(data_frame:pd.DataFrame):
    logging.info(f'Nan values in data frame {data_frame.isna().sum().sum()}')
    for item in data.columns:
        logging.info(f'Number of Nan values in column {item} {data_frame[item].isna().sum().sum()}')


def load_data()->pd.DataFrame:
    df_read_json = pd.read_json('patient_data.json', lines=True)
    logging.info("DataFrame using pd.read_json() method:")
    logging.info(df_read_json)
    return df_read_json

def create_dataset(data:pd.DataFrame):
    transform_basic=Compose([
        ToTensor(),
        Resize((256,256)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data=[{'class':tirads[data['tirads'][idx]],'img_path':data['images_path'][idx]} for idx in data['tirads'].index if data['tirads'][idx]!=None]

    x_data=[]
    y_data=[]
    for itm in data:
        x_data.append(Image.open(itm['img_path'][0]))
        y_data.append(torch.tensor(itm['class']))
    # for itm in data:
    datasets = DDTIThyroidUltrasoundImagesDataset(x_data,y_data,transform_basic)
    #     datasets.append(dataset)
    return datasets


def split_datset(dataset:list)->list[Subset]:
    generator1 = torch.Generator().manual_seed(42)
    train_set,test_set=random_split(dataset, [0.8, 0.2], generator=generator1)
    logging.info(f'Train set {len(train_set)} test set {len(test_set)}')
    return train_set,test_set

def create_dataloaders(train_set:Subset,test_set:Subset,BATCH_SIZE:int)->list[DataLoader]:
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_dataloader,test_dataloader


def setup_data(BATCH_SIZE:int)->list[DataLoader]:
    df=load_data()
    dataset=create_dataset(df)
    train_set,test_set=split_datset(dataset=dataset)
    train_dataloader,test_dataloader=create_dataloaders(train_set=train_set,test_set=test_set,BATCH_SIZE=BATCH_SIZE)
    return  train_dataloader,test_dataloader


# df=load_data()
# dataset=create_dataset(df)
# print(dataset.__getitem__(0))
# print(type(dataset))
# # train_dataloader,test_dataloader=setup_data(BATCH_SIZE=32)
# print(type(train_dataloader))
# for batch,(X,y) in enumerate(train_dataloader):
#     print(type(X),type(y))
# paths=[PATH_XML_FILES,PATH_IMAGES]
# count_data(paths=paths)
# generate_dataset(data)
# df=load_data()
# print(df.head())
# dataset=create_dataset(df)

# print(dataset[0].__len__())
# logging.info(df.info())
#create_masks(data_frame=df)
# path='C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/DDTI Thyroid Ultrasound Images/1_1.jpg'
# removal_artefacts(path)