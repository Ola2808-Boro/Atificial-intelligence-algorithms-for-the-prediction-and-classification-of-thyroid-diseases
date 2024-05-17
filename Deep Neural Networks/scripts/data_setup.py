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
from skimage.io import imread, imsave
from pydicom import dcmread,multival
from pydicom.data import get_testdata_file
import matplotlib.pyplot as plt
import time
from pathlib import Path
from scipy.ndimage import convolve
from pydicom import dcmread
import sys
import cv2
from Criminisi_algorithm import CriminisiAlgorithm
from skimage.measure import label

BASE_DIR_DATBASE='C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/'
logging.basicConfig(level=logging.INFO,filename='trained_models.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
PATH_XML_FILES=f'{BASE_DIR_DATBASE}DDTI Thyroid Ultrasound Images/*.xml'
PATH_IMAGES=f'{BASE_DIR_DATBASE}DDTI Thyroid Ultrasound Images/*.jpg'
PATH_DICOM_IMAGES=f'{BASE_DIR_DATBASE}GUMED'
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

def read_data():

    dicom_files=[path.replace('\\','/')  for path in glob.glob(f'{PATH_DICOM_IMAGES}/*') if '.csv' not in path and '.jpg' not in path and '.png' not in path and '.jpeg' not in path]
    dicom_files+=[path.replace('\\','/') for path in glob.glob(f'{PATH_DICOM_IMAGES}/*/*') if  '.csv' not in path and '.jpg' not in path and '.png' not in path and '.jpeg' not in path]
    dicom_attributes=set()
    meta_data=[]
    for path in dicom_files:
        if not Path(path).is_dir():
            dicom = dcmread(path)
            dicom_attributes.update(dicom.dir())
    dicom_attributes = list(dicom_attributes)
    dicom_attributes.remove('PixelData')
    dicom_attributes.remove('PatientName')
    print(dicom_attributes)
    


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

def plot_img_hist(img_path:str):
    dicom = dcmread(img_path)
    print(type(dicom))
    # img = plt.imread(img_path)
    plt.hist(dicom.pixel_array.ravel(),256,[0,256]) 
    plt.show() 

def component_analysis(img_path,edges):
    print(type(edges))
    print('Edges 2',edges.astype('uint8').shape)
    labeled_vol, num_features = label(edges, connectivity=1, return_num=True)
    labeled_vol[labeled_vol>1]=1
    #print(np.unique(edges.astype('int8')))
    #ret0, labelss0 = cv2.connectedComponents(edges.astype('uint8'))
    #labelss0[labelss0[labelss0>1]]=1
    #print(labelss0[labelss0>1])
    #print(np.unique(edges.astype('int8')))
    #ret1, labelss1 = cv2.connectedComponents(edges[:,:,1].astype('uint8'))
    #ret2, labelss2= cv2.connectedComponents(edges[:,:,2].astype('uint8'))
    #print(ret0,labelss0)
    #print(ret1,labelss1)
    #print(ret2,labelss2)
    plot_image(img_path,labeled_vol,title=['Original image','Component analysis'])
    return labeled_vol
    #plot_image(img_path,labelss1,title=['Original image','Component analysis'])
    #plot_image(img_path,labelss2,title=['Original image','Component analysis'])
    #return [labelss0,labelss1,labelss2]
 

def plot_image(img_path:str,img_array:np.array,title:list[str]):

    plt.figure(figsize=(10, 5))
    # Original Image
    plt.subplot(1, 2, 1)
    #Display the original image using matplotlib
    plt.imshow(dcmread(img_path).pixel_array[300:768-300,200:1024-600,:],cmap=plt.cm.bone)
    plt.title(title[0])
    plt.axis('off')

    # Edge-detected Image
    plt.subplot(1, 2, 2)
    #Display the edge-detected image using matplotlib with a grayscale color map.
    plt.imshow(img_array, plt.cm.bone)
    plt.title(title[1])
    plt.axis('off')

    # Show the plot containing both images.
    plt.show()

def edge_connection_algorithm(img_path:str):
    #plot_img_hist(img_path=img_path)
    #1. read grayscale image
    image = dcmread(img_path).pixel_array
    #image=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #cv2.imwrite(img_path+'.png',image.pixel_array)
    #image=cv2.imread(img_path+'.png', cv2.IMREAD_GRAYSCALE)
    masks=[]
    #2. apply to kernels to calculate horizontal and vertical gradient
    gx = np.array([[1, 0], [0, -1]])
    gy = np.array([[0, 1], [-1, 0]])
    #gx = np.array([[[1, 0], [0, -1]],[[1, 0], [0, -1]],[[1, 0], [0, -1]]])
    #gy = np.array([[[0, 1], [-1, 0]],[[0, 1], [-1, 0]],[[0, 1], [-1, 0]]])
    print(image.shape)
    print('x',gx.shape,gx)
    print('y',gy.shape,gy)
    for channel in range(3):
        croped_img=image[300:768-300,200:1024-600,channel].copy()
        print(f'croped_img shape {croped_img.shape}')
        gradient_x = convolve(croped_img, gx)
        gradient_y = convolve(croped_img, gy)
        #3. compute the magnitude 
        print(gradient_x.shape, gradient_x)
        print(gradient_y.shape, gradient_y)
        magnitude=np.sqrt(pow(gradient_x,2)+pow(gradient_y,2))
        print('Magnitude',magnitude)
        logging.info(f'Magnitude: {magnitude}')
        threshold=5
        #ret, edges = cv2.threshold(croped_img, int(threshold), 255, cv2.THRESH_BINARY)
        edges = np.where(magnitude > threshold,1,0)
        logging.info(edges)
        print('Edges 1',edges.shape)
        #plt.hist(croped_img.ravel(),256,[0,256]) 
        #plt.show()
        plot_image(img_path=img_path,img_array=edges,title=['Original image','Edge-detected Image'])
        mask=component_analysis(img_path=img_path,edges=edges)
        masks.append(mask)
    mask_f=np.array(masks[1]+masks[2])
    plt.imshow(mask_f)
    plt.show()
    return mask_f,image[300:768-300,200:1024-600,:]
    # for channel in range(3):
    #     print(image.shape)
    #     print('x',gx.shape,gx)
    #     print('y',gy.shape,gy)
    #     croped_img=image[300:768-300,200:1024-600].copy()
    #     print(f'croped_img shape {croped_img.shape}')
    #     gradient_x = convolve(croped_img, gx)
    #     gradient_y = convolve(croped_img, gy)
    #     #3. compute the magnitude 
    #     print(gradient_x.shape, gradient_x)
    #     print(gradient_y.shape, gradient_y)
    #     magnitude=np.sqrt(pow(gradient_x,2)+pow(gradient_y,2))
    #     print('Magnitude',magnitude)
    #     logging.info(f'Magnitude: {magnitude}')
    #     threshold=5
    #     #ret, edges = cv2.threshold(croped_img, int(threshold), 255, cv2.THRESH_BINARY)
    #     edges = np.where(magnitude > threshold,magnitude,0)
    #     logging.info(edges)
    #     print('Edges 1',edges.shape)
    #     #plt.hist(croped_img.ravel(),256,[0,256]) 
    #     #plt.show()
    #     plot_image(img_path=img_path,img_array=edges,title=['Original image','Edge-detected Image'])
    #     mask=component_analysis(img_path=img_path,edges=edges)
    #     masks.append(mask)
    #return masks,image.pixel_array[300:768-300,200:1024-600,:]


# df=load_data()
# print(df['tirads'].value_counts())
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

BASE_DIR_DATBASE='C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/'
PATH_DICOM_IMAGES=f'{BASE_DIR_DATBASE}GUMED/58BA9974'
edges,images=edge_connection_algorithm(PATH_DICOM_IMAGES)
# plt.imshow(edges[1],cmap='gray')
# plt.show()
# print(edges[1].shape)
# print(edges)
#print(edges[1].shape,images.shape)
logging.info(edges)
Image.fromarray(CriminisiAlgorithm(img=images,mask=edges,patch_size=9,plot_progress=True).inpaint()).save('test.png', quality=100)
#img=np.asarray(Image.open('C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/Deep Neural Networks/scripts/test.png'))
#Image.fromarray(CriminisiAlgorithm(img=img,mask=edges[2],patch_size=9,plot_progress=True).inpaint()).save('test2.png', quality=100)
#img=np.asarray(Image.open('C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/Deep Neural Networks/scripts/test2.png'))
# Image.fromarray(CriminisiAlgorithm(img=images,mask=edges[0],patch_size=9,plot_progress=True).inpaint()).save('test3.png', quality=100)
#TODO na kazdym kanale osobno liczyc gradient, pozniej magnieture i pozniej connectedComponent