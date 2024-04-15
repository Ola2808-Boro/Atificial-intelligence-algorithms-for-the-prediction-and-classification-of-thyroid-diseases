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
 
logging.basicConfig(level=logging.INFO,filename='data_setup.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
PATH_XML_FILES='C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/DDTI Thyroid Ultrasound Images/*.xml'
PATH_IMAGES='C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/DDTI Thyroid Ultrasound Images/*.jpg'


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
    for xml_file in xml_files[:2]:
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


def create_masks(data_frame:pd.DataFrame):

    for idx in data_frame.index:
        images_path=data_frame['images_path'][idx]
        masks_path=data_frame['masks_path'][idx]
        print(len(images_path))
        for idx,img_path in enumerate(images_path):
            print(img_path)
            for mask in data_frame['mark'][idx]:
                regex=re.findall(r'_\d+',img_path)[0]
                if mask[0]['num']==regex.replace('_',''):
                    print(f"img {img_path}, regex {regex}")
                    img=Image.open(img_path)
                    draw = ImageDraw.Draw(img)
                    points=mask[0]['svg']
                    x_coordinates=[int(itm) for itm in re.findall(r'"x":\s*(\d+)',points)]
                    y_coordinates=[int(itm) for itm in re.findall(r'"y":\s*(\d+)',points)]
                    coordinates=list(zip(x_coordinates,y_coordinates))
                    logging.info(f"Coordinates num :{mask[0]['num']} {coordinates}")
                    draw.polygon(coordinates, fill=(255,255,255), outline=(255, 255, 255))
                    img.save(masks_path[idx])
                    img.show()


def analyze_data(data_frame:pd.DataFrame):
    print(f'Nan values in data frame {data_frame.isna().sum().sum()}')
    for item in data.columns:
        print(f'Number of Nan values in column {item} {data_frame[item].isna().sum().sum()}')
def load_data()->pd.DataFrame:
    df_read_json = pd.read_json('patient_data.json', lines=True)
    print("DataFrame using pd.read_json() method:")
    print(df_read_json)
    return df_read_json

paths=[PATH_XML_FILES,PATH_IMAGES]
#count_data(paths=paths)
#generate_dataset(data)
df=load_data()
create_masks(data_frame=df)