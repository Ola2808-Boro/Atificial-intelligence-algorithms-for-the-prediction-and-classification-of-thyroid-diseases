import seaborn as sns
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import json
from random import randrange


MAIN_PATH="Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/"

columns={
    'age':'continuous',
	'sex':['M', 'F'],
	'on thyroxine':	'boolean',
	'query on thyroxine':	'boolean',
	'on antithyroid medication':'boolean',
	'sick':'boolean',
	'pregnant':'boolean',
	'thyroid surgery':'boolean',
	'I131 treatment':'boolean',
	'query hypothyroid':'boolean',
	'query hyperthyroid':'boolean',
	'lithium':'boolean',
	'goitre':'boolean',
	'tumor':'boolean',
	'hypopituitary':'boolean',
	'psych':'boolean',
	'TSH measured':'boolean',
	'TSH':'continuous',
	'T3 measured':'boolean',
	'T3':'continuous',
	'TT4 measured':'boolean',
	'TT4':'continuous',
	'T4U measured':'boolean',
	'T4U':'continuous',
	'FTI measured':'boolean',
	'FTI':'continuous',
	'TBG measured':'boolean',
	'TBG':'continuous',
	'referral source':['WEST', 'STMW', 'SVHC', 'SVI', 'SVHD', 'other'],
    'diagnosis':{
    'hyperthyroid conditions':
        {
		'A':'hyperthyroid',
		'B':'T3 toxic',
		'C':'toxic goitre',
		'D':'secondary toxic',
        },
	'hypothyroid conditions':
        {
        'E':'hypothyroid',
		'F':'primary hypothyroid',
		'G':'compensated hypothyroid',
		'H':'secondary hypothyroid',
        },	
	'binding protein':
		{
            'I':'increased binding protein',
		    'J':'decreased binding protein',
        },
	'general health':
		{
            'K':'concurrent non-thyroidal illness'
        },
	'replacement therapy':
        {
            'L':'consistent with replacement therapy',
            'M':'underreplaced',
            'N':'overreplaced',
        },
	'antithyroid treatment':
		{
            'O':'antithyroid drugs',
            'P':'I131 treatment',
            'Q':'surgery',
        },
	'miscellaneous':
		{
            'R':'discordant assay results',
            'S':'elevated TBG',
            'T':'elevated thyroid hormones',
        }
}
}
data_distribution={
    'possible_cases':64,
    'configuration':['TSH measured', 'T3 measured', 'TT4 measured', 'T4U measured','FTI measured', 'TBG measured'],
    'data':[],
    'distribution':{},
    'distribution_decoded':{}
}
	

def read_data(filename:str):
    return pd.read_csv(filename)

def create_csv_file():
    content=pd.read_csv(f"{MAIN_PATH}thyroid0387.data",header=None)
    content.columns=list(columns.keys())
    content.to_csv(f"{MAIN_PATH}thyroid0387.csv")


def detect_nan_value(thyroid_data):
    with open(f"{MAIN_PATH}thyroid0387_clean.csv",mode='w+') as f:
        for idx,row in thyroid_data.iterrows():
            #print(idx,list(row.items()))
            for item in list(row.items()):
                name,value=item
                if not pd.isna(value):
                    f.write(f'{str(idx)}\n')
                    print(value)
                


def analysis():
    thyroid_data=pd.read_csv(f"{MAIN_PATH}thyroid0387.csv")
    data=thyroid_data[data_distribution['configuration']]
    for idx,row in data.iterrows():
        code=[]
        for idx,item in row.items():
            if item=='t':
                code.append(1)
            else:
                code.append(0)
        if ''.join(str(value) for value in code) not in data_distribution['distribution'].keys():
            data_distribution['distribution'].update({
                ''.join(str(value) for value in code):1
            })
        else:
            data_distribution['distribution'][''.join(str(value) for value in code)]+=1
        data_distribution['data'].append(code)
    data_analysis=pd.DataFrame(data_distribution['data'],columns=data_distribution['configuration'])
    print(data_distribution['distribution'])
    for key in data_distribution['distribution'].keys():
        decode=''
        print(str(key))
        counter=0
        for item in str(key):
            print(item,counter)
            if item=='1':
                if counter==5:
                    decode+=data_distribution['configuration'][counter]
                else:
                    decode+=data_distribution['configuration'][counter]+' '
            counter+=1
        data_distribution['distribution_decoded'].update({
            decode:data_distribution['distribution'][key]
        })
    print(data_distribution['distribution_decoded'])
    json_object=json.dumps(data_distribution,indent=len(list(data_distribution.keys())))
    with open(f"{MAIN_PATH}thyroid0387.json",'w+') as f:
        f.write(json_object)
    #data_analysis_distribution=pd.DataFrame(data_distribution['distribution'])
    #print(data_analysis_distribution)
    #data_analysis.to_csv('Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/databases/Thyroid Disease Garvan Institute/thyroid0387_analysis.csv')


def analysis_to_select_data():
    with open(f"{MAIN_PATH}thyroid0387.json",'r') as f:
        json_object=json.load(f)
    decoded_data=json_object['distribution_decoded']
    max_value=sorted(decoded_data.values())[-1]
    print(max_value)
    max_value_key=list(decoded_data.keys())[list(decoded_data.values()).index(max_value)]
    keys=max_value_key.split('measured ')
    print(keys)
    thyroid_data=read_data(f"{MAIN_PATH}thyroid0387.csv")
    thyroid_data_selected=thyroid_data.loc[(thyroid_data['TSH measured']=='t')&(thyroid_data['T3 measured']=='t')&(thyroid_data['TT4 measured']=='t')&(thyroid_data['T4U measured']=='t')&(thyroid_data['FTI measured']=='t')]
    thyroid_data['diagnosis_extracted']=thyroid_data['diagnosis'].str.extract(r'([a-zA-Z-])')
    print(thyroid_data.head())
    print(thyroid_data['diagnosis_extracted'].value_counts(),type(thyroid_data['diagnosis_extracted'].value_counts()))
    ax=sns.barplot(data=thyroid_data['diagnosis_extracted'].value_counts())
    ax.bar_label(ax.containers[0], fontsize=10)
    plt.savefig(f'{MAIN_PATH}thyroid_diagnosis_distribution.png')
    plt.clf()
    thyroid_data_selected=thyroid_data.loc[(thyroid_data['TSH measured']=='t')&(thyroid_data['T3 measured']=='t')&(thyroid_data['TT4 measured']=='t')&(thyroid_data['T4U measured']=='t')&(thyroid_data['FTI measured']=='t')]
    thyroid_data_selected['diagnosis_extracted']=thyroid_data_selected['diagnosis'].str.extract(r'([a-zA-Z-])')
    ax=sns.barplot(data=thyroid_data_selected['diagnosis_extracted'].value_counts())
    ax.bar_label(ax.containers[0], fontsize=10)
    plt.savefig(f'{MAIN_PATH}thyroid_diagnosis_selected_distribution.png')
    # print(thyroid_data_selected.head())
    thyroid_data_selected.drop(['TBG','diagnosis'],axis=1,inplace=True)
    thyroid_data_selected.to_csv(f'{MAIN_PATH}thyroid_diagnosis_selected_distribution.csv')
    # print(thyroid_data_selected['diagnosis_extracted'].value_counts())

def analysis_selected_data():
    thyroid_data=read_data(f'{MAIN_PATH}thyroid_diagnosis_selected_distribution.csv')
    thyroid_data=clean_data(filename='thyroid_diagnosis_selected_distribution.csv')
    for column in thyroid_data.columns:
        print('Columns',column)
        if 'F' in list(thyroid_data[column].unique()) and len(list(thyroid_data[column].unique()))==2:
            thyroid_data[column].replace({'F': 0, 'M': 1}, inplace=True)
        elif 't' in list(thyroid_data[column].unique()) or 'f' in list(thyroid_data[column].unique()):
            thyroid_data[column].replace({'f': 0, 't': 1}, inplace=True)
        else:
            if column=='referral source':
                thyroid_data[column].replace({'WEST':0,
                                            'STMW':1,
                                            'SVHC':2,
                                            'SVI':3, 
                                            'SVHD':4, 
                                            'other':5}, inplace=True)
                
            elif column=='diagnosis_extracted':
                print('Change')
                thyroid_data[column].replace({'A':0,
                                            'B':1,
                                            'C':2,
                                            'D':3,
                                            'E':4,
                                            'F':5,
                                            'G':6,
                                            'H':7,
                                            'I':8,
                                            'J':9,
                                            'K':10,
                                            'L':11,
                                            'M':12,
                                            'N':13,
                                            'O':14,
                                            'P':15,
                                            'Q':16,
                                            'R':17,
                                            'S':19,
                                            'T':20,
                                            '-':21}, inplace=True)
    print('Unique',thyroid_data['diagnosis_extracted'].unique())
    print('Unique',thyroid_data['referral source'].unique())
    thyroid_data.drop(['Unnamed: 0.1',  'Unnamed: 0'],axis=1,inplace=True)
    thyroid_data.to_csv(f'{MAIN_PATH}thyroid_diagnosis_selected_distribution_coded.csv')
    print(thyroid_data.columns)
    corr=thyroid_data.corr()
    print(corr)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(f'{MAIN_PATH}thyroid_diagnosis_selected_correlation.png')
    


def visualization():
    thyroid_data=read_data(f'{MAIN_PATH}thyroid_diagnosis_selected_distribution_coded.csv')
    thyroid_data.drop(labels='Unnamed: 0',axis=1,inplace=True)
    #columns_list=['TSH','T3','TT4','T4U','FTI']
    thyroid_data=thyroid_data[['TSH','T3','TT4','T4U','FTI']]
    labels=list(thyroid_data.columns)
    r = randrange(1)
    g = randrange(1)
    b = randrange(1)
    #colors=[[r, g, b] for label in labels]
    #print(colors,labels)

    fig, ax = plt.subplots()
    print(type(thyroid_data),thyroid_data.to_numpy())
    bplot = ax.boxplot(thyroid_data,
                    patch_artist=True,
                    labels=labels)
        # for patch, color in zip(bplot['boxes'], colors):
        #     patch.set_facecolor(color)
    plt.xticks(rotation=90)
    plt.savefig(f'{MAIN_PATH}thyroid_diagnosis_selected_box_plots_all.png')
    plt.clf()

def clean_data(filename:str):
    thyroid_data=read_data(f"{MAIN_PATH}{filename}")
    thyroid_data=thyroid_data.replace('?',np.nan)
    print(thyroid_data.dtypes)
    print(thyroid_data.info())
    #detect_nan_value(thyroid_data)
    thyroid_data.dropna(inplace=True)
    print(thyroid_data.info())

    return thyroid_data

#analysis_to_select_data()
#analysis_selected_data()
visualization()


# clean_data()