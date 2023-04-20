from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
from data.LaprasDataProcessing import laprasLoader
from data.CasasDataProcessing import casasLoader
from data.ArasDataProcessing import arasLoader
from data.OpportunityDataProcessing import opportunityLoader
from tsaug import *


# Static variable
#timespan = 1000 # for each timespan sec (1000==1 sec)
#min_seq = 10 # minimum sequence length
#min_samples = 10 # minimum # of samples

# for storing dataset element
class TSDataSet:
    def __init__(self,data, label, length):
        self.data = data
        self.label = int(label)
        self.length= int(length)

# use for lapras dataset
def label_num(filename):
    label_cadidate = ['Chatting', 'Discussion', 'GroupStudy', 'Presentation', 'NULL']
    label_num = 0
    for i in range(len(label_cadidate)):
        if filename.find(label_cadidate[i]) > 0:
            label_num = i+1    
    return label_num

# use for dataset normalization 
def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
        
    return round(df_norm,3)

class TimeseriesDataset(Dataset):   
    def __init__(self, data, window, target_cols):
        self.data = torch.Tensor(data)
        self.window = window
        self.target_cols = target_cols
        self.shape = self.__getshape__()
        self.size = self.__getsize__() 
    def __getitem__(self, index):
        x = self.data[index:index+self.window]
        y = self.data[index+self.window,0:target_cols]
        return x, y 
    def __len__(self):
        return len(self.data) -  self.window     
    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)    
    def __getsize__(self):
        return (self.__len__())

def visualization_data(dataset_list, file_name, activity_num):
    print("Visualizing Dataset --------------------------------------")
    label_count = [0 for x in range(activity_num)]
    # for visualization
    for k in range(len(dataset_list)):
        visual_df = pd.DataFrame(dataset_list[k].data)

        fig, ax = plt.subplots(figsize=(10, 6))
        axb = ax.twinx()

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True)

        # Plotting on the first y-axis
        for i in range(len(dataset_list[0].data[0])):
            ax.plot(visual_df[i], label = str(i+1))

        ax.legend(loc='upper left')
        
        plt.savefig(file_name+'visualization/'+str(dataset_list[k].label)+'_'+str(label_count[dataset_list[k].label-1])+'.png')
        plt.close(fig)
        label_count[dataset_list[k].label-1]+=1

    print("Visualizing Dataset Finished--------------------------------------")

def count_label(dataset_list):

    # find types and counts of labels
    types_label_list =[]
    count_label_list = []

    for i in range(len(dataset_list)):
        if(dataset_list[i].label not in types_label_list):
            types_label_list.append(dataset_list[i].label)
            count_label_list.append(1)
        else:
            count_label_list[types_label_list.index(dataset_list[i].label)]+=1

    print('types_label :', types_label_list)
    print('count_label :', count_label_list) 
    print('sum of # episodes:', sum(count_label_list))  
                
    return types_label_list, count_label_list

def count_label_labellist(labellist):
    # finding types and counts of label
    types_label_list =[]
    count_label_list = []
    for i in range(len(labellist)):
        if(labellist[i] not in types_label_list):
            types_label_list.append(labellist[i])
            count_label_list.append(1)
        else:
            count_label_list[types_label_list.index(labellist[i])]+=1

    print('types_label :', types_label_list)
    print('count_label :', count_label_list)   
    print('sum of # episodes:', sum(count_label_list))
                
    return types_label_list, count_label_list

def padding_by_max(lengthlist, normalized_df):
   
    # reconstruction of datalist    
    datalist=[]
    reconst_list =[]
    count_lengthlist = 0
    print("max padding (length): ", max(lengthlist))

    # reconstruction of normalized list    
    # for each row
    for i in range(len(lengthlist)):
        reconst_list =[]    
        # cut df by each length
        for j in range(count_lengthlist,(count_lengthlist+lengthlist[i])):
            reconst_list.append(normalized_df.iloc[j,:].tolist())            
        count_lengthlist += lengthlist[i]

        #padding to each data list
        if((max(lengthlist)-lengthlist[i])%2 == 0):
            p2d = (0, 0, int((max(lengthlist)-lengthlist[i])/2), int((max(lengthlist)-lengthlist[i])/2))
        else :
            p2d = (0, 0, int((max(lengthlist)-lengthlist[i]+1)/2)-1, int((max(lengthlist)-lengthlist[i]+1)/2))
        datalist.append(F.pad(torch.tensor(reconst_list),p2d,"constant",-1))
        
    # convert to tensor    
    datalist = torch.stack(datalist)
    return datalist

def padding_by_mean(lengthlist, normalized_df):
    # reconstruction of datalist    
    datalist=[]
    reconst_list =[]
    count_lengthlist = 0
    mean_length = int(sum(lengthlist)/len(lengthlist))
    print("mean padding (length):", mean_length)
    for i in range(len(lengthlist)):
        reconst_list =[]    
        # cut df by each length
        if(lengthlist[i]>=mean_length): # length is larger than mean
            for j in range(count_lengthlist, count_lengthlist+mean_length):
                reconst_list.append(normalized_df.iloc[j,:].tolist())
            datalist.append(torch.tensor(reconst_list))
        else: # length is smaller than mean
            for j in range(count_lengthlist, (count_lengthlist+lengthlist[i])):
                reconst_list.append(normalized_df.iloc[j,:].tolist())
            # padding to the end 
            p2d = (0, 0, 0, mean_length-lengthlist[i])
            datalist.append(F.pad(torch.tensor(reconst_list),p2d,"constant",-1))    
        count_lengthlist += lengthlist[i]
    
    # convert to tensor    
    datalist = torch.stack(datalist)    
    return datalist

def reconstrct_list(length_list, normalized_df):
    
    # reconstruction of datalist    
    data_list=[]
    reconst_list =[]
    count_lengthlist = 0

    # for each row
    for i in range(len(length_list)):
        reconst_list =[]    
        # append by each length
        for j in range(count_lengthlist,(count_lengthlist+length_list[i])):
            reconst_list.append(normalized_df.iloc[j,:].tolist())            
        count_lengthlist += length_list[i]
        data_list.append(torch.tensor(reconst_list))
    return data_list

# split data into train/validate/test 
def loading_data(dataset, padding, timespan, min_seq, min_samples): 

    # Constructing data structure for each dataset
    if dataset == 'lapras':
        dataset_list = laprasLoader('data/Lapras/*.csv', timespan, min_seq)        
        #visualization_data(dataset_list, 'KDD2022/data/Lapras/', 5)
    elif dataset == 'casas':
        dataset_list = casasLoader('data/CASAS/*.txt', timespan, min_seq)
        #visualization_data(dataset_list, 'KDD2022/data/CASAS/', 15)
    elif dataset == 'aras_a':
        dataset_list = arasLoader('data/Aras/HouseA/*.txt', timespan, min_seq)
        #visualization_data(dataset_list, 'KDD2022/data/Aras/HouseA/', 27*100 + 27)
    elif dataset == 'aras_b':
        dataset_list = arasLoader('data/Aras/HouseB/*.txt', timespan, min_seq)
        #visualization_data(dataset_list, 'KDD2022/data/Aras/HouseB/', 27*100 + 27)
    elif dataset == 'opportunity':
        dataset_list = opportunityLoader('data/Opportunity/*.dat', timespan, min_seq)
        #visualization_data(dataset_list, 'KDD2022/data/Opportunity/', 5)
    
    # Data Augmentation Module
    print('Augmentation-------------------')

    my_aug = (    
        #AddNoise(scale=0.01)
        #Convolve(window="flattop", size=11)
        #Crop(size=1)
        #Drift(max_drift=0.7, n_drift_points=5)
        #Dropout( p=0.1,fill=0)
        #Pool(size=2) 
        #Quantize(n_levels=20)
        #Resize(size=200)
        #Reverse()
        TimeWarp(n_speed_change=5, max_speed_ratio=3)
    )

    

 #    for i in range(dataset_len):            
 #       aug = my_aug.augment(dataset_list[i].data)  
 #       ts_ds = TSDataSet(aug, dataset_list[i].label, dataset_list[i].length)
 #       dataset_list.append(ts_ds)
    
    # For give the same number of data size
    types_label_list, count_label_list = count_label(dataset_list)
    max_label_count = max(count_label_list)

    sub_count_label = [0] * len(types_label_list)
    for i in range(len(types_label_list)):
        sub_count_label[i] = max_label_count - count_label_list[i]

    print(sub_count_label)

    copy_count_label = sub_count_label.copy()

    dataset_len = len(dataset_list)

# temporal aspect data augmentation
#    for i in range(dataset_len): 
#        for j in range(math.ceil(sub_count_label[dataset_list[i].label-1]/count_label_list[dataset_list[i].label-1])): 
#            if copy_count_label[dataset_list[i].label-1] > 0:          
#                aug = my_aug.augment(dataset_list[i].data.T)   
#                ts_ds = TSDataSet(aug.T, dataset_list[i].label, len(aug.T))
#                dataset_list.append(ts_ds)
#                copy_count_label[dataset_list[i].label-1] = copy_count_label[dataset_list[i].label-1]-1   
    
#    for i in range(len(dataset_list)): 
#        aug = my_aug.augment(dataset_list[i].data.T)   
#        ts_ds = TSDataSet(aug.T, dataset_list[i].label, len(aug.T))
#        dataset_list.append(ts_ds)

# sensor aspect data augmentation
    for i in range(dataset_len): 
        for j in range(math.ceil(sub_count_label[dataset_list[i].label-1]/count_label_list[dataset_list[i].label-1])): 
            if copy_count_label[dataset_list[i].label-1] > 0:          
                aug = my_aug.augment(dataset_list[i].data)   
                ts_ds = TSDataSet(aug, dataset_list[i].label, dataset_list[i].length)
                dataset_list.append(ts_ds)
                copy_count_label[dataset_list[i].label-1] = copy_count_label[dataset_list[i].label-1]-1   

    for i in range(len(dataset_list)): 
        aug = my_aug.augment(dataset_list[i].data)   
        ts_ds = TSDataSet(aug, dataset_list[i].label, dataset_list[i].length)
        dataset_list.append(ts_ds)


            



    print('Before padding-----------------')
    types_label_list, count_label_list = count_label(dataset_list)
    
    # convert object-list to list-list
    label_list=[]
    # store each length of instances
    length_list=[]
    # for temporal storage
    temp_list=[]

    # Normalized Module
    # for each instance
    for i in range(len(dataset_list)):
        # select datalist by min_samples
        if(count_label_list[types_label_list.index(dataset_list[i].label)] >= min_samples):
            #datalist.append(dataset_list[i].data)        
            label_list.append(dataset_list[i].label)
            length_list.append(dataset_list[i].length)

            for j in range(dataset_list[i].length):
                temp_list.append(dataset_list[i].data[j])     
               
    # normalization of dataframe
    normalized_df = min_max_scaling(pd.DataFrame(temp_list))
    normalized_df = normalized_df.fillna(0)

    # reconstruction of list (padding is option : max or mean)
    if padding == 'max':
        datalist = padding_by_max(length_list, normalized_df)
        #print('tensor_shape', datalist.size())
    elif padding =='mean':
        datalist = padding_by_mean(length_list, normalized_df)
        #print('tensor_shape', datalist.size())
    else:
        datalist = reconstrct_list(length_list, normalized_df)
    

    print('After padding-----------------')
    
    return datalist, label_list, types_label_list

def splitting_data(dataset, test_ratio, valid_ratio, padding, seed, timespan, min_seq, min_samples): 
    datalist, labellist, num_classes = loading_data(dataset, padding, timespan, min_seq, min_samples)

    labellist = (np.array(labellist)-1).tolist()
    count_label_labellist(labellist)
    # Split train and valid dataset
    train_list, test_list, train_label_list, test_label_list = train_test_split(datalist, labellist, test_size=test_ratio, stratify= labellist, random_state=seed) 
    train_list, valid_list, train_label_list, valid_label_list= train_test_split(train_list, train_label_list, test_size=valid_ratio, stratify=train_label_list, random_state=seed)

    print(f"Train Data: {len(train_list)} --------------") 
    count_label_labellist(train_label_list)
    print(f"Validation Data: {len(valid_list)} --------------")
    count_label_labellist(valid_label_list)
    print(f"Test Data: {len(test_list)} --------------") 
    count_label_labellist(test_label_list)
    return num_classes, datalist.cuda(), train_list.cuda(), valid_list.cuda(), test_list.cuda(), torch.tensor(labellist).cuda(), torch.tensor(train_label_list).cuda(), torch.tensor(valid_label_list).cuda(), torch.tensor(test_label_list).cuda()