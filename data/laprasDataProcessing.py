# Lapras data format : Sensor type, sensor state, start time, end time 
# File name = activity label : [1 : 'Chatting', 2: 'Discussion', 3: 'GroupStudy', 4: 'Presentation', 5: 'NULL']
# Examples(csv) : Seat Occupy,1,1.490317862115E12,1.490319250294E12,23.136316666666666
# [1, 2, 3, 5, 4] : activity # (Chatting, Discussion, GroupStudy, NULL, Presentation), [119, 52, 40, 116, 129], # of activities

from glob import glob
import numpy as np
import pandas as pd

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

# Lapras construction
def laprasLoader(file_name, timespan, min_seq):
    
    print("Loading Lapras Dataset--------------------------------------")
    
    # for storing file names
    file_list = [] 

    # extract file names
    for x in glob(file_name):
        file_list.append(x)
    # sort list by file name
    file_list.sort() 

    # for finding sensor types
    sensor_list = []
    # for finding each instane's start time and end time
    time_list = []

    # find # of labels, sensor types, 
    for file in file_list:
        # 0: sensor type, 1: sensor state, 2: start_time, 3: end_ time, 4: duration
        temp_df = pd.read_csv(file, sep = ',', header = None).to_numpy() 

        print(file)           

        # if the file is not empty
        if(len(temp_df)>0):

            # get first row's start and end times (variable initialization)
            start_time = temp_df[0, 2]
            end_time = temp_df[len(temp_df)-1,3]            
            # for each row
            for i in range(0, len(temp_df)):
                # find each instane's start time and end time
                if(temp_df[i, 2] < start_time):
                    start_time = temp_df[i, 2] 
                if(temp_df[i, 3] > end_time):
                    end_time = temp_df[i, 3]

                # find entire sensor types
                if temp_df[i, 0] not in sensor_list:
                    sensor_list.append(temp_df[i, 0]) 

        # store each instane's start time and end time
        time_list.append([start_time, end_time])
        
    print(sensor_list)  # used only when initially extracting the sensor list    

    
    
    # for constructing dataset's data structure (return variable : an object list)
    dataset_list = []


    # for indicating the current file
    file_loc = 0

    # for setting sensor list index
    sensor_list = ['Seat Occupy', 'Sound', 'Brightness', 'Light', 'Existence', 'Projector', 'Presentation']


    # construct dataset's data structure 
    # for each file
    for file in file_list:
        # 0: sensor type, 1: sensor state, 2: start_time, 3: end_ time, 4: duration
        temp_df = pd.read_csv(file, sep = ',', header = None).to_numpy()

        # if the file is not empty
        if(len(temp_df)>0):              
            #print(int((time_list[count_file-1][1]-time_list[count_file-1][0])/(timespan)),len(item_list))
            # for temp variables for constructing dataset
            # for each instane's start time and end time
            start_time = time_list[file_loc][0]
            end_time = time_list[file_loc][1]  
            temp_dataset = np.zeros(((int)((end_time-start_time)/(timespan)), len(sensor_list)), dtype=int)
            
            # if the activity is not that short
            if(len(temp_dataset)> min_seq):
                # for each row of a file
                for i in range(0, len(temp_df)):
                #print("1", temp_df[i, 3], temp_df[i, 2], time_list[count_file][0] )
                #print(int((temp_df[i, 3]-time_list[count_file][0])/(timespan)), int((temp_df[i, 2]-time_list[count_file][0])/(timespan)))
                    # for each timestamp, fill in np if any value exists
                    for j in range(int((temp_df[i, 2]-start_time)/(timespan)), \
                                int((temp_df[i, 3]-start_time)/(timespan))):                    
                        # environment driven event
                        if(temp_df[i,0] == 'Sound' or temp_df[i,0] == 'Brightness'):
                            temp_dataset[j][sensor_list.index(temp_df[i, 0])] = int(temp_df[i,1])%10
                        # user driven event + actuator driven event
                        else:
                            temp_dataset[j][sensor_list.index(temp_df[i, 0])] += 1

                # append a instance into a dataset
                dataset_list.append(TSDataSet(temp_dataset, label_num(file), len(temp_dataset)))
        
        # for next file
        file_loc+=1

    print("Loading Lapras Dataset Finished--------------------------------------")
    return dataset_list