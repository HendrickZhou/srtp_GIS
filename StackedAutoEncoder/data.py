# -*- coding: utf-8 -*-
'''
this modual reads the file, and use the interface to convert the data to program-readable
data structure
@author Zhou Hang
'''

# extract the data/label from the filename
# return the data as tuple consisted of train set/label and eval test/label
def extract_data(filename):
    file = open(filename)
    for line in file:
        
def extract_label(filename):
    # TO DO
def dense_to_one_hot(label, class_num):
    # TO DO

start = 0
# next_batch operation for the layers start from 2nd layer
def next_batch(input,labels,batch_size):
	global start
	begin = start
	end = begin + batch_size
	start = end
	return input[begin:end], labels[begin:end]
 
 
# this class gather the data read from the file, and offer interface for get them
# it also define the method that help with shuffle the data
class data(object):
    def __init__(self, label, data):
        #self.labels = label
        self.datas = data
        
    def next_batch(batch_size):
        # TO DO
    
    @property
    def class_num(self):
        return self.class_num
    @property
    def label(self):
        return self.labels
    
    @property
    def data(self):
        return self.datas
    