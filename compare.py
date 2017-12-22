# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 15:18:10 2017

@author: zuodi
"""

import numpy as np
import csv
import os
import time
import codecs

gpu_result = []
#gpu_file= codecs.open('WWOC_matrix_all_part.csv','r')
gpu_file= codecs.open('WWOC_matrix_int.csv','r')
gpu_reader = csv.reader(gpu_file,delimiter=',')#,quoting=csv.QUOTE_NONNUMERIC)
# Since the file is too big, I read one line at a time
# gpu_result = gpu_reader.next()
        
cpu_result = []
#cpu_file = codecs.open('float_co_occurMat.csv','r')
cpu_file = codecs.open('single_co_occurMat.csv','r')
cpu_reader = csv.reader(cpu_file,delimiter=',')#,quoting=csv.QUOTE_NONNUMERIC)

cpu_sum=0
gpu_sum=0
error = []
for i in range(0,5):
    #gpu_result = gpu_file.readline().split(',')
    #cpu_result = cpu_file.readline().split(',')
    gpu_result = gpu_reader.next()
    cpu_result = cpu_reader.next()
    for j in range(0,10000):
        cpu_sum+=float(cpu_result[j])
        gpu_sum+=float(gpu_result[j])
        if gpu_result[j] == cpu_result[j]:
            error.append(0.0)
        elif gpu_result[j] != '' and cpu_result[j] != '' and float(cpu_result[j]) !=0:
            error_rate = np.abs(float(gpu_result[j])/float(cpu_result[j]) -1.0)
            error.append(error_rate)
            '''
            if np.abs(error_rate) > 0.05 :
                print error_rate
                print i,j
                print float(gpu_result[j]),float(cpu_result[j])
                print 
                '''
                
print(np.max(error))
print(np.mean(error))

for i in range(len(error)):
    if error[i] > 0.00001:
        print i,error[i]