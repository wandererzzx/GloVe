#!/usr/bin/env python
 
"""
.
.
.
Python Code
.
.
.
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 21:48:57 2017

@author: zuodi
"""
import numpy as np
from pycuda import driver, compiler, gpuarray, tools,scan
import time 
import pycuda.autoinit
import os


WWOC_M_kernel = """

// Check whether a word is freq
// If yes, return the corresponding index
// If no, return 0
__device__ int isWordFreq(int word, int *freq_word, int num_of_freq_word){
    int i;
    // Start from 1, 0 is UNKNOWN
    for (i=1; i < num_of_freq_word; i++) {
        if (freq_word[i] == word)
            return i;
    }
    
    // Possible point of optimization
    
    return 0;
}

#include "stdio.h"
// Length of context
#define window 10
__global__ void WWOC_M(int* corpora,    // The input text
                      int* freq_word,   // Array of frequent words
                      int* start_ptr,   // The starting position of each text
                      int work_load,    // work load of each block
                      int num_of_texts, // Length of corpora
                      int M_dim,        // Dimension of output, also number of frequent words
                      int debug,        // 0 or non-0: whether is debugging
                      float* M){        // Ouput
        
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bdim = blockDim.x;
    int num_of_freq_word = M_dim;
    // output matrix M is num_of_freq_word by num_of_freq_word
    
    if((threadIdx.x==1) && (blockIdx.x==1))
        //printf("Start kernel");
        //printf("num_of_freq_word=%d",num_of_freq_word);
        printf("Num of texts=%d",num_of_texts);
    
    



// Note:
// "The index of word": index in corpora[]  
// "The index of text": index in start_ptr[]
// start_ptr[] contains the index of the first word of each text
// e.g. corpora[start_ptr[3]+10] is text 3, word 10, 
// if corpora has at least 4 texts and text 3 has at least 11 words

// "Mapped index of word": Mapped from 0~260,000 to 0~9,999. Also index in output matrix


    
    // Calculate starting point of each thread
    int ideal_start_point = bx * work_load;               //Ideal starting point, index of first word this block process
    
    // debugging
    if ((debug) && (tx==0) &&(bx==633))
        printf("Finding start and end point...");
        
    int i;                                              // The index of the text this thread is processing
    
    // Each block process at least work_load words
    // but each text should be read in complete
    for (i=0 ;i<num_of_texts;i++){     
        if (start_ptr[i] < ideal_start_point){
            i++;
            continue;
            }
        else{
            ideal_start_point = start_ptr[i];           // Actual start point for each block
            break;
        }
    }
    
    int end_point;                         // The index of the first word NEXT BLOCK process
    int j;                                 // The index of the first text NEXT BLOCK process
    //__shared__ start_point[blockDim.x];
    //start_point[tx] = ideal_start_point;
    
    
    for (j=0;j <= num_of_texts;j++){   
    // use the value of previous i, but keep the i since we are using i later
        if (start_ptr[j] < (bx+1)*work_load ){
            j++;
            continue;
            }
        else{
            end_point = start_ptr[j];           // Actual end point for each block
            break;
        }
    }
        
    // Special case        
    if (bx+1 == gridDim.x){
        end_point = start_ptr[num_of_texts];
    }        
    // One block reads texts from ideal_start_point to end_point
    
    
    
    // i is the index of the first text to process
    // j-1 is the index of the last text to process
        
    
    
    int word_idx = 0;           // Index of context word
    int out_idx_x, out_idx_y;   // Mapped indexes of target word and context word
    
    
    //__shared__ int freq_word_idx=[]
    
    
    int tx_ptr = tx + ideal_start_point;            // The index of target word this thread is processing
    
    
    
    // Update text index according to thread pointer, in case blockDim.x > length of current text
    // e.g. The first text has 500 words, second has 300, but one block has 1024 threads,
    // 500 threads look at text 0, 300 look at text 1, the rest look at text2.
    
    // This while clause will be used to update i multiple times later
    while( (tx_ptr>=start_ptr[i+1]) && (i<j-1))
        i = i+1;
    
    // The index of the first and last word in current text
    int start_of_current_text = start_ptr[i];
    int start_of_next_text = start_ptr[i+1];


    // debugging
    //if ( (tx==0) &&(bx > 900))
        //printf("Start and end point found!From %d to %d,bx=%d...",ideal_start_point,end_point,bx);  
    
    // Main loop
    
    while(tx_ptr < end_point){
            
        out_idx_x = isWordFreq(corpora[tx_ptr], freq_word, num_of_freq_word);   //Map target word 

        if (out_idx_x <= 0){
            tx_ptr = tx_ptr + blockDim.x;   // coalesced!
            while( (tx_ptr>=start_ptr[i+1]) && (i<j-1))
                i = i+1;
            
            continue;       // target word not frequent
        }
        else if(out_idx_x>=10000){      // Should not happen, just in case
            printf("out_idx_x=%d, tx=%d,bx=%d,tx_ptr=%d,i=%d",out_idx_x,tx,bx,tx_ptr,i);
        }
    
        
        start_of_current_text = start_ptr[i];
        start_of_next_text = start_ptr[i+1];
        
            
        // Debugging
        //if ((debug) && (tx==0) &&(bx==633))
            //printf("In while loop, tx_ptr=%d,i=%d,bdim=%d...",tx_ptr,i,bdim);

        
        // process context before the target
        for (word_idx = tx_ptr - window; word_idx<tx_ptr; word_idx+=1){       
                
                // Debugging
            //if ((debug) && (out_idx_x == 1) && isWordFreq(corpora[word_idx], freq_word, num_of_freq_word) ==1201){
                //printf("Error found! bx=%d,tx=%d,tx_ptr=%d,i=%d...",bx,tx,tx_ptr,i);
            //}

        
        
            if ((word_idx < ideal_start_point) || (word_idx >= end_point)){
                continue;
            }
            if ((word_idx <start_of_current_text) || (word_idx >= start_of_next_text)){        // In previous text?
                continue;               
            }
            
            if ((debug) && (tx==0) &&(bx==633))
                printf("word_idx=%d,i=%d,",word_idx,i);
            
            // Compute
            out_idx_y = isWordFreq(corpora[word_idx], freq_word, num_of_freq_word);
                       
            
            if (out_idx_y > 0){       // Frequent word
               if(out_idx_y <10000)
                   // Add 1/distance
                   atomicAdd(&M[out_idx_x * num_of_freq_word + out_idx_y],1.0f/((float)(tx_ptr-word_idx)));
                   
                   // Add 1 regardless of distance
                   // Use this to test correctness 
                   //atomicAdd(&M[out_idx_x * num_of_freq_word + out_idx_y],1);
                else                  // Should not happen, just in case
                    printf("out_idx_y=%d, tx=%d,bx=%d,tx_ptr=%d,i=%d",out_idx_y,tx,bx,tx_ptr,i);
            }        
        }
            
            
        
        //if ((debug) && (tx==0) &&(bx==633))
            //printf("In the middle of while loop, tx_ptr=%d,i=%d,bdim=%d...",tx_ptr,i,bdim);
        
        // process context after the target
        for (word_idx = tx_ptr + 1; word_idx<tx_ptr + window+1; word_idx+=1){

            //if ((debug) && (out_idx_x == 1) && isWordFreq(corpora[word_idx], freq_word, num_of_freq_word) ==1201){
                //printf("Error found! bx=%d,tx=%d,tx_ptr=%d,i=%d...",bx,tx,tx_ptr,i);
            //}

        
            if(word_idx >= end_point)
                break;      // Out of boundary
            if (word_idx >= start_of_next_text){        // In previous text?
                break;
            }
            else if (word_idx <start_of_current_text) { // This should not happen
                printf("Error!");
                break;
            }
                
            // Compute
            out_idx_y =  isWordFreq(corpora[word_idx], freq_word, num_of_freq_word);        
            
            if (out_idx_y > 0){       // is Frequent word
               if(out_idx_y <10000)
                   atomicAdd(&M[out_idx_x * num_of_freq_word + out_idx_y],1.0f/((float)(word_idx-tx_ptr)));
                   //atomicAdd(&M[out_idx_x * num_of_freq_word + out_idx_y],1);
               else                  // Should not happen, just in case
                   printf("out_idx_y=%d, tx=%d,bx=%d,tx_ptr=%d,i=%d",out_idx_y,tx,bx,tx_ptr,i);
            }
        }
        
        tx_ptr = tx_ptr + blockDim.x;   // coalesced!
        
        // Update index of text
        while( (tx_ptr>=start_ptr[i+1]) && (i<j-1))
            i = i+1;
        //if ((debug) && (tx==0) &&(bx==633))
            //printf("At end of while loop, tx_ptr=%d,i=%d,bdim=%d...",tx_ptr,i,bdim);
        
    }
    

}
"""

mod = compiler.SourceModule(WWOC_M_kernel)     
WWOC_M = mod.get_function("WWOC_M")




import csv 
# Load frequent words
freq_word = []
with open('freq_wordlist.csv','r') as freq_file: 
    temp = freq_file.readlines()#.split()
    for line in temp:
        freq_word.append(line.split())

d_freq_word = gpuarray.to_gpu(np.array([eachIdx for word,eachIdx in freq_word]).astype(np.int32))




# Load the corpora
corpora = []
for oneFile in os.listdir(os.getcwd()):
    if oneFile.startswith("wiki") and oneFile.endswith("csv"):  # Load all files
    #if oneFile.startswith("wiki_00001") and oneFile.endswith("csv"):  # Load one file
        with open(oneFile,"r") as corporaFile:
            for eachText in corporaFile.readlines():
                corpora.append(eachText.split())



# Turn the corpora into 1D array
allTexts = []
for eachText in corpora:
    allTexts+=eachText
allTexts = np.array(allTexts,dtype=np.int32)
# corpora to device
d_allTexts = gpuarray.to_gpu(allTexts)

# The starting point of each text of corpora in 1D
size_of_text = []
for eachText in corpora:
    size_of_text.append(len(eachText))
size_of_text.append(len(eachText))    
    
size_of_text = np.array(size_of_text,dtype=np.int32)
d_size_of_text = gpuarray.to_gpu(size_of_text)


# Determine work_load of each block
'''
Note: we want to balance work load of each block, so they start at the first 
text with index greater than block_index * work_load.
'''
work_load =int( ((max(size_of_text)-1)/1024 +1)*1024    ) # each blcok deals with # words
print 'work_load=',work_load
print
print 'max of size_of_text=',max(size_of_text)
assert work_load > max(size_of_text)     # If not, some words will not be covered


# Output: Word-word cooccurrence matrix
M = np.zeros((len(freq_word),len(freq_word)),dtype = np.float32)
d_M = gpuarray.to_gpu(M)


# Compute exclusive scan on gpu and cpu and compare
scan_kernel = scan.ExclusiveScanKernel(np.int32,"a+b",0)
scan_kernel(d_size_of_text)
py_scan = np.cumsum(size_of_text,axis=0)
# Turn py inclusive to exclusive scan
for i in range(len(py_scan)):
    py_scan[i] -= size_of_text[i]

# Compare python result and gpu result
compare = np.equal(py_scan,d_size_of_text.get().astype(np.int32))
for i in range(len(compare)):
    if compare[i] == False:
        print i
assert (False not in compare)   # No difference in scan of python and gpu

d_start_ptr = d_size_of_text
#d_start_ptr = gpuarray.to_gpu(np.array(py_scan).astype(np.int32))


start = time.time()
WWOC_M(d_allTexts,              # Corpora
       d_freq_word,             # List of frequent word
       d_start_ptr,             # Index of first word of each text
       np.int32(work_load),     # Work load of each block
       np.int32(len(corpora)),  # Number of texts
       np.int32(10000),         # Number of frequent words
       np.int32(0),             # Debugging?
       d_M,                     # Output
       block = (1024,1,1),
       grid = ((len(allTexts)-1)/work_load+1,1)
       #shared = np.int32(len(corpora))
       
       )
end = time.time()
print 'Running time:',end-start
print

output = d_M.get()

#print output.shape

#output = np.sum(output,axis=0)
#print output

# Save the output
with open("WWOC_matrix_all.csv","w+") as output_file:
    np.savetxt(output_file,output)
'''
output_file = 'WWOC_matrix_int.csv'
np.savetxt(output_file,output,delimiter = ',')
'''