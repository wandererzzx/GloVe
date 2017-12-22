# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:33:47 2017

@author: zuodi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:58:19 2017

@author: zuodi
"""

'''
Environment: python 3, nltk
'''


import os
import ast

from sklearn.feature_extraction.text import CountVectorizer



     

import csv

# Load the data
dataset = []        # Undecoded json format
AA_file = sorted(os.listdir(r'.\text\AA'))
for eachFile in AA_file:
    with open(u'.\\text\\AA\\' +eachFile,encoding="utf8") as oneFile:
        allLines = oneFile.readlines()
        for oneLine in allLines:
            dataset.append(oneLine)
    
data = []           # Decoded
for each in dataset:
   oneDict = ast.literal_eval(each)
   data.append(oneDict)
  
# Length of corpora to process
length = len(data)   

def get_vector(data,t):
    if type(t) == str:
        pass
    elif type(t) == int:
        t = get_text(data,t).split()
    elif type(t) == list:
        return get_vector(t[0])
    else:
        raise Exception("t should be str, integer or list.")
        #t = get_text(1).split()
    t_token=[]
    for each in t:
        each = each#.encode('utf-8')
        stemmed_word = analyzer(each)
        if len(stemmed_word) != 0:      
            stemmed_word = stemmed_word[0]#.encode('utf-8')#str(stemmed_word[0])
        else:       # Nothing in it
            #print each
            continue
    
        if vectorizer.vocabulary_.get(stemmed_word) != None:
            t_token.append(vectorizer.vocabulary_.get(stemmed_word))
        else:
            #print each
            #print analyzer(each)
            pass
            
    return t_token
def get_text(data,i):
    #global data
    return data[i]['text']#.decode('string_escape')






# Only the text
corpora = []
for i in range(len(data)):
    oneText = str(get_text(data,i))
    #oneText = get_text(i)#, errors='replace'
    corpora.append(oneText)
            
vectorizer = CountVectorizer()
#X = vectorizer.fit_transform(corpora[0:length]).toarray()
vectorizer.fit_transform(corpora[0:length])
analyzer = vectorizer.build_analyzer()
#histogram_x = X.sum(axis=0)

# Get the corresponding number of a certain word
#print(vectorizer.vocabulary_.get('document'))

#wordDict = vectorizer.vocabulary_

# Get the most frequent words
import collections
import math
import random
import zipfile
import urllib
#import tensorflow as tf
def build_dataset(words,vocab_size):
    count = [['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocab_size-1))
    data = []
    for word, freq in count:
        #print(word,freq)
        #dictionary[word] = len(dictionary)
        data.append(word)
    return data
    
# Take 10,000 the most frequent words
high_freq_num = 10000    
        
all_text = []       
for i in range(length):
    all_text += get_vector(data,i)

word_by_high_freq = build_dataset(all_text,high_freq_num)

    

def write_csv(length):
    # Write the high frequency words to csv
    global word_by_high_freq
    global vectorizer
    # Write the index of frequent words
    with open("freq_word.csv","w+",newline="") as freq_file:
         writer = csv.writer(freq_file,delimiter=' ')
         writer.writerow(word_by_high_freq[1:])
    
    def word_of_idx(input_idx):
        for (word, idx) in vectorizer.vocabulary_.items():
            if idx == input_idx:
                return word
            

         
    # Write the freqent words list
    freq_wordslist = []
    for eachIdx in word_by_high_freq:
        if word_of_idx(eachIdx) != None:
            freq_wordslist.append(word_of_idx(eachIdx))
    with open("freq_wordlist.csv","a+",newline='',encoding='utf-8') as csvfile:
        for item in freq_wordslist:
            
            csvfile.write("%s " % item)
            csvfile.write("%d\n" % vectorizer.vocabulary_.get(item))
            #writer.writerow([freq_wordslist])
    
    MB = 2**20
    dir_path = os.path.dirname(os.path.realpath("__file__"))
    allfiles = os.listdir(dir_path)
    for each in allfiles:
        if each.endswith(".csv"):
            #print each
            pass
        #cmd = input("Delete all the csv files, continue? (y/n)\n")
    cmd ='y'
    if cmd == 'y' or cmd == "Y":    
        for each in allfiles:   
            if each.endswith(".csv"):
                os.remove(os.path.join(dir_path, each))
    
    j=1
    #for i in range(len(data)):
    for i in range(length):    
        #text = get_text(i)
        text_token = get_vector(data,i)
    
    # Replace the infrequent words with 0
        for i in range(len(text_token)):
            if text_token[i] not in word_by_high_freq:
                text_token[i] = 0 
    
        file_name = "wiki_"+str(j).zfill(5)+"token.csv"
        if os.path.isfile(file_name):
                #os.remove(file_name)
                if os.stat(file_name).st_size > 1*MB:
                    # If the file is big enough, choose a new file to write
                    j+=1
                    file_name = "wiki_"+str(j).zfill(5)+"token.csv"
        with open(file_name,"a+",newline='') as csvfile:        
            writer = csv.writer(csvfile,delimiter=' ')
            writer.writerow(text_token)
        

# Run write_csv(length) to write the data to csv file




# Example: get the word for a given index
input_idx = 9787
for (word, idx) in vectorizer.vocabulary_.items():
    if idx == input_idx:
        print(word)
        break
    
# Example: get the index for a given word
# word_dict = vectorizer.vocabulary_
# print(word_dict[word])