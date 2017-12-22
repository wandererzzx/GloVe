import numpy as np
import os
import csv
import time
# read csv file and return a list with each row in the csv file as an element
def read_csv(csv_path,delimiter=' '):
	'''
	csv_path: preprocessed csv file path
	'''
	with open(csv_path) as csvfile:
		reader = csv.reader(csvfile,delimiter=delimiter)
		data = list(reader)
	return data


# count the co_occurrance of words in single article
def co_occurrance_count(article,freq_dict,co_occurMat,window_size=10):
	'''
	article: a list of word integers in single article
	freq_dict: a list indicates the most frequent words
	co_occurMat: word-word co-occurrence matrix
	window_size: the size of context window
	'''
	length = len(article)
	for i in range(length):  # iterate through every word in article
		target = article[i]
		context = []
		if target in freq_dict: # if the word is frequent word, get its index in freq_dict
			ind_tar = freq_dict.index(target)

			# find the context of our target word
			if i < window_size: 
				context_left = article[0:i]
				if i+1+window_size < length:
					context_right = article[i+1:i+1+window_size]
				else:
					context_right = article[i+1:]

			else:
				context_left = article[i-window_size:i]
				if i+1+window_size < length:
					context_right = article[i+1:i+1+window_size]
				else:
					context_right = article[i+1:]

			# populate co_occurMat with 1/distance to target word
			for count,context_word in enumerate(context_left):
				if context_word in freq_dict:
					ind_con = freq_dict.index(context_word)
					co_occurMat[ind_tar,ind_con] += 1.0/np.float32(len(context_left)-count)
					#co_occurMat[ind_tar,ind_con] += 1
			for count,context_word in enumerate(context_right):
				if context_word in freq_dict:
					ind_con = freq_dict.index(context_word)
					co_occurMat[ind_tar,ind_con] += 1.0/np.float32(count+1)
					#co_occurMat[ind_tar,ind_con] += 1


def convert(data_dir,dict_source):
	freq_dict = read_csv(dict_source)[0]
	size = len(freq_dict)

	print('dict_size:{}'.format(size))
	print('dict_samples:{}'.format(freq_dict[0:10]))

	# initialize co_occurrance matrix
	co_occurMat = np.zeros((size,size))

	file_num = len(os.listdir(data_dir))
	# read preprocessed article csv
	for number,name in enumerate(os.listdir(data_dir)):
		wiki = data_dir + name
		articles = read_csv(wiki)

		num_article_per_file = len(articles)
		print('counting file {}/{}:'.format(number+1,file_num))
		print('number of articles:{}'.format(num_article_per_file))

		# count each article
		for article in articles:
			co_occurrance_count(article,freq_dict,co_occurMat,window_size=10)

		del articles # prevent OOM error
	return co_occurMat

if __name__ == '__main__':

	# directory of preprocessed csv file wiki_00,wiki_01...
	data_dir = os.path.abspath('.')
	# frequent word file 
	dict_source = os.path.abspath('.') + '\\freq_word.csv'
	# the file storing output co-occurrence matrix
	save_file = os.path.abspath('.') + '\\CPU_co_occurMat.csv'

	start = time.time()
	co_occurMat = convert(data_dir,dict_source)
	time_spent = time.time() - start

	print(co_occurMat)
	print(time_spent)

	# save the co_occurMat to csv file
	np.savetxt(save_file,co_occurMat,delimiter=',')



