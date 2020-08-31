# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 01:44:12 2020

@author: Shufyan
"""
"""Import all the required libraries:"""
import nltk
import os
from nltk.util import ngrams
from collections import Counter
import pandas as pd
import re
import unicodedata
from nltk.corpus import stopwords# add appropriate words that will be ignored in the analysis
ADDITIONAL_STOPWORDS = []

"""Generic function to perform some basic cleaning (if required):"""
def basic_clean(text):
    """
    A simple function to clean up the data. All the words that
    are not designated as a stop word is then lemmatized after
    encoding and basic regex parsing are performed.
    """
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
    text = (unicodedata.normalize('NFKD', text)
      .encode('ascii', 'ignore')
      .decode('utf-8', 'ignore')
      .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]


"""This is the actual function to return ngrams:"""
def extract_ngrams(path, num):   

    corpus = []
    
    for i in next(os.walk(path))[2]:
        if i.endswith('.txt'):
            f = open(os.path.join(path,i))
            corpus.append(f.read())
              
    frequencies = Counter([])
    for text in corpus:
        text = re.sub(r'[^\w\s]', '', text)
        token = nltk.word_tokenize(text)
        token = list(map(str.strip, token))
        ngm = ngrams(token, num)
        frequencies += Counter(ngm)
    
    return frequencies 

if __name__ == '__main__':
    
    """Here you'll perform the actual code to call ngram function:"""
    n = 3
    folderPath = '<Folder_Path_for_Text_Files>'
    outputFilePath = '<Output_Folder_Path_With_CSV_File_Name>'
    df = pd.DataFrame.from_dict(extract_ngrams(folderPath, n).most_common())  
      
    wordColNames = []
    for i in range(n):
        wordColNames.append('word'+str(i+1))
     
    df[wordColNames] = pd.DataFrame(df[0].tolist())
    df = df.drop([0], axis=1)
    df = df.rename(columns={1:'freq'})
    print(df)
    
    df.to_csv(outputFilePath, header=True, index=None, sep=',', mode='w')