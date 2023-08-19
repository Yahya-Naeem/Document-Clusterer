#!/usr/bin/env python
# coding: utf-8

# In[1]:


#File read [[],[],[]]
#preprocess 
#word2vev
#[[yahya,IR],[]]
#yahya [10,0.5      ]
#tf-idf
#k-mean(n_clusters = 5)
#k-mean.fit(data)
#Print cluster -  
get_ipython().system('pip install pdfminer')


# In[50]:


import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
def remove_metadata(content):
    words = ['Name:','Path:',':-','-:','			','Article-I.D.:','Xref:','Reply-To:','Followup-To:','VERSION 1.1','Supersedes:','','Newsgroups:','Subject:','Message-ID:','From:','Date:','Sender:','Expires:','References:','Organization:','>','<','Keywords:','Summary:','Lines:']
    sw = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    updated_content = []
    for line in content:
        #removing punctuations
        if not any(word in line.split() for word in words):
            tokens = word_tokenize(line)
            #removing sw
            filter_line = ' '.join(stemmer.stem(word) for word in tokens if word not in sw)
            filter_line = filter_line.translate(str.maketrans('','', string.punctuation))
            updated_content.append(filter_line.lower().strip())
    #print(updated_content)
    return '\n'.join(updated_content)


def file_processing():
    directory = './Files'
    out_dir = './Cleaned_files'
    inside_dir = os.listdir(directory)
    for f in inside_dir:
        #print('Name:', f)
        if not f.startswith('.'):
            file_path = os.path.join(directory, f)
            with open(file_path,'r') as handle:
                content = handle.readlines()
                #pre-process content and remove meta data 
                content = remove_metadata(content)
                file = os.path.join(out_dir, f)
                with open(file,'w') as output_file:
                    output_file.write(content)
                    #print('Content writen')    
    print('pre-processing Done')
    
#_______________________________INT-MAIN_______________________________
file_processing()


# In[64]:


from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
import math

def file_processing():
    directory = './Cleaned_files'
    inside_dir = os.listdir(directory)
    All_files = {}
    for f in inside_dir:
        file_path = os.path.join(directory, f)
        with open(file_path, 'r') as handle:
            content = handle.readlines()
            content = [line.rstrip('\n') for line in content]
            All_files[f] = content

    # Forming the inverted index:
    all_words = []
    
    for words in All_files.values():
        for word in words:
            if word not in all_words:
                all_words.append(word)
    
    # Applying Word2Vec
    model = Word2Vec(sentences=[all_words], min_count=1)
    word_vectors = {}
    for word in all_words:
        if word in model.wv:
            word_vectors[word] = np.mean(model.wv[word], axis=0)
        else:
            word_vectors[word] = np.zeros(model.vector_size)
    
    # Making the inverted index for TF-IDF
    inverted_index = {}
    for term in all_words:
        inverted_index[term] = {}
        for doc, words in All_files.items():
            tf = words.count(term)  # Counts the term frequency in the document
            if tf > 0:
                inverted_index[term][doc] = tf

    # Calculate TF-IDF * Word Embedding value and update inverted index
    for term in inverted_index.keys():
        for doc in inverted_index[term].keys():
            inverted_index[term][doc] = (
                word_vectors[term]
                * inverted_index[term][doc]
                * (math.log(30 / len(inverted_index[term].keys())) + 1)
            )

    doc_embeddings = []
    for doc, words in All_files.items():
        embedding = np.zeros(model.vector_size)
        for term in words:
            if term in inverted_index and doc in inverted_index[term]:
                tfidf_embedding = inverted_index[term][doc]
                embedding += tfidf_embedding
        doc_embeddings.append(embedding)

    # Convert document embeddings to an array
    X = np.array(doc_embeddings)

    # Apply k-means clustering
    k = 5  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)

    # Get cluster labels for each document
    cluster_labels = kmeans.labels_

    # Print the document and its assigned cluster label
    for i, (doc, _) in enumerate(All_files.items()):
        print(f"Document '{doc}' is assigned to cluster {cluster_labels[i]}")

file_processing()


# In[65]:


from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
import math

def file_processing():
    directory = './Cleaned_files'
    inside_dir = os.listdir(directory)
    All_files = {}
    for f in inside_dir:
        file_path = os.path.join(directory, f)
        with open(file_path, 'r') as handle:
            content = handle.readlines()
            content = [line.rstrip('\n') for line in content]
            All_files[f] = content

    # Forming the inverted index:
    all_words = []
    sentences = []  # List of sentences for Word2Vec
    
    for words in All_files.values():
        sentence = []
        for word in words:
            if word not in all_words:
                all_words.append(word)
            sentence.append(word)
        sentences.append(sentence)
    
    # Applying Word2Vec
    model = Word2Vec(sentences=sentences, min_count=1)
    word_vectors = {}
    for word in all_words:
        if word in model.wv:
            word_vectors[word] = np.mean(model.wv[word], axis=0)
        else:
            word_vectors[word] = np.zeros(model.vector_size)
    
    # Making the inverted index for TF-IDF
    inverted_index = {}
    for term in all_words:
        inverted_index[term] = {}
        for doc, words in All_files.items():
            tf = words.count(term)  # Counts the term frequency in the document
            if tf > 0:
                inverted_index[term][doc] = tf

    # Calculate TF-IDF * Word Embedding value and update inverted index
    for term in inverted_index.keys():
        for doc in inverted_index[term].keys():
            inverted_index[term][doc] = (
                word_vectors[term]
                * inverted_index[term][doc]
                * (math.log(30 / len(inverted_index[term].keys())) + 1)
            )

    doc_embeddings = []
    for doc, words in All_files.items():
        embedding = np.zeros(model.vector_size)
        for term in words:
            if term in inverted_index and doc in inverted_index[term]:
                tfidf_embedding = inverted_index[term][doc]
                embedding += tfidf_embedding
        doc_embeddings.append(embedding)

    # Convert document embeddings to an array
    X = np.array(doc_embeddings)

    # Apply k-means clustering
    k = 5  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)

    # Get cluster labels for each document
    cluster_labels = kmeans.labels_

    # Print the document and its assigned cluster label
    for i, (doc, _) in enumerate(All_files.items()):
        print(f"Document '{doc}' is assigned to cluster {cluster_labels[i]}")

file_processing()

