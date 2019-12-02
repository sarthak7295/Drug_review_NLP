#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


# In[23]:


# mystr = "After a much needed pedicure, I noticed my big toe was starting to look sort of discolored/grayish at the cuticle. A few months passed and it became worse. The toenail thickened and became flaky. I would sometimes experience excruciating pain and a throbbing sensation in my toe along with a reddened outer area. My physician prescribed Terbinafine after an initial liver function test. This test was repeated 1 month after treatment, three months after that, and since my results have been fine, he has ordered no more repeats. I am about to start my 5th month (out of 6) and am already seeing great results. It\'s a slow process (he said it could take a year) but my new nail is coming in great!"


# In[25]:


# summ = summarize(mystr)

# print(mystr)
# print()
# print()
# print(summ)


# In[5]:


def summarize(review):
    
    freq_table = make_freq_table(review)
    
    sent_scores = calculate_sentence_weights(review, freq_table)
    
    summary = extract_summary(review, sent_scores)
    
    return summary


# In[2]:


def make_freq_table(review):
    
    stem = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    words = nltk.word_tokenize(review)
    
    freq_table = {}
    for word in words:
        new_word = stem.stem(word)
#         new_word = word
        if(new_word in stop_words):
            continue

        if(new_word in freq_table):
            freq_table[new_word] += 1
        else:
            freq_table[new_word] = 1

#     print(freq_table)

#     print([w for w in freq_table])

#     print("\n")

#     sents = nltk.sent_tokenize(mystr)

#     sents[0][:7]
    
    return freq_table


# In[3]:


def calculate_sentence_weights(review, freq_table):
    
    sent_scores = {}
    sentences = nltk.sent_tokenize(review)

    for sent in sentences:
        sent_wc = len(nltk.word_tokenize(sent))
        sent_wc_wo_stopwords = 0

        for word_weight in freq_table:
            if word_weight in sent.lower():
                sent_wc_wo_stopwords += 1
                if sent[:7] in sent_scores:
                    sent_scores[sent[:7]] += freq_table[word_weight]
                else:
                    sent_scores[sent[:7]] = freq_table[word_weight]
        try:
            sent_scores[sent[:7]] /= sent_wc_wo_stopwords
        except:
            tmp = 0

#     print(sent_scores)

    return sent_scores


# In[4]:


def extract_summary(review, sent_scores):
    
    sentences = nltk.sent_tokenize(review)
    
    sum_values = 0

    for entry in sent_scores:
        sum_values += sent_scores[entry]

    # Getting sentence average value from source text
    if len(sent_scores)!=0:
        average_score = (sum_values / len(sent_scores))
    else:
        average_score = 0

    sentence_ctr = 0
    summary = ''

    for sentence in sentences:
        if((sentence[:7] in sent_scores) and (sent_scores[sentence[:7]] >= (1*average_score))):
            summary += ' ' + sentence
            sentence_ctr += 1

#     print(summary)

    return summary


# In[ ]:




