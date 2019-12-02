import pandas as pd
import numpy as np
import itertools
import sys
import re
import time
from nltk.corpus import stopwords

def clean_string(s):
    s = s.replace('&amp;',"&")
    cleanr = re.compile('<.*?>')
    re.sub(cleanr, ' ', s)
    s = re.sub(r'[?|#]',r' ',s)
    s = re.sub(r'[,|)|(|\|/]',r' ',s)
    s = re.sub(r'[\n|\r]', r' ', s)
    s = re.sub(r'[.]+(?=[.])', r' ', s)
    s = re.sub(r'[!]+(?=[!])', r' ', s)
    s = s.replace("\'","'")
    s = s.replace('\"',"'")
    return s

def clean_lstm(s):
    s = s.lower()
    stp_wrd = set(stopwords.words('english'))
    clean_num = re.compile('[^0-9a-z #+_]')
    s = clean_num.sub('', s)
    s = ' '.join(word for word in s.split() if word not in stp_wrd) 
    return s

def clean_reviews(df,col_name='review',is_lstm=False):
    ctr = 0
    df_m = df.copy(deep=True)
    rows,_ = df_m.shape
    for row in range(rows):
        ctr = ctr + 1
        per = round((ctr*100)/rows)
        if(round(per)%5==0):
            sys.stdout.write("Removing unwanted string characters from {} column--> {} percentage complete\r"\
                         .format(col_name,round((ctr*100)/rows ,2)))    
            sys.stdout.flush()
        tmp = df_m.iloc[row][col_name].replace('&#039;',"'")
        tmp = tmp.replace('&rsquo;',"'")
        tmp = clean_string(tmp)
        if(is_lstm):
            tmp = clean_lstm(tmp)
        df_m.at[row,'review'] = tmp
    print("Review cleanup Completed...")
    print("Removing row with nan values")
    train_na = df_m.isna().values
    index = np.argwhere(train_na==True)
    index = list(itertools.chain.from_iterable(np.delete(index,1,1).tolist()))
    per = len(index)*100/len(df_m)
    print("Percentage of nan rows in dataset--> {} %".format(round(per,2)))
    df_m = df_m.drop(index)
    print("Removed {} rows with na values".format(len(index)))
    return df_m