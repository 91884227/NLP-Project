#!/usr/bin/env python
# coding: utf-8

# # import tool

# In[18]:


import json
import pickle
from sklearn.model_selection import train_test_split
import sys


# # import self defined tool

# In[19]:


from module.class_label_preprocessing import label_preprocess


# In[21]:


if __name__ == '__main__':
    # SPILT_RATE = 0.1
    SPILT_RATE = float(sys.argv[1])

     # read data
    data = [json.loads(line) for line in open('./data/train_gold.json', 'r')]    

    # deal with data
    data_categories = [i['categories'] for i in data]
    class_label_preprocess = label_preprocess(data_categories)

    y = [class_label_preprocess.encode(i) for i in data_categories]
    X = [(i['text'], i['reply']) for i in data]

    # spilt-data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = SPILT_RATE)

    # For json save
    y_train = [[int(j) for j in i] for i in  y_train] 
    y_test = [[int(j) for j in i] for i in  y_test] 

    # save
    with open('./data/X_train.json', 'w') as outfile:
        json.dump(X_train, outfile)

    with open('./data/X_test.json', 'w') as outfile:
        json.dump(X_test, outfile)

    with open('./data/y_train.json', 'w') as outfile:
        json.dump(y_train, outfile)

    with open('./data/y_test.json', 'w') as outfile:
        json.dump(y_test, outfile)

    # save class
    with open('./module/label_preprocess.pkl', 'wb') as output:
        pickle.dump(class_label_preprocess, output, pickle.HIGHEST_PROTOCOL)

