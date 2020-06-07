#!/usr/bin/env python
# coding: utf-8

# # import tool

# In[11]:


import os
import numpy as np
import pandas as pd

import random
import json
from tqdm import tqdm
import datetime
from statistics import mean 
import pickle
import sys

from IPython.display import clear_output
# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

# bert
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
clear_output()


# # import self define module

# In[2]:


from module.class_label_preprocessing import label_preprocess


# # PARAMETER

# In[4]:


def create_data_loader(X, y, batch_size_):
    X_text = [i[0] for i in X]
    X_reply = [i[1] for i in X]
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    buf = [tokenizer.encode_plus(i[0], i[1], do_lower_case = False, add_special_tokens = True, max_length = MAX_LENGTH, pad_to_max_length = True) for i in tqdm(X)]   
    input_ids = torch.LongTensor( [i['input_ids'] for i in buf] )
    token_type_ids = torch.LongTensor( [i['token_type_ids'] for i in buf])
    attention_mask = torch.LongTensor( [i['attention_mask'] for i in buf])

    label = torch.FloatTensor(y)

    dataset = TensorDataset(input_ids, token_type_ids, attention_mask, label)
    loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size_, shuffle = True)

    return(loader)


# In[5]:


def acc_for_update(data_loader):
    buf = []
    with torch.no_grad():  
        for data in data_loader:
            input_ids, token_type_ids, attention_mask, labels = [t.to(DEVICE) for t in data]

            outputs = model(input_ids = input_ids, 
                                token_type_ids = token_type_ids, 
                                attention_mask = attention_mask) 

            predict_prop = list(outputs[0].cpu().detach().numpy()[0])
            X = list(np.arange(len(predict_prop)))
            X.sort(key=dict(zip(list(X), list(predict_prop))).get, reverse=True)

            predict_label = X[:6]

            true_label = list(np.where(data[3][0].numpy() == 1)[0])
            buf = buf + [ [i in predict_label for i in true_label] ]
            
    return( mean([mean(i) for i in buf]) )


# In[3]:


# PRETRAINED_MODEL_NAME = "bert-base-cased" # https://huggingface.co/transformers/pretrained_models.html
# BATCH_SIZE = 2
# EPOCHS = 5
# DEVICE = "cuda: 0"
# ifLIMIT = True
# MAX_LENGTH = 50
# ID = "test"
# LEARNING_RATE = 1e-5

if __name__ == '__main__':
    PRETRAINED_MODEL_NAME = sys.argv[1]
    BATCH_SIZE = int( sys.argv[2])
    EPOCHS = int( sys.argv[3])
    DEVICE =  sys.argv[4]
    ifLIMIT = int( sys.argv[5])
    MAX_LENGTH = int( sys.argv[6])
    ID =  sys.argv[7]
    LEARNING_RATE = float( sys.argv[8])

    print("load data...")
    with open('./data/X_train.json') as json_file:
        X_train = json.load(json_file)
    with open('./data/y_train.json') as json_file:
        y_train = json.load(json_file)
    with open('./data/X_test.json') as json_file:
        X_test = json.load(json_file)
    with open('./data/y_test.json') as json_file:
        y_test = json.load(json_file)

    if( ifLIMIT ):
        X_train = X_train[:100]
        y_train = y_train[:100]
        X_test = X_test[:100]
        y_test = y_test[:100]

    with open('./module/label_preprocess.pkl' , 'rb') as input:
        label_preprocessing = pickle.load(input)

    print("create data loader...")
    train_loader = create_data_loader(X_train, y_train, batch_size_ = BATCH_SIZE)
    train_loader_1 = create_data_loader(X_train, y_train, batch_size_ = 1)
    test_loader = create_data_loader(X_test, y_test, batch_size_ = 1)

    print("create model...")
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, 
                                                          num_labels = len(y_train[0]))
    model.to(DEVICE)
    clear_output()

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    loss_fn = nn.MSELoss(reduction='sum')

    print("start training...")

    state_of_the_art = 0
    for epoch in range(EPOCHS):
        running_loss = 0.0

        for data in train_loader:    
            input_ids, token_type_ids, attention_mask, labels = [t.to(DEVICE) for t in data]

            optimizer.zero_grad()

             # forward pass
            outputs = model(input_ids = input_ids, 
                            token_type_ids = token_type_ids, 
                            attention_mask = attention_mask)          

            # loss
            loss = loss_fn(outputs[0], labels)
            # loss = loss_fn(torch.sigmoid(buf1), buf2)

            loss.backward()
            optimizer.step()
            running_loss = running_loss + loss.item()

        print("\n===EPOCH %d/%d==="% (epoch+1, EPOCHS))
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("running_loss: %.4f" %  running_loss)
        print("evaluation score for training set: %.4f" % acc_for_update(train_loader_1))

        buf = acc_for_update(test_loader)
        print("evaluation score for testing set: %.4f" % buf)

        if( buf > state_of_the_art):
            save_directory = "%s_model_%d" % (ID, buf*10000)
            os.chdir("./model_save") 
            os.mkdir(save_directory)
            model.save_pretrained(save_directory)
            os.chdir("..")

            state_of_the_art = buf
            print("save model : %s" % save_directory)

