#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
class label_preprocess:
    def __init__(self, list_):
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(categories)
    
    def encode(self, list_):
        return( list( self.mlb.transform([ list_ ])[0] ) )
    
    def decode(self, list_):
        buf = self.mlb.inverse_transform(np.array(list_).reshape(1, len(self.mlb.classes_)))[0]
        return( buf )

