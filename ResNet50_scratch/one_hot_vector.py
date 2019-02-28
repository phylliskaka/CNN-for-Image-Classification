#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: phyllis
"""

import numpy as np 

def convert_to_one_hot(label, n):
    """
    input: label is an array like object, shape(m, 1) 
    n is an integer object, equal to the maximum value of label[0] -1 
    output an matrix of dimension (m, n)
    """
    m = label.shape[0]

    one_hot_matrix = np.ndarray((m, n))
    one_hot_vector = np.zeros((n))
    for i in range(m):
        one_hot_vector[int(label[i][0]-1)] = 1
        one_hot_matrix[i] = one_hot_vector
        one_hot_vector = np.zeros((n))
    
    return one_hot_matrix

a = np.ones((30, 1))
x = convert_to_one_hot(a, 3)
        
