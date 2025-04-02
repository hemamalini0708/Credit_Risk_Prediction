import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from log_file import phase_1
logger = phase_1("final_testing")

def testing_():
    try:
        model_ = pickle.load(open("Credit_card.pkl",'rb'))
        print(type(model_))
        temp = np.random.random((7,2))
        temp = temp.ravel()
        if model_.predict([temp])[0] == 0:
            return 'Bad Customer'
        else:
            return 'Good Customer'
    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")