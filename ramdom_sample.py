import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import sys
import logging
from log_file import phase_1
logger = phase_1("ramdom_sample")

def missing_handling_random(X_train,X_test):
    try:
        var = ["MonthlyIncome", "NumberOfDependents"]
        for i in var:
            X_train[i+"_random_sample"] = X_train[i].copy() # shallow copy
            sample_ = X_train[i].dropna().sample(X_train[i].isnull().sum(), random_state=42)
            sample_.index = X_train[X_train[i].isnull()].index
            X_train.loc[X_train[i].isnull(), i+"_random_sample"] = sample_

            X_test[i+"_random_sample"] = X_test[i].copy()
            sample_ = X_test[i].dropna().sample(X_test[i].isnull().sum(), random_state=42)
            sample_.index = X_test[X_test[i].isnull()].index
            X_test.loc[X_test[i].isnull(), i+"_random_sample"] = sample_

            X_train = X_train.drop([i], axis=1)
            X_test = X_test.drop([i], axis=1)

        return  X_train,X_test

    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")










