import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import sklearn
from log_file import phase_1
logger = phase_1("hyp_test")
from scipy.stats import pearsonr

def hypothesis(X_train,Y_train,X_test,Y_test):
    try:
        cor_personr_value = []
        waste_columns = []
        for i in X_train.columns:
            result = pearsonr(X_train[i], Y_train)
            cor_personr_value.append(result)
        cor_personr_value = np.array(cor_personr_value)
        p_value = pd.Series(cor_personr_value[:, 1], index = X_train.columns)
        for j in p_value.index:
            if p_value[j] > 0.05:
                waste_columns.append(j)
                logger.info(f"Targeted Columns : {j} : {p_value[j]}")
        X_train = X_train.drop(waste_columns, axis=1)
        X_test =  X_test.drop(waste_columns, axis=1)
        return  X_train,X_test
    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")
