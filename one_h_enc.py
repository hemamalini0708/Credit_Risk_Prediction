import numpy as np
import pandas as pd
import sklearn
import sys
import logging
import matplotlib.pyplot as plt
from log_file import phase_1
logger = phase_1("one_h_enc")
from sklearn.preprocessing import OneHotEncoder
one_hot_enc = OneHotEncoder()

# to convert columns  categricol to numerical
# It works on Equality Base columns only
def one_hot_cat_num(train_cat,test_cat):
    try:
        one_hot_enc.fit(train_cat[["Gender", "Region"]])
        res = one_hot_enc.transform(train_cat[["Gender", "Region"]]).toarray()

        train_cat[one_hot_enc.get_feature_names_out()[0]] = res[:, 0]
        train_cat[one_hot_enc.get_feature_names_out()[1]] = res[:, 1]
        train_cat[one_hot_enc.get_feature_names_out()[2]] = res[:, 2]
        train_cat[one_hot_enc.get_feature_names_out()[3]] = res[:, 3]
        train_cat[one_hot_enc.get_feature_names_out()[4]] = res[:, 4]
        train_cat[one_hot_enc.get_feature_names_out()[5]] = res[:, 5]
        train_cat[one_hot_enc.get_feature_names_out()[6]] = res[:, 6]

        res_ = one_hot_enc.transform(test_cat[["Gender", "Region"]]).toarray()

        test_cat[one_hot_enc.get_feature_names_out()[0]] = res_[:, 0]
        test_cat[one_hot_enc.get_feature_names_out()[1]] = res_[:, 1]
        test_cat[one_hot_enc.get_feature_names_out()[2]] = res_[:, 2]
        test_cat[one_hot_enc.get_feature_names_out()[3]] = res_[:, 3]
        test_cat[one_hot_enc.get_feature_names_out()[4]] = res_[:, 4]
        test_cat[one_hot_enc.get_feature_names_out()[5]] = res_[:, 5]
        test_cat[one_hot_enc.get_feature_names_out()[6]] = res_[:, 6]

        train_cat = train_cat.drop(["Gender", "Region"], axis=1)
        test_cat = test_cat.drop(["Gender", "Region"], axis=1)
        logger.info(f"Categorical Columns Converted into NUmerical Columns Sucessfully")

        return train_cat,test_cat
    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")





