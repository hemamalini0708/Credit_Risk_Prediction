import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import OrdinalEncoder
odinal_enc_ = OrdinalEncoder()
import sys
import logging
from log_file import phase_1
logger = phase_1("odinal_enc")


def cat_num_odinal_enc(train_set,test_set):
    try:
        odinal_enc_.fit(train_set[['Rented_OwnHouse', 'Occupation', 'Education']])
        sol = odinal_enc_.transform(train_set[['Rented_OwnHouse', 'Occupation', 'Education']])
        train_set[odinal_enc_.get_feature_names_out()[0]] = sol[:, 0]
        train_set[odinal_enc_.get_feature_names_out()[1]] = sol[:, 1]
        train_set[odinal_enc_.get_feature_names_out()[2]] = sol[:, 2]

        sol_ = odinal_enc_.transform(test_set[['Rented_OwnHouse', 'Occupation', 'Education']])
        test_set[odinal_enc_.get_feature_names_out()[0]] = sol_[:, 0]
        test_set[odinal_enc_.get_feature_names_out()[1]] = sol_[:, 1]
        test_set[odinal_enc_.get_feature_names_out()[2]] = sol_[:, 2]
        logger.info(f"Odinal Features converted to numerical successfully")
        return train_set,test_set

    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")