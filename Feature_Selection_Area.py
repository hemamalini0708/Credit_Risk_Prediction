import numpy as np
import pandas as pd
import sklearn
import sys
import logging
import matplotlib.pyplot as plt
from log_file import phase_1
logger = phase_1("Feature_Selection_Area")
from sklearn.feature_selection import VarianceThreshold
quansi_con = VarianceThreshold(threshold=0.1)


def constant(train_data,test_data):
     try:
         quansi_con.fit(train_data)
         res = train_data.columns[~quansi_con.get_support()]
         logger.info(f"Columns with 0.1 variance are : {res} ")
         return res
     except Exception as e:
         er_type, er_lineno, er_msg = sys.exc_info()
         logger.error(f"Error from lone no :  {er_lineno.tb_lineno} Issue : {er_msg}")