"""
In this file we are going to load the data and call functions for model development
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import pickle
import sklearn
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
import logging
from log_file import phase_1
logger = phase_1('main')
from ramdom_sample import missing_handling_random
from sklearn.preprocessing import OneHotEncoder
one_hot_enc = OneHotEncoder()
from one_h_enc import one_hot_cat_num
from sklearn.preprocessing import OrdinalEncoder
odinal_enc_ = OrdinalEncoder()
from odinal_enc import cat_num_odinal_enc
from sklearn.feature_selection import VarianceThreshold
quansi_con = VarianceThreshold(threshold=0.1)
from Feature_Selection_Area import constant
from scipy.stats import pearsonr
from hyp_test import hypothesis
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from models import multi_models
from best_model import selecting_best_model
from final_file import f_m
from final_testing import testing_



class SERVICES:
    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info(f"Data loaded successfully")

            # To check How many rows and Columns are there in this data
            logger.info(f"The No.of Columns : {self.df.shape[0]}, The No.of Rows : {self.df.shape[1]}")

            # Checking if the data contains any missing values:
            # Identify whether there are any null values in the dataset.
            # Count the number of null values in each row and each column.
            self.col_with_nullvalues = []
            for i in self.df.columns:
                if self.df[i].isnull().sum() > 0:
                    self.col_with_nullvalues.append(i)
            logger.warning(f"Columns with null values : {self.col_with_nullvalues}")

            # In last two rows the data was miss matched for that all the data showing as null values
            # so for that removing last two rows
            self.df = self.df.drop([150000, 150001], axis=0)
            self.col_with_null_values = [i for i in self.df if self.df[i].isnull().sum() > 0]
            logger.warning(f"Columns with null values : {self.col_with_null_values}")

            # So, we have found null values in the following columns: **[MonthlyIncome, MonthlyIncome.1, NumberOfDependents]
            # I have a doubt about whether **'MonthlyIncome'** and **'MonthlyIncome.1'** are the same columns.
            # So, I am comparing both columns to check if they are the same.
            # If both columns have the same values, remove one of them because it is a duplicate.
            self.c = 0
            for j in self.df.index:
                # If a null value matches another null value, leave it as it is → PASS
                if np.isnan(self.df["MonthlyIncome"][j]) == np.isnan(self.df["MonthlyIncome.1"][j]):
                    pass
                # If an index value matches another index value, leave it as it is → PASS
                elif self.df["MonthlyIncome"][j] == self.df["MonthlyIncome.1"][j]:
                    pass
                else:
                # If the above two conditions are not satisfied, then increase the **C** value.
                    self.c = self.c+1

            if self.c == 0:
                logger.info(f"Both Columns Having The same data : [MonthlyIncome] : [MonthlyIncome.1]")
            else:
                logger.info(f"Both columns not having same data : [MonthlyIncome] : [MonthlyIncome.1] ")

            # If both columns have the same data, drop one of them because it is a duplicate.
            self.df = self.df.drop(["MonthlyIncome.1"], axis=1)

            # TRANING AND TESTING
            self.X = self.df.iloc[:, :-1] #Independent
            self.Y = self.df.iloc[:, -1] # dependent
            self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.X,self.Y,test_size=0.2,random_state=42)
            logger.info(f"Size Of Traning Data : {len(self.X_train)}, {len(self.Y_train)}")
            logger.info(f"Size Of Testing Data : {len(self.X_test)}, {len(self.Y_test)}")

        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")

            # After identifying the columns with null values,
            # we need to check whether both columns have a numerical data type or not.
    def missing_data(self):
        try:
            logger.info(f"Dtype of MonthlyIncome : {self.df["MonthlyIncome"].dtype}")
            logger.info(f"Dtype of NumberOfDependents : {self.df["NumberOfDependents"].dtype}")

            # NumberOfDependents columns is in string format, so I am converting it into a numerical format.
            col_with_null_values_ = ["MonthlyIncome", "NumberOfDependents"]
            for i in col_with_null_values_:
                if self.df[i].dtype == np.float64:
                    pass
                elif self.df[i].dtype == object:
                    self.X_train[i] = pd.to_numeric(self.X_train[i])
                    self.X_test[i] = pd.to_numeric(self.X_test[i])
            logger.info(f"Dtype of NumberOfDependents : {self.X_train["NumberOfDependents"].dtype}")
            logger.info(f"Dtype of NumberOfDependents column converted into numerical sucessfully")

            # After converting the text column into a numerical column...
            # ***************************HANDLING NULL VALUES *************
            # We know that these two columns **[NumberOfDependents, MonthlyIncome]** have null values, right?
            # So now, we need to handle those null values using the best technique: **Random Sample Technique**.
            # We need to create a file named **random_sample.py**.
            # After creating the file **random_sample.py**, we need to access it in the **main.py** file.
            self.X_train,self.X_test =  missing_handling_random(self.X_train,self.X_test)
            logger.info(f"Null Values in Trainig Data : {self.X_train.isnull().sum()}")
            logger.info(f"Null Values in Testing Data : {self.X_test.isnull().sum()}")
            # Using **random_sample**, we removed null values replaced them with
            # new columns: [MonthlyIncome_random_sample]&[NumberOfDependents_random_sample]

        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")

            # ******----------------------------VARIABLE TRANSFORMATION*********--------------------------------------
            # VARIABLE TRANSFORMATION ---> using Log Techinque
            # it coverts all columns to log and returning to the same columns
    def Variable_tf(self,train_num,test_num):
        try:
            for k in train_num.columns:
                train_num[k] = np.log(train_num[k]+1)
                test_num[k] = np.log(test_num[k]+1)
                return  train_num,test_num
            logger.info(f"All Columns Converted into Log Level : ")
        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

            #*******------------------------OUTLIER HANDLING*********--------------------------------------------------
            # The data points which are far away from the remaning data points is called outliers
            # If we have any outliers in our data we need to handle it or remove it
            # The best Technique is (5TH Quantile & 95TH Quantile)

    def outlier_handling(self,train_num,test_num):
        try:
            for i in train_num.columns:
                upper = train_num[i].quantile(0.95)
                lower = train_num[i].quantile(0.05)
                train_num[i] = np.where(train_num[i] > upper,upper,
                                        np.where(train_num[i] < lower, lower,train_num[i]))
                test_num[i] = np.where(test_num[i] > upper,upper,
                                       np.where(test_num[i] < lower, lower,test_num[i]))
                return train_num,test_num
        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")

            # *******----------------CATEGORICAL-NUMERICAL***********---------------------------------------------------
            # Checking if there are any categorical columns.
            # If found, keep only the categorical columns and remove the numerical columns.
    def cat_num(self):
        try:
            # Removing Numberical Columns
            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            # Keeping Categorical Columns
            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')
            logger.info(f"Catgeorical column from the data : {self.X_train_cat.columns}")
            logger.info(f"Categorical Columns from the data  : {self.X_test_cat.columns}")

            #Since we have both nominal and odinal encodings
            # to convert categorical to numerical columns on Equality Base --|
            # OneHotEncoding----> Accessing from one_hot_encoding file
            self.X_train_cat,self.X_test_cat = one_hot_cat_num(self.X_train_cat,self.X_test_cat)

            # OdinalEncoding----> Accessing from OdinalEncoding file
            self.X_train_cat, self.X_test_cat = cat_num_odinal_enc(self.X_train_cat, self.X_test_cat)

            # Acessing that Variable_tf function to here
            self.X_train_num,self.X_test_num = self.Variable_tf(self.X_train_num,self.X_test_num)

            # Acessing that  outlier_handling function to here
            self.X_train_num,self.X_test_num = self.outlier_handling(self.X_train_num,self.X_test_num)
            logger.info(f"Outliers handled from the training and testing data successfully")
            logger.info(f"Feature Engineering Completed ")

            # Know we have to Combine Both Categorical & Numerical Columns by using Concatination
            self.training_ind_data = pd.concat([self.X_train_num,self.X_train_cat], axis=1)
            self.testing_ind_data = pd.concat([self.X_test_num,self.X_test_cat], axis=1)
            logger.info(f"Concatinated Both columns  Successfully")

            # To know the size of training_data testing_data
            logger.info(f"Size Of the Traning Data : {self.training_ind_data.shape}")
            logger.info(f"Size Of The Testing Data : {self.testing_ind_data.shape}")

            # accessing Feature_Selection_Area file to main file
            cols = constant(self.training_ind_data,self.testing_ind_data)
            logger.info(f"Columns with 0.1 variance are : {cols}")
            self.training_ind_data = self.training_ind_data.drop(cols, axis=1)
            self.testing_ind_data = self.testing_ind_data.drop(cols, axis=1)
            logger.info(f"After Removing Some Unwanted Columns there shape are : {self.training_ind_data.shape}")
            # we have done only for X_train and test -> Independents columns
            # know we are working with Y_train and test -> dependent columns
            # convert columns to 0 and 1 -> dependent columns
            self.Y_train = self.Y_train.map({'Good': 1, 'Bad': 0}).astype(int)
            self.Y_test = self.Y_test.map({'Good': 1, 'Bad': 0}).astype(int)
            # print(self.Y_train.head(10))
            # print(self.Y_test.head(10))
            logger.info(f"Converted Column values Good - 1 and Bad - 0 successfully")
            # HYPOTHESIS TEST
            # The purpose of hypothesis testing is to test whether
            # the null hypothesis (there is no difference, no effect) can be rejected or approved.
            self.training_ind_data,self.testing_ind_data = hypothesis(self.training_ind_data,self.Y_train,self.testing_ind_data,self.Y_test)
            # After removing unwanted columns using h_t to know shape of columns
            logger.info(f"After Removing unwanted columns using Hypothesisi testing  from train data : {self.training_ind_data.shape}")
            logger.info(f"After removing unwanted columns using hypothesis testing from test data: {self.testing_ind_data.shape}")
            return self.training_ind_data,self.testing_ind_data,self.Y_train,self.Y_test
        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")


            #******-----------------------DATA-BALANCING************----------------------------------------------------
    def data_Balancing(self):
        try:
            self.training_ind_data, self.testing_ind_data, self.Y_train, self.Y_test = self.cat_num()
            logger.info("Before Upsampling Technique")

            logger.info(f"No Of Rows for 0 : {sum(self.Y_train == 0)}")
            logger.info(f"No Of Rows for 1 : {sum(self.Y_train == 1)}")

            sm = SMOTE(random_state=2)
            self.training_ind_data_up,self.Y_train_up = sm.fit_resample(self.training_ind_data,self.Y_train)
            logger.info("After Upsampling Technique")

            logger.info(f"No Of Rows for 0 : {sum(self.Y_train_up == 0)}")
            logger.info(f"No Of Rows for 1 : {sum(self.Y_train_up == 1)}")
            return self.training_ind_data_up,self.Y_train_up,self.testing_ind_data,self.Y_test

        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")


            #*******---------------------------SCALING_DATA******------------------------------------------------------
    def scaling_data(self):
        try:
            self.training_ind_data_up, self.Y_train_up, self.testing_ind_data, self.Y_test = self.data_Balancing()
            logger.info(f"info : {self.training_ind_data_up.head(3)}")
            sc = StandardScaler()
            sc.fit(self.training_ind_data_up)
            self.scaled_training_inde_cols = sc.transform(self.training_ind_data_up)
            self.scaled_test_inde_cols = sc.transform(self.testing_ind_data)
            logger.info(f"info : {self.scaled_training_inde_cols[:]}")
            #multi_models(self.scaled_training_inde_cols,self.Y_train_up,self.scaled_test_inde_cols,self.Y_test)
            #selecting_best_model(self.scaled_training_inde_cols, self.Y_train_up, self.scaled_test_inde_cols, self.Y_test)
            f_m(self.scaled_training_inde_cols, self.Y_train_up, self.scaled_test_inde_cols, self.Y_test)
            outcomes = testing_()
            logger.info(f"Model prediction  : {outcomes}")
            logger.info(f"columns names : {self.training_ind_data_up.columns}")

        except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")


if __name__=="__main__":
    try:
        path = "creditcard.csv"
        obj = SERVICES(path)
        obj.missing_data()
        obj.scaling_data()
    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")

