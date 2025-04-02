import numpy as np
import pandas as pd
import sklearn
import sys
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from log_file import phase_1
logger = phase_1("best_model")


def selecting_best_model(X_train,Y_train,X_test,Y_test):
    try:
        knn_reg = KNeighborsClassifier(n_neighbors=3)
        knn_reg.fit(X_train,Y_train)
        knn_predict = knn_reg.predict(X_test)

        nb_reg = GaussianNB()
        nb_reg.fit(X_train, Y_train)
        nb_predict = nb_reg.predict(X_test)

        LR_reg = LogisticRegression()
        LR_reg.fit(X_train, Y_train)
        LR_predict = LR_reg.predict(X_test)

        DT_reg = DecisionTreeClassifier(criterion='entropy')
        DT_reg.fit(X_train, Y_train)
        DT_predict = DT_reg.predict(X_test)

        RF_reg = RandomForestClassifier(n_estimators=99, criterion='entropy')
        RF_reg.fit(X_train, Y_train)
        RF_predict = RF_reg.predict(X_test)

        fpr_knn, tpr_knn, threshold = roc_curve(Y_test, knn_predict)
        fpr_nb, tpr_nb, threshold = roc_curve(Y_test, nb_predict)
        fpr_LR, tpr_LR, threshold = roc_curve(Y_test, LR_predict)
        fpr_DT, tpr_DT, threshold = roc_curve(Y_test, DT_predict)
        fpr_RF, tpr_RF, threshold = roc_curve(Y_test, RF_predict)

        plt.figure(figsize=(5,2))
        plt.plot([0,1],[0,1], "k--")
        plt.plot(fpr_knn, tpr_knn, color = 'r', label="KNN")
        plt.plot( fpr_nb, tpr_nb, color='b', label="NB")
        plt.plot(fpr_LR, tpr_LR, color='g', label="LR")
        plt.plot(fpr_DT, tpr_DT, color='black', label="DT")
        plt.plot(fpr_RF, tpr_RF, color='y', label="RF")

        plt.legend(loc=0)
        plt.show()
    except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")
