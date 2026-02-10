
# Machine Learning AMAPE Heart Disease Analysis
# Author: Joshua Emmons

# VERS 2.5 DTD 08 FEB 2026

import pandas as pd                              # needed to read the data
import numpy as np
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler       # standardize data
from sklearn.linear_model import Perceptron            # the algorithm
from sklearn.metrics import accuracy_score             # grade the results
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression    # the algorithm
from sklearn.svm import SVC                          # the algorithm
from sklearn.tree import DecisionTreeClassifier         # the algorithm
from sklearn.ensemble import RandomForestClassifier    # the algorithm
from sklearn.neighbors import KNeighborsClassifier     # the algorithm

heart_database = 'heart_database.db'
heart_table = 'heart_table'
results = 'Results.db'


#set show all rows
pd.set_option('display.max_rows', None)

# Load and Read the Data from Github Repos
data_url = "https://raw.githubusercontent.com/Jemmons-EE/heart_analysis/refs/heads/main/heart1.csv"


try:
    heart_data = pd.read_csv(data_url)

    print("Dataframe: heart1.csv Loaded")

except Exception as e:
    print(f"An error occurred: {e}")

# # # Train-test split
# define datasets
x = heart_data.iloc[:, :-1].values
y = heart_data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)

#Transform for train and test data
sc = StandardScaler()                    # create the standard scalar
sc.fit(X_train)                          # compute the required transformation
X_train_std = sc.transform(X_train)      # apply to the training data
X_test_std = sc.transform(X_test)        # and SAME transformation of test data

#Transform for Combined Accuracy
sc.fit(x)                          # compute the required transformation
x_combined = sc.transform(x)      # apply to the combined



#Perceptron
perc = 'Perceptron'
ppn = Perceptron(max_iter=7, tol=1e-3, eta0=0.001,
                 fit_intercept=True, random_state=0, verbose=0)
ppn.fit(X_train_std, y_train)              # do the training


y_pred = ppn.predict(X_test_std)           # now try with the test data
y_comb_pred = ppn.predict(x_combined)       # pred combined


print( 'Perceptron Results:')
print( '  Accuracy: %.2f' % accuracy_score(y_test, y_pred))

print( '  Combined Accuracy: %.2f' % accuracy_score(y, y_comb_pred),'\n')
perc_acc = accuracy_score(y_test, y_pred)
perc_acc_comb = accuracy_score(y, y_comb_pred)


# #Logistic Regression
logr = 'Logistic Regression'
lr = OneVsRestClassifier(LogisticRegression(C=10, solver='liblinear', \
                            random_state=0) )
lr.fit(X_train_std, y_train)         # apply the algorithm to training data    
y_predlr = lr.predict(X_test_std)

lrcomb =  OneVsRestClassifier(LogisticRegression(C=10, solver='liblinear', \
                            random_state=0) )
lrcomb.fit(x_combined, y)         # apply the algorithm to training data
y_predlr_comb = lr.predict(x_combined)

logr_acc = accuracy_score(y_test, y_predlr)
logr_acc_comb = accuracy_score(y, y_predlr_comb)

print( 'Logistic Regression Results:')
print( '  Accuracy: %.2f' % logr_acc)
print( '  Combined Accuracy: %.2f' % logr_acc_comb,'\n')  


# # Support Vector Machine
svm_name = 'Support Vector Machine'
svm = SVC(kernel="poly", degree=3, C=1.15, random_state=0)
svm.fit(X_train_std, y_train)                      # do the training

y_pred_svm = svm.predict(X_test_std)                   # work on the test data
# combine the train and test sets
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

 # and analyze the combined sets
y_combined_pred_svm = svm.predict(X_combined_std)

print('Support Vector Machine Results:')
print('  Accuracy: %.2f' % accuracy_score(y_test, y_pred_svm))
print('  Combined Accuracy: %.2f' % \
    accuracy_score(y_combined, y_combined_pred_svm),'\n')

svm_acc = accuracy_score(y_test, y_pred_svm)
svm_acc_comb = accuracy_score(y, y_combined_pred_svm)


#Decision Tree Learning
dec_tree = 'Decision Tree'
tree = DecisionTreeClassifier(criterion='entropy',max_depth=9,random_state=0)
tree.fit(X_train,y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

y_tree_pred = tree.predict(X_test)
# see how we do on the combined data
y_combtree_pred = tree.predict(X_combined)

dec_acc = accuracy_score(y_test, y_tree_pred)
dec_tree_comb = accuracy_score(y_combined, y_combtree_pred) 

print('Decision Tree Learning Results:')
print('  Accuracy: %.2f' % dec_acc)
print('  Combined Accuracy: %.2f' % dec_tree_comb,'\n')


# #Random Forest
rand_for = 'Random Forest'
forest = RandomForestClassifier(criterion='entropy', n_estimators=29, \
                                    random_state=1, n_jobs=4)
forest.fit(X_train,y_train)

y_for_pred= forest.predict(X_test)         # see how we do on the test data

y_combfor_pred = forest.predict(X_combined)

rand_for_acc = accuracy_score(y_test, y_for_pred)
rand_for_acc_comb = accuracy_score(y_combined, y_combfor_pred)

print('Random Forest Learning Results:')
print('  Accuracy: %.2f' % rand_for_acc)
print('  Combined Accuracy: %.2f' % rand_for_acc_comb,'\n')


# # K Nearest Neighbor
knn_name = 'K Nearest Neighbor'
knn = KNeighborsClassifier(n_neighbors=33,p=2,metric='minkowski')
knn.fit(X_train_std,y_train)

# run on the test data and print results and check accuracy
y_knn_pred = knn.predict(X_test_std)
y_combknn_pred = knn.predict(X_combined_std)

knn_acc = accuracy_score(y_test, y_knn_pred)
knn_acc_comb = accuracy_score(y_combined, y_combknn_pred)

print('K-Nearest Neighbor Learning Results:')
print('  Accuracy: %.2f' % knn_acc)
print('  Combined Accuracy: %.2f' % knn_acc_comb ,'\n')

# SQL update
try:
    connec =  sqlite3.connect(heart_database)
    print(f"Connected to {heart_database}")

    heart_data.to_sql(heart_table, connec, if_exists='replace', index=False)
    print(f"Successfully wrote data to table '{heart_table}'")

    connec.close()
    print("Database connection closed.")
    print(f"Convert to SQL Database completed: Verify database file: {heart_database}")

except FileNotFoundError:
    print(f"Error: The file '{heart_data}' was not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")


#SQL Query
try:
    connec =  sqlite3.connect(results)
    cursor = connec.cursor()
    connec.commit()
    print(f"Last Row ID: {cursor.lastrowid}")

    new_data = (perc ,perc_acc, perc_acc_comb)
    new_data2 = (logr,logr_acc,logr_acc_comb)
    new_data3 = (svm_name,svm_acc,svm_acc_comb)
    new_data4 = (dec_tree,dec_acc,dec_tree_comb)
    new_data5 = (rand_for,rand_for_acc,rand_for_acc_comb)
    new_data6 = (knn_name,knn_acc,knn_acc_comb)

    print(f'Made New Data Lines')

    #create Table
    create_result_query = '''
        CREATE TABLE IF NOT EXISTS Results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine TEXT NOT NULL,
            accuracy DECIMAL (3, 2),
            combined_accuracy DECIMAL (3, 2)
        );
        '''
    cursor.execute(create_result_query)

    sql_insert =  '''
        INSERT INTO Results (machine, accuracy, combined_accuracy) VALUES (?, ?, ?);
        '''
    
    print(f'Adding New Data Lines to Results')
    cursor.execute(sql_insert,new_data)
    cursor.execute(sql_insert,new_data2)
    cursor.execute(sql_insert,new_data3)
    cursor.execute(sql_insert,new_data4)
    cursor.execute(sql_insert,new_data5)
    cursor.execute(sql_insert,new_data6)

    connec.commit()
    print(f"Row added ID: {cursor.lastrowid}")

    connec.close()

except sqlite3.Error as e:
    print(f"A database error occurred: {e}")
except Exception as e:
    print(f"An error occurred: {e}")   
                     

#Send SQL as CSV for Tableau
try:
    connec = sqlite3.connect('Results.db')
    conv_query = pd.read_sql_query("SELECT * FROM Results", connec)
    conv_query.to_csv('Results_transport.csv', index=False)
    connec.close()
    print("Sent Results to CSV")
except sqlite3.Error as e:
    print(f"An error occurred: {e}")
                                