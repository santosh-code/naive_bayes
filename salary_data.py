import pandas as pd
import numpy as np
train=pd.read_csv("C:/Users/USER/Desktop/naive_bayes/SalaryData_Train.csv")
test=pd.read_csv("C:/Users/USER/Desktop/naive_bayes/SalaryData_Test.csv")


dum1=pd.get_dummies(train.sex)
dum1_1=pd.get_dummies(test.sex)

dum2=pd.get_dummies(train.native)
dum2_2=pd.get_dummies(test.native)

dum3=pd.get_dummies(train.workclass)
dum3_3=pd.get_dummies(test.workclass)

dum4=pd.get_dummies(train.education)
dum4_4=pd.get_dummies(test.education)

dum5=pd.get_dummies(train.Salary)
dum5_5=pd.get_dummies(test.Salary)

train_merge=pd.concat([dum1,dum2,dum3,dum4,train.age,train.hoursperweek,train.capitalloss,train.capitalgain,dum5],axis='columns')
train_merge.columns

test_merge=pd.concat([dum1_1,dum2_2,dum3_3,dum4_4,test.age,test.hoursperweek,test.capitalloss,test.capitalgain,dum5_5],axis='columns')
test_merge.columns

x_train=train_merge.iloc[0:30161,0:69]
y_train=train_merge.iloc[0:30161,70]

x_test=test_merge.iloc[0:15060,0:69]
y_test=test_merge.iloc[0:15060,70]

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)
pred=gnb.predict(x_test)

acc=np.mean(pred==y_test)
acc

###acc=0.80

from sklearn.naive_bayes import MultinomialNB 

mb=MultinomialNB ()
mb.fit(x_train,y_train)
pred=mb.predict(x_test)

acc=np.mean(pred==y_test)
acc

##accuracy =0.77