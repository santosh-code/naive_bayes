# Import the raw_sms dataset
library(readr)
SalaryData_Train<-read.csv(file.choose())
SalaryData_Test<-read.csv(file.choose())

str(SalaryData_Train)
str(SalaryData_Test)

#######EDA
#univariet analy
hist(SalaryData_Train$age)
boxplot(SalaryData_Train$age)

SalaryData_Train$educationno<-as.factor(SalaryData_Train$educationno)
SalaryData_Test$educationno<-as.factor(SalaryData_Test$educationno)

model<-naiveBayes(Salary~.,data =SalaryData_Train )
model

model_predict<-predict(model,SalaryData_Test)

acc<-mean(model_predict==SalaryData_Test$Salary)
acc

CrossTable(model_predict,SalaryData_Test$Salary ,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
