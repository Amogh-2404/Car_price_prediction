import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
# Data collection and processing using pandas datarfame
car_dataset = pd.read_csv("./car_data.csv")
car_dataset.head()
# checking missing values
car_dataset.isnull().sum()
# Our machine learning model cannot understand 'Petrol','Diesel','Dealer',etc:-
# So we have to convert it into numerical form so that it can be used by our model
""" To do so we assign the following values
    Petrol - 0
    Diesel - 1
    CNG    - 2
    Dealer - 0
    Individual - 1
    Manual - 0
    Automatic - 1
"""
#Encoding the data
car_dataset["Fuel_Type"] = car_dataset["Fuel_Type"].replace(["Petrol","Diesel","CNG"],[0,1,2])
car_dataset["Seller_Type"] = car_dataset["Seller_Type"].replace(["Dealer","Individual"],[0,1])
car_dataset["Transmission"] = car_dataset["Transmission"].replace(["Manual","Automatic"],[0,1])
# can also be done in the following manner
""" car_dataset.replace({Fuel_Type:{"Petrol":0,"Diesel":1,"CNG":2}},inplace = True)"""
# Splitting the data into training and test_data
X = car_dataset.drop(["Car_Name","Selling_Price"],axis = 1)
Y = car_dataset["Selling_Price"]

# Splitting training and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=2)
# Model training using Linear Regression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)
# Model Evaluation
training_data_prediction = lin_reg_model.predict(X_train)

# R2 error method of evaluation
error_score = metrics.r2_score(Y_train,training_data_prediction)
print("R squared error: ",error_score)

# We use the accuracy score in the case of classification problems but in case of Regression we use methods like 'r2_score'
# Visualizing actual and predicted prices
plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual prices vs Predicted prices")
plt.show()
# Now it is time to test our model using the test data
test_data_prediction = lin_reg_model.predict(X_test)
error_score_test = metrics.r2_score(Y_test,test_data_prediction)
print("R squared error: ",error_score_test)

# Visualizing test data prediction
plt.scatter(Y_test,test_data_prediction)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual prices vs Predicted prices")
plt.show()
# Model training using Lasso Regression
# Lasso regression performs well than linear regression in case where the variables are not directly proportional
lasso_model = Lasso()
lasso_model.fit(X_train,Y_train)
# Model Evaluation
training_data_prediction_lasso = lasso_model.predict(X_train)

# R2 error method of evaluation
error_score_lasso = metrics.r2_score(Y_train,training_data_prediction_lasso)
print("R squared error: ",error_score_lasso)

# We use the accuracy score in the case of classification problems but in case of Regression we use methods like 'r2_score'
# Visualizing actual and predicted prices
plt.scatter(Y_train,training_data_prediction_lasso)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual prices vs Predicted prices")
plt.show()
# Now it is time to test our model using the test data
test_data_prediction_lasso = lasso_model.predict(X_test)
error_score_test_lasso = metrics.r2_score(Y_test,test_data_prediction_lasso)
print("R squared error: ",error_score_test_lasso)

# Visualizing test data prediction
plt.scatter(Y_test,test_data_prediction_lasso)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual prices vs Predicted prices")
plt.show()
