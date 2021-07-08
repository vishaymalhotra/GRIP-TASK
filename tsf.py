#!/usr/bin/env python
# coding: utf-8

# # NAME- VISHAY MALHOTRA
# DATA SCIENCE AND BUSINESS ANALYTICS INTERN AT THE SPARKS FOUNDATION

# # Task 1 : Prediction using Supervised Machine Learning
# GRIP @ The Sparks Foundation
# In this regression task I tried to predict the percentage of marks that a student is expected to score based upon the number of hours they studied.
# 
# This is a simple linear regression task as it involves just two variables.  
# 
# Technical Stack : Sikit Learn, Numpy Array, Pandas, Matplotlib

# In[2]:


# importing necessary libraries required for task

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression


# # STEP-2 FETCHING RAW DATA

# In[4]:


data=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
data


# From the graph we can safely assume a positive linear relation between the number of hours studied and percentage of score

# # STEP-2 Input Data Visualization

# In[5]:


# data visualization and plotting of data

plt.title("HOURS STUDY Vs MARKS OBTAINED")
plt.xlabel("Numbers of hours study")
plt.ylabel("MARKS OBTAINED")
plt.scatter(data.Hours,data.Scores,color='blue',label='Data Distribution')
plt.legend(['Data Distribution'])


# # Step 3 - Data Preprocessing
# This step involved division of data into "attributes" (inputs) and "labels" (outputs).

# In[13]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values


# # Step 4 - Model Training
# Splitting the data into training and testing sets, and training the algorithm.

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
regressor = LinearRegression()  
regressor.fit(X_train.reshape(-1,1), y_train) 

print("Training complete.")


# # Step 5 - Plotting the Line of regression
# Now since our model is trained now, its the time to visualize the best-fit line of regression.

# In[15]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line,color='red');
plt.show()


# # Step 6 - Making Predictions
# Now that we have trained our algorithm, it's time to test the model by making some predictions.
# 
# For this we will use our test-set data

# In[16]:


# Testing data
print(X_test)
# Model Prediction 
y_pred = regressor.predict(X_test)


# # Step 7 - Comparing Actual result to the Predicted Model result

# In[17]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 
df


# In[18]:


#Estimating training and test score
print("Training Score:",regressor.score(X_train,y_train))
print("Test Score:",regressor.score(X_test,y_test))


# In[19]:


# Plotting the Bar graph to depict the difference between the actual and predicted value

df.plot(kind='bar',figsize=(5,5))
plt.grid(which='major', linewidth='0.5', color='red')
plt.grid(which='minor', linewidth='0.5', color='blue')
plt.show()


# In[20]:


# Testing the model with our own data
hours = 9.25
test = np.array([hours])
test = test.reshape(-1, 1)
own_pred = regressor.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # Step 8 - Evaluating the model
# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. Here different errors have been calculated to compare the model performance and predict the accuracy

# In[21]:


from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-2:', metrics.r2_score(y_test, y_pred))


# R-2 gives the score of model fit and in this case we have R-2 = 0.9454906892105355 which is actually a great score for this model.

# # Conclusion
# I was successfully able to carry-out Prediction using Supervised ML task and was able to evaluate the model's performance on various parameters.¶
# Thank You

# In[ ]:




