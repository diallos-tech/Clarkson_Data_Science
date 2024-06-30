# %% [markdown]
# # Assignment 03
# # Due: Wednesday, June 19, 2024, 3:59 PM
# ## Instructions:
# 1. Once the notebook is completed, export to .py file.  Submit both the notebook and the .py file.  To do this, click export at the top of the notebook or ctrl + shift + p at the top of the notebook and type in export.  Export to python file should show up as a search result.  Also:
#     - Ensure that your .py file is an exact replica of your .ipynb file.  
#     - Ensure your .py and .ipynb files run successfully without any errors.  You should be able to click `Run All` in VS Code and run the notebook without error before converting to a .py file.  When completed, you should be able to run the .py file from the terminal or command prompt.
# 2. DO NOT submit the data from the assignment and keep your data file and python file in the same directory. Do not use your local directory path to read files (e.g., avoid using paths like C:/your/directory/file.csv).  Just read in the file directly as if though it was in the same directory as your .ipynb file
# 3. Whenever we ask to .head(10) the results or print out a value, please use `print()` so for example `print(df.head(10))`. Print only the answers to the questions that have been asked.  Do not print the head of a dataset unless explicitly asked. 
# 4. Whenever displaying a graph use `plt.show()`
# 5. For theoretical answers/short answers, please use print() (e.g., print("your answer")).
# 6. Wherever we have code displayed to print out values, use that code as a template to print out your output.  For example, if we give you `print(f'Threshold for best accuracy: {}')` please use code in this style to print out your output.  For some questions, this is only a template as we expect you to print out multiple answers (For example....Fit a linear regression model to each of the 5 features INDIVUDALLY and print out the slope and intercept for each. (Don't forget train/test split) requires you use the template for each feature)
# 7. Do not include pip install commands in your code. You can assume that all required libraries are already installed.

# %% [markdown]
# # Assignment 03 - Part 1: Gradient Descent
# ## Instructions:
# Download your file from https://clarksonmsda.org/datafiles/fuel/.  Your number is the same as the previous assignment.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()

# %% [markdown]
# 1. Use the gradient descent script we developed in class to find the line of of best fit for your fuel file.  You may want to run for more epochs, since we are working with more data.  Graph the loss as a function of epoch, as we did in class.

# %%
data = pd.read_csv('gas12.csv')

vehicle_weight = data['vehicle_tons']
fuel_consump = data['gas_pumped']

l_rate = 0.02
n = len(vehicle_weight)
slope = 0
intercept = 0

line = []

for i in np.arange(0,100,2):
    y_predict = vehicle_weight*slope + intercept
    D_m = (-2/n) * np.sum(vehicle_weight * (fuel_consump - y_predict)) # Derivative wrt m
    D_c = (-2/n) * np.sum(fuel_consump - y_predict)  # Derivative wrt c
    slope = slope - (l_rate * D_m)
    intercept = intercept - (l_rate * D_c)
    loss = np.sum((fuel_consump - y_predict)**2/n)
    plt.scatter(i,loss)
    line.append({'Slope':slope, 'Intercept':intercept, 'Loss':loss})

plt.show()


# %% [markdown]
# 2. Print your final slope and intercept terms (we also referred to our slope as m and our intercept as c)

# %%
print(f'Final slope: {slope:,.2f}')
print(f'Final intercept: {intercept:,.2f}')

# %% [markdown]
# 3. Print the total final loss (sum of squared residuals divided by n)

# %%
print(f'Total final loss: {loss:,.2f}')

# %% [markdown]
# 4. Plot the final line and points

# %%
plt.scatter(vehicle_weight, fuel_consump, color='blue')
y_final = vehicle_weight * slope + intercept
plt.plot(vehicle_weight, y_final, color='red')

plt.xlabel('Vehicle Weight (tons)')
plt.ylabel('Gas Pumped)')
plt.title('Vehicle Weight vs Gas Pumped')
plt.show()

# %% [markdown]
# 5. Let's observe the effect of initializing the slope (m) and intercept (c) as random integers?  
# 
#     a. In the cell below copy your code above and generate a random integer for the initial slope and intercept from 10 to 20 (hint: use np.random.randint and the high and low arguments).  As above, plot the loss as a function of epoch

# %%
# Set a random seed of 42, so your random numbers always come out the same
np.random.seed(42)
# Use same code as above (except initialize slope and intercept to random integers)
# Your code below 
l_rate = 0.02
n = len(vehicle_weight)
slope = np.random.randint(10,21)
intercept = np.random.randint(10,21)

for i in np.arange(0,100,2):
    y_predict = vehicle_weight*slope + intercept
    D_m = (-2/n) * np.sum(vehicle_weight * (fuel_consump - y_predict)) # Derivative wrt m
    D_c = (-2/n) * np.sum(fuel_consump - y_predict)  # Derivative wrt c
    slope = slope - (l_rate * D_m)
    intercept = intercept - (l_rate * D_c)
    loss = np.sum((fuel_consump - y_predict)**2/n)
    plt.scatter(i,loss)

plt.show()

# %% [markdown]
# b. Print out your final slope and intercept for this new training.

# %%
print(f'Final slope: {slope}')
print(f'Final intercept: {intercept}')

# %% [markdown]
# c. Plot the final line and points

# %%
plt.scatter(vehicle_weight, fuel_consump, color='green')
y_final = vehicle_weight * slope + intercept
plt.plot(vehicle_weight, y_final, color='red')
plt.show()

# %% [markdown]
# d. What effect does it have on the loss and the approach to the actual slope values?

# %%
print("""The loss got smaller which gives us a larger slope and intercept""")

# %% [markdown]
# 6. Try different learning rates.  
# 
#     a. Use the list of learning rates below (called learning_rates).  Loop through those learning rates and try different rates for 100 epochs.  For each rate, print out the learning rate, final slope, final intercept and total final loss and graph the loss against the epoch as above.

# %%
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1, 5]
epochs = 100

for learning_rate in learning_rates:
    slope = 0
    intercept = 0
    for i in np.arange(epochs):
        y_predict = vehicle_weight*slope + intercept
        D_m = (-2/n) * np.sum(vehicle_weight * (fuel_consump - y_predict)) # Derivative wrt m
        D_c = (-2/n) * np.sum(fuel_consump - y_predict)  # Derivative wrt c
        slope = slope - (learning_rate * D_m)
        intercept = intercept - (learning_rate * D_c)
        loss = np.sum((fuel_consump - y_predict)**2/n)
        plt.scatter(i,loss)
        
    print(f'\nLearning Rate: {learning_rate}')
    print(f'Final slope: {slope}')
    print(f'Final intercept: {intercept}')
    print(f'Total final loss: {loss}')
    
    plt.show()

# %% [markdown]
# b. What is the 'best' learning rate?  What happens when the learning rate is too big?  What happens when it is too small? 

# %%
print("""My best learning rate is at 0.01. When the learning rate is too big the the slope and intercept becomes negative and the line gets flatter as the loss gets larger. When it's too small then the line almost a straightline.""")

# %% [markdown]
# Extra Credit:
# 7. Propose a method for implementing stochastic gradient descent.

# %%
print("""Your answer here""")

# %% [markdown]
# # Assignment 02 - Part 2: Student Performance with sklearn
# ## Instructions:
# Use the `student.csv` file

# %% [markdown]
# 8. Plot each x against the y (Performance Index).  Make the title of each graph the name of the feature vs Performance Index (e.g. 'Hours Studied vs. Performance Index')

# %%
students_df = pd.read_csv('students.csv')


# %%
students_df['Extracurricular Activities'] = encoder.fit_transform(students_df[['Extracurricular Activities']]).toarray()
X = students_df[[name for name in students_df.columns if name not in ['Performance Index']]]
y = students_df['Performance Index']

for c in X.columns:
    plt.figure(figsize=(12,6))
    plt.scatter(X[c], y, c='green', alpha = 0.1)
    plt.title(f'{c} vs Performance Index')
    plt.xlabel(c)
    plt.ylabel('Performance Index')
    plt.tight_layout()

plt.show()

# %% [markdown]
# 9. Fit a linear regression model to each of the 5 features INDIVIDUALLY and print out the slope, intercept and mean squared error for each. (Don't forget train/test split - use test_size of 0.2 and a random_state of 42).  hint: You may think to One-Hot Encode the Extracurricular Activities variable, but since it is just Yes or No, best to engineer a single feature/column where Yes=1 and No=0

# %%
for name in ['Hours Studied','Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced']:
    features = students_df[[name]]
    target= students_df['Performance Index']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr = lr.fit(X_train, y_train)
    mse = mean_squared_error(y_test, y_pred=lr.predict(X_test))

    print(f'\nFeature: {name}\n  Slope: {lr.coef_}\n  Intercept {lr.intercept_}\n  Mean Squared Error {mse:,.2f}')

# %% [markdown]
# 10. Which is the best single feature based on mean_squared_error?

# %%
print("""The best signle feature is Previous Score as it has the lowest mean square error!""")

# %% [markdown]
# 11. Use different test sizes in the array [0.3,0.5,0.7] with that one single feature.  Also use a random_state of 42.  Print out the test size, slope, intercept and test mean squared error.

# %%
for size in np.array([0.3,0.5,0.7]):
    features = students_df[['Previous Scores']]
    target= students_df['Performance Index']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=size, random_state=42)
    lr = LinearRegression()
    lr = lr.fit(X_train, y_train)
    mse = mean_squared_error(y_test, y_pred=lr.predict(X_test))

    print(f'\nTest Size: {size}\n  Slope: {lr.coef_}\n  Intercept {lr.intercept_}\n  Mean Squared Error {mse}')


# %% [markdown]
# 12. What is the result of changing the training size?

# %%
print("""Changing the trainning size increase the error, and lowered the slope and intercept. """)

# %% [markdown]
# 13. Fit a linear regression model to all the features (multiple x features).  Print out the slope and the weights for each feature.

# %%
f = students_df[['Hours Studied','Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced']]
t = students_df['Performance Index']

X_train, X_test, y_train, y_test = train_test_split(f, t, test_size=0.2, random_state=42)
reg = LinearRegression().fit(X_train, y_train)

for f_name, w in zip(reg.feature_names_in_, reg.coef_):
    print(f'\nFeature Name: {f_name}\n  Weight: {w}')

# %% [markdown]
# 14. Call predict with new data and print out the mean squared error.

# %%
new_pred = reg.predict(pd.DataFrame([[40, 40, 70, 5, 20]], columns=X_train.columns))
print(f'Mean Squared Error: \n{y_test, new_pred}')

# %% [markdown]
# 15. Plot your test y values against your predicted y values

# %%
y_pred = reg.predict(X_test)
plt.scatter(y_test, y_pred)
plt.show()

# %% [markdown]
# 16. Is the result intuitive? (Does it seem to fit what you would expect for a student who performs well vs. poorly)

# %%
print("""Yes, this seems the make sense as the x and y are linearly dependent.""")


