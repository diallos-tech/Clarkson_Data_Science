# %% [markdown]
# # Assignment 05
# # Due: Monday July 8th, 2024, 3:59 PM
# # Late submissions until July 10th, 3:59 PM
# ## Instructions:
# 1. Once the notebook is completed, export to .py file.  Submit both the notebook and the .py file.  To do this, click export at the top of the notebook or ctrl + shift + p at the top of the notebook and type in export.  Export to python file should show up as a search result.  Also:
#     - Ensure that your .py file is an exact replica of your .ipynb file.  
#     - Ensure your .py and .ipynb files run successfully without any errors.  You should be able to click `Run All` in VS Code and run the notebook without error before converting to a .py file.  When completed, you should be able to run the .py file from the terminal or command prompt.
# 2. DO NOT submit the data from the assignment and keep your data file and python file in the same directory. Do not use your local directory path to read files (e.g., avoid using paths like C:/your/directory/file.csv).  Just read in the file directly as if though it was in the same directory as your .ipynb file.  DO NOT CHANGE THE NAME OF THE FILE....
# 3. Whenever we ask to .head(10) the results or print out a value, please use `print()` so for example `print(df.head(10))`. Print only the answers to the questions that have been asked.  Do not print the head of a dataset unless explicitly asked. 
# 4. Whenever displaying a graph use `plt.show()`
# 5. For theoretical answers/short answers, please use print() (e.g., print("your answer")).
# 6. Wherever we have code displayed to print out values, use that code as a template to print out your output.  For example, if we give you `print(f'Threshold for best accuracy: {}')` please use code in this style to print out your output.  For some questions, this is only a template as we expect you to print out multiple answers (For example....Fit a linear regression model to each of the 5 features INDIVUDALLY and print out the slope and intercept for each. (Don't forget train/test split) requires you use the template for each feature)
# 7. Do not include pip install commands in your code. You can assume that all required libraries are already installed.

# %% [markdown]
# # Assignment 5: SVM With Real Estate Data

# %% [markdown]
# ## Instructions:
# 
# For this assignment we are going to use the northeast_realestate.parquet file that we did in the first assignment.  We will basically:
# 
# 1. Complete some basic EDA
# 2. Filter out bad data
# 3. Create a train test split
# 4. Fit an SVM and a Logistic Regression to compare
# 5. Use GridSearchCV to find the best parameters for an SVM and analyze the best model

# %% [markdown]
# 1. Import data and packages here.  Import parquet file using `pd.read_parquet("northeast_realestate.parquet")`.  Print out the head of the dataframe.

# %%
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics
from sklearn.inspection import DecisionBoundaryDisplay

# %%
data = pd.read_parquet('northeast_realestate.parquet', engine='pyarrow')
print(data.head())

# %% [markdown]
# 2. Print out the shape of the dataframe

# %%
print(data.shape)


# %% [markdown]
# 3. Create a new feature like we did in class called `'citystate'` that is a combination of the city and state separated by an underscore.  Print out the head of this new column.

# %%
data['citystate'] = data['city']+'_'+data['state']
print(data['citystate'].head())

# %% [markdown]
# 4. Filter only on 'Staten Island_New York','Worcester_Massachusetts','Manhattan_New York' and 'Portland_Maine' and print out the shape of the resultant data frame

# %%
q = "(citystate==['Staten Island_New York','Worcester_Massachusetts','Manhattan_New York','Portland_Maine'])"
data = data.query(q)
print(data.shape)

# %% [markdown]
# # EDA

# %% [markdown]
# 5. For each column in this new data frame, print out the number of null values, the number of not null values and the percent of nulls in each column. You can either put all values in a data frame and print it out or loop through the columns and print them out separately

# %%
for c, v in data.items():
    print((f'Column: {c}\n Number null: {v.isnull().sum()}\n Number not null: {v.notnull().sum()}\n Proportion null: {v.isnull().sum()/v.shape[0]}'))

# %% [markdown]
# 6. Select only the 'price','bed','bath','house_size' and 'citystate' variables and drop the nulls and drop the duplicates.  Print the head of the dataset.

# %%
data = data[['price','bed','bath','house_size','citystate']].dropna().drop_duplicates()
print(data.head())

# %% [markdown]
# 7. Print the shape of this new dataframe

# %%
print(data.shape)

# %% [markdown]
# 8. Create a pair scatter plot of all the numeric variables (all the variables except the citystate).  I think we learned how to do this in `03_data_prep_and_preprocessing_notebook_02.ipynb`

# %%
g = sns.PairGrid(data.drop(columns=['citystate']))
g.map(plt.scatter)
plt.show()

# %% [markdown]
# 9. Which variables seem to be correlated with eachother?

# %%
print("""Number of bed and bath,  house size and bed as well as house size and bath seems to have the highest correlation""")

# %% [markdown]
# 10. Create two column plots displaying the count of observations for all values of bed and bath

# %%
fig, axs = plt.subplots(nrows=1, ncols=2, figsize= (8,4))

variables = ['bed','bath']
for ax, var in zip(axs.flatten(), variables):
    counts = data[var].value_counts()
    ax.bar(counts.index, counts.values)
    ax.set_title(f'{var} Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel(var)

plt.show()

# %% [markdown]
# 11. Print out the row(s) for the properties with the highest number of beds.  Do any bed values seem like outliers?  Does the price seem to match your intuition about the number of beds?

# %%
properties = data.dropna().drop_duplicates().sort_values(by='bed',ascending=False)
print(f'Row with highest number of beds:\n{properties.head(5)}')
print("""Your answer here""")

# %% [markdown]
# 12. Create a histogram of the price

# %%
plt.hist(data['price'], bins=100, edgecolor = 'black')
plt.title('Distribution of Property Price')
plt.ylabel('Frequency')
plt.xlabel('Price')
plt.xlim(left=0, right=1*1e7)
plt.show()

# %% [markdown]
# 13. Is the price skewed?  What type of transformation (see `03_data_prep_and_preprocessing_notebook_02.ipynb`) should we use to transform a value with such extreme values?

# %%
print("""Yes, the price is skewed left, we can use log transformation to transform the data""")

# %% [markdown]
# 14. Plot a histogram of the transformed price.  Does it look more normal?

# %%
plt.hist(np.log(data['price']), bins=50)
plt.title('Distribution of Property Price')
plt.ylabel('Frequency')
plt.xlabel('Price')
plt.show()
print("""Yes, this is more normally distributed now""")

# %% [markdown]
# 15. Do the same for house_size. Plot the histogram of house_size

# %%
plt.hist(data['house_size'], bins=50)
plt.title('Distribution of Property house size')
plt.ylabel('Frequency')
plt.xlabel('House Size')
plt.show()

# %% [markdown]
# 16. Should house_size be transformed?

# %%
print("""Yes, this should be transformed as it skewed left""")

# %% [markdown]
# 17. Plot a histogram of transformed house_size.  Does is appear more normal?

# %%
plt.hist(np.log(data['house_size']), bins=100)
plt.title('Distribution of Property house size')
plt.ylabel('Frequency')
plt.xlabel('House Size')
plt.show()
print("""Yes, this is more normal""")

# %% [markdown]
# 18. Create a correlation matrix for the numeric variables.   Use a heatmap and the "flare" color palette

# %%
corr = data.drop(columns='citystate').corr() * 100
sns.heatmap(corr, annot=True, linewidth=.1, vmin=0, vmax=100,
            fmt=".2f", cmap=sns.color_palette("flare", as_cmap=True))
plt.show()

# %% [markdown]
# 19.  What variables are most correlated?  If we were to complete PCA in this assignment (which we aren't), which X variables would be good candidates for combining for PCA?

# %%
print("""The variables that are more correlated are: Bed and Bath, House Size and Bath, Bed and House Size""")

# %% [markdown]
# 20.  Calculate and print out the percentage of rows by citystate.

# %%
citystate_counts = data['citystate'].value_counts()
total_count = data['citystate'].shape[0]

row_percentage = round((citystate_counts / total_count) * 100,6)

print(f'The row percentage by CityState is: \n\n{row_percentage}')

# %% [markdown]
# 21. Is the data set balanced?  Which citystate combination occurs the most?  The least?

# %%
print("""No, the data is not balanced. The citystate that occurs the most is Staten Island_New York and the one that occur the least is Portland_Maine""")

# %% [markdown]
# 22. Create your X and y variables.  Transform house_size and price the way we did above.  Your y, or target variable, should be citystate.

# %%
data['house_size'] = np.log(data['house_size'])
data['price'] = np.log(data['price'])

X = data.drop(columns='citystate')
y = data['citystate']

# %% [markdown]
# 23. Create a train test split.  Use 0.20 test size and random_state = 42.  Print the shape of your x train, x test, y train and y test data sets

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# 24. Print out the proportion of observations by citystate.

# %%
print(f'The proportion of observation by city is: {y_train.value_counts()/y_train.shape[0]}')

# %% [markdown]
# 25.  Does the distribution of values match the overall data set?

# %%
print("""Yes, it is closer to the original datasets. """)

# %% [markdown]
# 26. Override your train test split, except stratify by the y variable

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# %% [markdown]
# 27. Print out the distribution of y values.

# %%
print(f'The y values distribution is: \n\n{y_train.value_counts()/y_train.shape[0]}')

# %% [markdown]
# 28.  Is the distribution of your y values closer to the original dataset?

# %%
print("""Yes, it is close to original datasets.""")

# %% [markdown]
# 29. Scale your training data using `fit_transform` and then use `transform` to transform the test data as well using the same scaler (don't call `fit_transform again`, just `fit`).  Print out the head of your scaled X data.

# %%
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
X_scaled = np.concatenate([X_train_scaled, X_test_scaled])
print(f"Train Data: \n{pd.DataFrame(X_scaled, columns=X_train.columns).head()}")

# %% [markdown]
# 30. Create a separate scatter plot of your training data (or use a FacetGrid and use col='citystate') for each citystate with price on the x axis and house_size on the y.

# %%
train_data = X_train.copy()
train_data['citystate']= data['citystate'][X_train.index]
g = sns.FacetGrid(train_data, col='citystate')
g.map(plt.scatter, 'price', 'house_size')
plt.show()

# %% [markdown]
# 31.  Create a single scatter plot with price on the x and house_size on the y.  Color the observations by citystate and add a legend.

# %%
sns.scatterplot(data=data, x=data['price'], y=data['house_size'], hue=data['citystate'])
plt.title('Price vs House_Size')
plt.xlabel('Price')
plt.ylabel('House_Size')
plt.legend()
plt.show()

# %% [markdown]
# 32. Which citystates seem like they might be the most easy to categorize or separate using these two variables, from what you can see in your graphs.  Which ones seem like they will be more difficult to separate?

# %%
print("""Manhattan_New York and Staten Island_New York seem to be easy to separate. Portland_Maine and Worcester_Massachusetts are more difficult to separate.""")

# %% [markdown]
# 33. Fit a logistic regression and print out the accuracy_score and f1_score (use the macro average) for the training and testing data

# %%
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
y_test_pred = logreg.predict(X_test_scaled)
y_train_pred = logreg.predict(X_train_scaled)
print(f'The Test data accuracy is: {metrics.accuracy_score(y_test, y_test_pred)} \nand the F1_score is: {
      metrics.f1_score(y_test,y_test_pred, average='macro')}')


print(f'The Train data accuracy is: {metrics.accuracy_score(y_train, y_train_pred)} \nand the F1_score is: {
      metrics.f1_score(y_train,y_train_pred, average='macro')}')

# %% [markdown]
# 34. Do the same, except with a SVC.  Use C=1 and gamma=1.

# %%
svc = SVC(C=1, gamma=1)
svc.fit(X_train_scaled, y_train)

y_test_pred = svc.predict(X_test_scaled)
y_train_pred = svc.predict(X_train_scaled)

print(f'The Test data accuracy is: {metrics.accuracy_score(y_test, y_test_pred)} \nand the F1_score is: {
      metrics.f1_score(y_test,y_test_pred, average='macro')}')

print(f'The Train data accuracy is: {metrics.accuracy_score(y_train, y_train_pred)} \nand the F1_score is: {
      metrics.f1_score(y_train,y_train_pred, average='macro')}')

# %% [markdown]
# 35. Which model seems to perform better on the test set?

# %%
print("""The SVC model performed better.""")

# %% [markdown]
# 36. Do grid search to find the best hyperparameters
# 
# - Use KFold and n_splits=5 and a random_state of 7.  Set shuffle=True
# - Use the params dictionary given (you should calculate 175 models)
# - Use your scaled data
# - Print out the best score and best parameters.  Look at the sklearn documentation for how to select the best score and parameters from the grid search.

# %%
svc = SVC()
svc.fit(X_train_scaled, y_train)
params = {'C': [0.1, 1, 10, 100, 1000, 10000, 20000],
          'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

n_folds = KFold(n_splits=5, random_state=7, shuffle=True)

best_model = GridSearchCV(estimator=svc, param_grid=params, scoring='f1_macro', n_jobs=-1, verbose=1, cv=n_folds)

best_model.fit(X_train_scaled, y_train)

print(f'Best Score: {best_model.best_score_}')
print(f'Best Parameters: {best_model.best_params_}')

# %% [markdown]
# 37. Do the same as above, except use your unscaled data.  Print out the best score and parameters

# %%
svc = SVC()
svc.fit(X_train, y_train)
params = {'C': [0.1, 1, 10, 100, 1000, 10000, 20000],
          'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

n_folds = KFold(n_splits=5, random_state=7, shuffle=True)

best_model = GridSearchCV(estimator=svc, param_grid=params, n_jobs=-1, verbose=1, cv=n_folds)

best_model.fit(X_train, y_train)

print(f'Best Score: {best_model.best_score_}')
print(f'Best Parameters: {best_model.best_params_}')

# %% [markdown]
# 38.  Which combination of data and parameters (scaled vs unscaled, gamma, C) seem to produce the best model?

# %%
print("""For scaled data C of 100 and gamma of 1 and for unscaled data C of 1000 and gamma of 0.01 are best to produce the best model.""")

# %% [markdown]
# 39. Fit a final model using your best parameters above on the scaled data.  Print out the accuracy_score and f1_score (use macro average) for the training and test data.

# %%
svc = SVC(C=100, gamma=1)
svc.fit(X_train_scaled, y_train)

y_pred = svc.predict(X_test_scaled)
y_pred_train = svc.predict(X_train_scaled)

print(metrics.accuracy_score(y_train, y_pred_train),
      metrics.f1_score(y_test, y_pred, average='macro'))


print(metrics.accuracy_score(y_test, y_pred),
      metrics.f1_score(y_test, y_pred, average='macro'))

# %% [markdown]
# 40. Is the f1 score for the test set (from the previous question, question 39) close to the f1 score you calculated for your best model above (question 36) through cross validation?  Said another way, would you say that your f1 score found in the grid search (question 36) is a good approximation of your actual f1 test score (from the previous question, question 39)?

# %%
print("""Yes, there are very close as on Q36 is 67.99% where as Q39 is 64.25%. So, yes it's not that far off.""")

# %% [markdown]
# 41. Plot a confusion matrix using `metrics.ConfusionMatrixDisplay.from_predictions`.  Use `normalize='true'` and `values_format=".0%"`.

# %%
metrics.ConfusionMatrixDisplay.from_predictions(y_test,
                                                y_pred,
                                                normalize='true',
                                                values_format=".0%",
                                                xticks_rotation='vertical')
plt.show()

# %% [markdown]
# 42. Where does there seem to be the most confusion (i.e. which classes seem to be confused for one another)?  Does this make sense given your scatter plots in questions 30-31?

# %%
print("""Predicted label 'Staten Island seem to be more confused on Portland_Maine and Worcester_massachusetts""")

# %% [markdown]
# 43. Use `metrics.classification_report` on the test data.  Print out the classification report.

# %%
print(metrics.classification_report(y_test, y_pred))

# %% [markdown]
# 44. What do your f1 scores tell you about which categories are easiest to classify and which are hardest?

# %%
print("""Manhattan_New York and Staten Island_New York are easier to classify whereas Portland_Maine and Worcester_Massachusetts are harder to classify.""")

# %% [markdown]
# 45. What recommendations would you give someone to improve their model

# %%
print("""I would recommend increasing the size of the datasets to get a bit more of a balance datasets to minimize confusion and get more accuracy.""")

# %% [markdown]
# # Bonus (10 Points)
# 
# 46. Use a label encoder on the y variable train data to encode as numbers instead of strings.  Use the same label encoder on the test data to transform that data as well.
# 

# %%
label = LabelEncoder()
y_train = label.fit_transform(y_train)
y_test = label.transform(y_test)

# %% [markdown]
# 47. Fit a model on only price and house_size.  Use the scaled data and the parameters from your best model above (use an SVC, not Logistic Regression) and the newly encoded y train as your y variable
# 

# %%
svc = SVC(C=100, gamma=1)

svc.fit( X_train_scaled[['price', 'house_size']], y_train)

y_train_pred = svc.predict(X_train_scaled[['price', 'house_size']])
y_test_pred = svc.predict(X_test_scaled[['price', 'house_size']])

# %% [markdown]
# 48. Print out the training and test accuracy and f1 scores (use macro average).  
# 

# %%
print(metrics.accuracy_score(y_test, y_test_pred),
      metrics.f1_score(y_test, y_test_pred, average='macro'))

print(metrics.accuracy_score(y_train, y_train_pred),
      metrics.f1_score(y_train, y_train_pred, average='macro'))

# %% [markdown]
# 49. How does the test accuracy and f1_scores seem to compare to our model above that had bed and bath in them?
# 

# %%
print("""There is no changes to the test or train accuraccy and f1-score""")

# %% [markdown]
# 50. Plot a confusion matrix with `normalize='true'` and `values_format=".0%"`.  Create a title indicating which labels from the label encoder correspond to which string
# 

# %%
metrics.ConfusionMatrixDisplay.from_predictions(y_test,
                                                y_test_pred,
                                                normalize='true',
                                                values_format=".0%",
                                                xticks_rotation='vertical')
plt.show()

# %% [markdown]
# 51. Where does the confusion seem to be?
# 

# %%
print("""Same areas as before just that model confusion increased.""")

# %% [markdown]
# 52. Plot the decision boundary with only scatter plot for Manhattan and Worchester
# 

# %%


# %% [markdown]
# 53. Plot the decision boundary with scatter plot for all four categories

# %%
plt.show()

# %% [markdown]
# 54. Do your decision boundaries make sense?

# %%
print("""Your answer here""")


