# %% [markdown]
# # Assignment 02
# # Due: Wednesday, June 12, 2024, 3:59 PM
# ## Instructions:
# - Once the notebook is completed, export to .py file.  Submit both the notebook and the .py file.  To do this, click export at the top of the notebook or ctrl + shift + p at the top of the notebook and type in export.  Export to python file should show up as a search result.
#  - DO NOT submit the data from the assignment and keep your data file and python file in the same directory.

# %% [markdown]
# # Assignment 02 - Part 1: Data Preparation and Preprocessing
# ## Instructions:
# - Use titanic_full.csv for the following steps.
# 
# 
# 
# 
# 

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('titanic full.csv')

# %% [markdown]
# 1. One hot encode the passenger gender field.  .head(10) the results

# %%
encoder = OneHotEncoder()
G_encoder = encoder.fit_transform(data[['sex']]).toarray()
gender = pd.DataFrame(G_encoder, columns=encoder.categories_)
print(gender.head(10))

# %% [markdown]
# 2. One hot encode the passenger cabin  .head(10) the results

# %%
C_encoder = encoder.fit_transform(data[['cabin']]).toarray()
cabin = pd.DataFrame(C_encoder,columns=encoder.categories_)
print(cabin.head(10))

# %% [markdown]
# 3. Prepare a dataframe which is appropriate to predict whether a passenger survived based on age, sex, cabin and pclass.  .head(10) the results

# %%
df = pd.DataFrame(data[['survived','age','pclass']])

combined_df = pd.concat([df, gender, cabin], ignore_index=True)
print(combined_df.head(10))

# %% [markdown]
# ## Instructions:
#  Use wifi_2023.csv for the next two questions

# %%
wifi_2023 = pd.read_csv('wifi_2023.csv', encoding='ISO-8859-1')

# %% [markdown]
# 4. Create one hot encoded day of week columns based off the FirstSeen field.  .head(10) the results

# %%
wifi_encoder = OneHotEncoder()
wifi_2023['FirstSeen'] = pd.to_datetime(wifi_2023['FirstSeen']).dt.day_name()
FS_encoder = wifi_encoder.fit_transform(wifi_2023[['FirstSeen']]).toarray()
DayOfWeek = pd.DataFrame(FS_encoder)
print(DayOfWeek.head(10))

# %% [markdown]
# 5. Create a cross feature which represents lat, lon and altitude.  Make this feature categorical - it should have 10 possible values.  That means you need to bin the cross feature into 10 bins (there may not be data the shows up in each bin, but you need to define 10 bins). 

# %%
wifi_2023['Cross_feature'] = wifi_2023['CurrentLatitude'] * wifi_2023['CurrentLongitude'] * wifi_2023['AltitudeMeters']
wifi_2023['Categorical_feature'] = pd.qcut(wifi_2023['Cross_feature'], 10, labels = False)

print(f'The cross feature is categorize in these 10 bins: {wifi_2023['Categorical_feature'].unique()}')


# %% [markdown]
# # Assignment 02 - Part 2: Prediction with Iterative Approach
# ## Instructions:
# - Download your file from https://clarksonmsda.org/datafiles/commute/. Your file number is in the `commute_file_assignments.xlsx` file

# %% [markdown]
# 6. Create a scatter plot which helps to show the relationship between sun_pct and commute_type

# %%
commute = pd.read_csv('commute12.csv')
sns.scatterplot(x=commute['sun_pct'], y=commute['commute_method'], s=100)
plt.show()

# %% [markdown]
# 7. Use an iterative approach to find the best sun_pct threshold overall for the entire dataset.  Print out the best accuracy and the threshold that accuracy occurs at.

# %%
def predict (sun, thres):
    if sun['sun_pct'] > thres:
        return 'foot'
    else:
        return 'car'

thres = 10
thre, accur = [],[]
rate = 0.1
while thres <=80:
    commute['Pred'] = commute.apply(lambda x:predict(x, thres), axis=1)
    commute['accuracy'] = (commute['commute_method'] == commute['Pred']).astype(int)
    accur.append(commute['accuracy'].mean())
    thre.append(thres)
    thres += rate

print(f"The threshold where the maximum accuracy occurs is: {round(thre[np.argmax(accur)], 2)} and the best accuracy is: {round(accur[np.argmax(accur)]*100, 2)}%")

# %% [markdown]
# 8. Create a scatter plot of the accuracy per iteration of your loop. 

# %%
plt.scatter(thre, accur, s=5, c='green')
plt.grid(True)
plt.show()

# %% [markdown]
# 9. Complete the same process as in question 6 above, but create a separate graph *for each day of the week*.

# %%
days_of_week =commute['day'].unique()

fig, axs = plt.subplots(len(days_of_week), 1, figsize=(10, 15), sharex=True)

for i, day in enumerate(days_of_week):
    day_data = commute[commute['day'] == day]
    axs[i].scatter(day_data['sun_pct'], day_data['commute_method'])
    axs[i].set_title(day)
    axs[i].set_xlabel('sun_pct')
    axs[i].set_ylabel('Commute_Method')


plt.tight_layout()
plt.show()

# %% [markdown]
# 10. Complete the same process as in question 7 above, but find the best sun_pct threshold *for each day of the week*.  Print out the thresholds for each day of the week and 

# %%
def predict(sun, thres):
    return 'foot' if sun > thres else 'car'

def find_best_threshold(day_data):
    Threshold = np.arange(0,101,1)
    Accuracy = []
    for thres in Threshold:
        prediction = day_data['sun_pct'].apply(lambda x: predict(x, thres))
        accuracy = (day_data['commute_method'] == prediction).mean()
        Accuracy.append(accuracy)
    best_idx = np.argmax(Accuracy)
    best_threshold = Threshold[best_idx]
    best_accuracy = Accuracy[best_idx]
    return best_threshold, best_accuracy, Threshold, Accuracy

days = commute['day'].unique()
best_thresholds = {}
WeekDayThresAccu = {}

for day in days:
    day_data = commute[commute['day'] == day].copy()
    best_threshold, best_accuracy, Threshold, Accuracy = find_best_threshold(day_data)
    best_thresholds[day]= (best_threshold, best_accuracy * 100)
    WeekDayThresAccu[day] = (Threshold, Accuracy)


print(f'The best Thresholds for each week is: {best_thresholds}')

# %% [markdown]
# 11. Complete the same process as in question 8 above, but graph the accuracy against the iteration *for each day of the week*.

# %%
Accuracy_for_each_Weekday = WeekDayThresAccu.copy()

fig, axs = plt.subplots(len(Accuracy_for_each_Weekday), 1, figsize=(12, 14), sharex=True)

for i, (weekday, (t_holds, accu)) in enumerate(Accuracy_for_each_Weekday.items()):
    axs[i].scatter(t_holds, accu, color='green')
    axs[i].set_title(f'{weekday} vs Best Accuracies')
    axs[i].set_ylabel('Best Accuracy (%)')
    axs[i].grid(True)


plt.xlabel('Threshold')
plt.tight_layout()
plt.show()

# %% [markdown]
# 12.  Is there a pattern in the thresholds, now that you can see them for each day of the week

# %%
print("""Yes, look like as the threshold passess approximately 30 then the accuracies decreases for every day of the week.""")

# %% [markdown]
# 13. Print out the overall accuracy of your predictions using the separate threshold for each day of the week.

# %%
overall_predictions = []

for day in days:
    day_data = commute[commute['day'] == day].copy()
    threshold = best_thresholds[day][0]
    day_data['Pred'] = day_data['sun_pct'].apply(lambda x: predict(x, threshold))
    overall_predictions.append(day_data)

overall_predictions_df = pd.concat(overall_predictions)
overall_accuracy = (overall_predictions_df['commute_method'] == overall_predictions_df['Pred']).mean()

print(f"Overall accuracy: {overall_accuracy * 100:.2f}%")

# %% [markdown]
# 14. Did the accuracy improve?

# %%
print("""No, the accuracy did not improve""")


