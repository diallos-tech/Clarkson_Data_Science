# %% [markdown]
# # Assignment 01
# # Due: Monday, June 3, 2024, 3:59 PM
# ## Instructions:
# - Once the notebook is completed, export to .py file.  Submit both the notebook and the .py file.  To do this, click export at the top of the notebook or ctrl + shift + p at the top of the notebook and type in export.  Export to python file should show up as a search result.
#  - DO NOT submit the data from the assignment and keep your data file and python file in the same directory.

# %% [markdown]
# # Assignment 01 - Part 1: Machine Learning Landscape
# ## Instructions:
# - use matplotlib and numpy only for part 1.
# - For the first part [home sensor doc](https://docs.google.com/document/d/17RO8tu7XDvOboaZlYYyc9lr1X4oOrJv5xlstrRlyMGQ/edit) contains all the info you need to understand the data.  Read this to select the correct columns below

# %% [markdown]
# # Import packages and data
# 
# There are a number of ways to import data in numpy 
# (obviously this would be much easier in pandas, but the point of the exercise
# is to use numpy).  We used the `np.genfromtxt` function in class.  Read the documentation for more information on how to import the data using
# only this function.  Be mindful of the `dtype` and `encoding` arguments.  You may want to store each column of the array as a separate entry in a dicationary, where the keys are the column names and the values are the columns.  This may make it easier to work with the data.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_1 = np.genfromtxt('homesensors_2021.csv', delimiter=',', dtype='str')
data_dict = {data_1[0,i]:data_1[1:,i] for i in range(0,37)}
data_dict['temp_8'] = data_dict['temp_8'].astype('float64')
dates = pd.to_datetime(data_dict['pitime'])

# %% [markdown]
# - draw a scatter plot of outside temperature over the entire year

# %%
plt.scatter(dates, data_dict['temp_8'], s= 0.1, c='blue')
plt.xticks(rotation = 90)
plt.title('Scatter Plot of Outside Temperature')
plt.xlabel('Dates')
plt.ylabel('Temperature')
plt.show()

# %% [markdown]
# - draw a histogram of temperature binned into 50 bins over the whole year

# %%

plt.hist(data_dict['temp_8'], bins=50)
plt.title('Outside Temperature distribution')
plt.ylabel('Values')
plt.show()

# %% [markdown]
# - draw a scatter plot of 'adc_67_avg' as x and 'adc_68_avg' as y, with an x range of 512-518

# %%
x_values = data_dict['adc_67_avg'].astype(float)
y_values = data_dict['adc_68_avg'].astype(float)
plt.scatter(x_values, y_values, s=1, c=data_dict['temp_8'])
plt.xlim(512,518)
plt.title('Liquid Moisture Basement Floor "EAST  vs WEST"')
plt.xlabel('adc_67_avg EAST')
plt.ylabel('adc_68_avg WEST')
plt.show()

# %% [markdown]
# - what might cause the stepping pattern in the relationship?

# %%
"""
This might be due to a faulty sensor, placement of the sensor or just calibration of the sensor.
"""

# %% [markdown]
# - draw a histogram of the outdoor relative humidity.

# %%
plt.hist(data_dict['rh_3'].astype(float), bins=7, color='green')
plt.title('Histogram of the outdoor relative humidity')
plt.ylabel('Values')
plt.xlabel('Bins=7')
plt.show()

# %% [markdown]
# - is the distribution normal?  what might bias this measurement?

# %%
"""
No this is not normally distributed. The sensor might be faulty and not reading the correct outputs.
"""

# %% [markdown]
# - draw a scatter plot of rh_4 to co2 colored by temp_8.  Use the 'summer' cmap

# %%
plt.scatter(data_dict['rh_4'].astype(float),data_dict['co2'].astype(float), s=1, c=data_dict['temp_8']) 
plt.title('Relative Humidity Floor 1 vs CO2 Equivalent Floor 1')
plt.xlabel('rh_4')
plt.ylabel('co2')
plt.show()

# %% [markdown]
# - draw a scatter plot of 'motion front yard right' and 'motion front yard left'.  use outside temperature as you color

# %%
plt.scatter(data_dict['trpin_40_sampleson'].astype(float), data_dict['trpin_44_sampleson'].astype(float), s=10, c = data_dict['temp_8'])
plt.title('Motion Front Yard Right vs Motion Front Yard Left')
plt.xlabel('motion front yard right')
plt.ylabel('motion front yard left')
plt.show()

# %% [markdown]
#  - what does the color tell us about the a bias in the data?

# %%
"""
The darker colors tells us that there is a high correlation whereas the lighter color tells us there is small to none correlation.
"""

# %% [markdown]
# # Assignment 01 - Part 2: Pandas and Numpy
#   - load the northeast_realestate.parquet

# %%
data_3 = pd.read_parquet('northeast_realestate.parquet')

# %% [markdown]
# - Replace the zipcode with a zero padded string, head(10) the result

# %%
data_3['zipcode'] = data_3['zip_code'].astype(str)
data_3['zipcode'] = data_3['zipcode'].str.zfill(10)
print(data_3['zipcode'].head(10))

# %% [markdown]
# - Find unique properties in Rye, NY.  head(15) the result

# %%
Rye_NY = data_3.query("(city == 'Rye') & (state == 'New York')")
print(Rye_NY.head(15))

# %% [markdown]
# - Show only properties sold in March and April. head(5) the result

# %%
data_3['date'] = pd.to_datetime(data_3['sold_date'])
march_april_properties = data_3.query("(date.dt.month == 3 or date.dt.month == 4)")
print(march_april_properties.head(5))

# %% [markdown]
# - What is the formatting string of the sold_date?

# %%
"""The formating of the sold_date is yyyy-mm-dd."""

# %% [markdown]
# - What is the difference between the average price of a home sold with St in the address (on a street) and Rd in the address (on a road)

# %%
# Calculate the average price for streets ending with 'St'
avg_price_st = data_3[data_3['street'].fillna('').str.endswith('St')]['price'].mean()

# Calculate the average price for streets ending with 'Rd'
avg_price_rd = data_3[data_3['street'].fillna('').str.endswith('Rd')]['price'].mean()

diff_price = avg_price_st - avg_price_rd

print(round(diff_price,2))


# %% [markdown]
# - Which are the top 20 cities with highest home sale prices in NY? (sorted by price DESC)

# %%
filter_ny = data_3[data_3['state'] == 'New York'].sort_values(by='price', ascending=False)

top_20_cities_NY = filter_ny.head(20)

print(top_20_cities_NY)

# %% [markdown]
# - load wifi_list.csv and wifi_2023.csv

# %%
wifi_list = pd.read_csv('wifi_list.csv', encoding='ISO-8859-1')
wifi_2023 = pd.read_csv('wifi_2023.csv', encoding='ISO-8859-1')

# %% [markdown]
# - generate a new dataframe showing only access points existing in 2012 and 2023 (use MAC as point of truth for whether the AP is new). Only select the MAC and SSID columns and select the unique rows for each year and then merge or join the results.  head(10) the results

# %%
wifi_2023_unique = wifi_2023[['MAC','SSID']].drop_duplicates()
wifi_list_unique = wifi_list[['MAC','SSID']].drop_duplicates()

df_0 = wifi_2023_unique.merge(wifi_list_unique, how = 'inner', left_on = 'MAC', right_on = 'MAC')
print(df_0.head(10))

# %% [markdown]
# - label the columns so you can see which attributes changed.  In  other words each column should get a prefix or postfix indicating it's year.

# %%
df_1 = wifi_2023_unique.merge(wifi_list_unique, how = 'inner', left_on = 'MAC', right_on = 'MAC', suffixes=['_2012', '_2023'])
print(df_1)

# %% [markdown]
# - Create dataframe indicating which SSIDs have changed and .head(5) the result

# %%
df_2 = df_1.query("(SSID_2012 != SSID_2023)")
print(df_2.head(5))


