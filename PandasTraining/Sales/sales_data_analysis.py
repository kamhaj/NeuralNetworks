import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
import os


''' Task #1: Merging 12 months of sales data (12 .cvs files) into a single file'''
# ## get files list from given directory (using list comprehension)
# files = [file for file in os.listdir("Sales_Data")]
#
# ## empty df creation
# all_months_data = pd.DataFrame()
#
# for file in files:
#     df = pd.read_csv("./Sales_Data/" + file)
#     all_months_data = pd.concat([all_months_data, df])
#
# ## save new file
# all_months_data.to_csv("Sales_Year_2019.csv", index=False)          # False index - omit first column


all_data = pd.read_csv("Sales_Year_2019.csv")




''' Clean up the data '''
## find rows of NaN values
nan_df = all_data[all_data.isna().any(axis=1)]      # 545 rows
#print(len(all_data))                               # 186850 rows

## drop rows of NaN values
all_data = all_data.dropna(how='any')               # if any NaN value in a row
#print(len(all_data))                               # 186305 rows




''' Augment data with additional columns '''
## add month column
all_data['Month'] = all_data['Order Date'].str[:2]

## try to convert to integer
# ValueError: invalid literal for int() with base 10: 'Or' - drop rows where Order Date has value 'Order Date'... for some reason...
all_data = all_data[all_data['Order Date'] != 'Order Date']         # 185950 rows
# convert to int
all_data['Month'] = all_data['Month'].astype('int32')

## add sales column
# convert columns to a right type (from string to int)
all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered'])      # make int
all_data['Price Each'] = pd.to_numeric(all_data['Price Each'])                  # make float
# add sales column (now we can multiple int * float)
all_data['Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']




''' What was the best month for sales and how much we have earned? '''
## group by months and add up
results = all_data.groupby('Month').sum()

## plot grouped values - barchar
#months = range(1,13)
#plt.bar(months, results['Sales'])
#plt.xticks(months)              # show ALL months on axis (not every 2)
#plt.ylabel('Sales in USD ($)')
#plt.xlabel('Month number')
#plt.show()





''' What city had the highest number of sales '''
## grab city
def get_city(address):
    return address.split(',')[1]

## different cities may have the same name - grab a state too
def get_state(address):
    return (address.split(',')[2]) .split(' ')[1]

## add a City column based on Address - .apply() method
all_data['City'] = all_data['Purchase Address'].apply(lambda x: f'{get_city(x)} ({get_state(x)})')    # x is cell content
                                                     #lambda x: x.split(',')[1] - also works


## so... what city had the highest number of sales?
results = all_data.groupby('City').sum()
list_of_cities = [city for city, df in all_data.groupby('City')]  # it will keep the order for x-axis accordingly to y-axis

## plot it
#plt.bar(list_of_cities, results['Sales'])
#plt.xticks(list_of_cities, rotation='vertical', size=8)              # show ALL months on axis (not every 2)
#plt.ylabel('Sales in USD ($)')
#plt.xlabel('City name')
#plt.show()





''' What time should we display advertisements to maximize likelihood of customer's buying products '''
# ## use order date and check its distribution over 24 hours
# # convert current date format (because it may change) to datetime object
# all_data['Order Date'] = pd.to_datetime(all_data['Order Date'])
# ## add hour and minute columns
# all_data['Order Hour'] = all_data['Order Date'].dt.hour
# all_data['Order Minute'] = all_data['Order Date'].dt.minute
# ## save new data
# all_data.to_csv('order_hour_column_added.csv', index=False)
## read saved data
all_data = pd.read_csv('order_hour_column_added.csv')

## plot it - distribution over 24 hours
# hours = [hour for hour, df in all_data.groupby('Order Hour')]
# print(hours)
# print(all_data.groupby('Order Hour'))
# plt.plot(hours, all_data.groupby('Order Hour').count(), color='g')
# plt.xticks(hours)              # show ALL months on axis (not every 2)
# plt.ylabel('No of orders')
# plt.xlabel('Hour of a day')
# plt.grid(color='b', linestyle='dashed', linewidth=0.2)
# plt.show()





''' What products are most often sold together '''
## keep only duplicate Order IDs
df = all_data[all_data['Order ID'].duplicated(keep=False)]

## get products from same Order ID into one line
df["Grouped"] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
# drop duplicated Order IDs
df = df[['Order ID', 'Grouped']].drop_duplicates()

## count pairs of what is sold together
count = Counter()

for row in df['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list, 2)))

#print(count.most_common(10))

# # if you would like to see the most common 3 items ordered together...
# count_triples = Counter()
#
# for row in df['Grouped']:
#     row_list = row.split(',')
#     count_triples.update(Counter(combinations(row_list, 3)))
#
# print(count_triples.most_common(10))




''' What product sold the most? Why do you think it sold the most? '''
product_group = all_data.groupby('Product')
quantity_ordered = product_group.sum()['Quantity Ordered']
products = [product for product, df in product_group]
print(product_group)

## plot it
# plt.bar(products, quantity_ordered, color='g')
# plt.xticks(products, rotation='80', size=8)              # show ALL months on axis (not every 2)
# plt.xlabel('Product')
# plt.ylabel('Quantity ordered')
# plt.grid(color='pink', linestyle='dashed', linewidth=0.5)
# plt.show()

## overlay previous graph with price tags to see if there is quantity-price correlation
prices = all_data.groupby('Product').mean()['Price Each']
# keep the same xlabel and plot a new y-axis in the same window (add subplot)
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.bar(products, quantity_ordered, color='g')
ax2.plot(products, prices, 'b-')

ax1.set_xlabel('Product Name')
ax1.set_ylabel('Quantity Ordered',  color='g')
ax2.set_ylabel('Price ($)',  color='b')
ax1.set_xticklabels(products, rotation='80', size=8)

plt.show()

