'''
Training Pandas commands
pokemon_data.csv        - comma separated
pokemon_data.xslx
pokemon_data.txt        - tab separated

'''

import pandas as pd
import re

''' Loading data into Pandas (CSVs, Excel, TXTs, ... '''
## 1 - .csv file
df = pd.read_csv("Pokemon_Data/pokemon_data.csv")
#print(df.head(3))

## 2 - .xlsx file
#df_xlsx = pd.read_excel("./Pokemon_Data/pokemon_data.xlsx")
#print(df_xlsx.tail()) # 5 by default

## 3 - .txt file
#df_txt = pd.read_csv("./Pokemon_Data/pokemon_data.txt", delimiter='\t')
#print(df_txt.head())


''' Reading data (getting rows, columns, headers, ...) '''
## read headers
df.columns

## read each column
#print(df['Name'][0:8])              # you can use df.Name, but it would not work with more than one-word key like 'Pokemon Name'
df[['Name', 'Type 1', 'HP']].head(10)


''' Iterate through each Row '''
## get specific row by index or indices range
df.iloc[2:6]                 # iloc = integer location
df.iloc[1, 3]               # 2nd row, 4th column

## iterate
counter = 0
for index, row in df.iterrows():
    if counter >= 5:
        break
    #print(index, row['Name'])
    counter+=1

''' Getting columns names '''
list(df.columns)


''' Getting rows based on a specific condition '''
## one filter/condition
df.loc[df['Type 1']  == 'Fire']  # function used to find specific data (not only integer based)

## create a new df after filtering (NOTICE the old index column values are the same)
new_df = df.loc[(df['Type 1']  == 'Grass') & (df['Type 2']  == 'Poison') & (df['HP'] > 70)]

## to update index values
new_df.reset_index(drop=True, inplace=True)      #new column created, drop old indices || inplace True saves memory - override new_df variable
## save filtered data
new_df.to_csv('./Pokemon_Data/filtered.txt', sep='\t')

''' High level description of your data - min, max, mean, std dev, etc. '''
df.describe()
df.describe()['HP']
df.describe()['HP']['std']


''' Sorting values (numerically and alphabetically) '''
df.sort_values('Name', ascending=False)
df.sort_values(['Name', 'HP'], ascending=[True, False])     # sort by Name ascending and then by HP descending


''' Making changes to the DataFrame '''
## add new column to sum up some data
df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + \
              df['Sp. Def'] + df['Speed']
## also would work
#df['Total'] = df.iloc[:, 4:10].sum(axis=1)      # remember that hardcoded values are stinky =]. Could get index of column name first in var

#df.drop(columns=['Total'])               # DOES NOT work
#df = df.drop(columns=['Total'])          # DOES work

## save new df to a file
df.to_csv('./Pokemon_Data/modified.csv', index=False)   #index False - get rid of first indeces column if unncecessary


''' Summing multiple columns to create a new column '''
''' Rearranging columns '''
''' Filtering data (based on multiple condition)'''
''' Reset index '''


'''Regex filtering (textual patterns) '''
df.loc[df['Name'].str.contains('Mega')]
df.loc[~df['Name'].str.contains('Mega')]            # ~  means NOT (!)

## but we can pass RegEx instead...
df.loc[df['Type 1'].str.contains('fire|Grass', flags=re.I, regex=True)]             #re.I - ignore case (capital letters)
df.loc[df['Name'].str.contains('^pi[a-z]*', flags=re.I, regex=True)]             #re.I - ignore case (capital letters)


''' Conditional changes '''
## assume we don't like the name 'Fire' for Type 1. Let's change it to Flame
#      ~~~~~~~~  rows  ~~~~~ ,  columns
df.loc[df['Type 1'] == 'Fire', 'Type 1'] = 'Flame'
## change it back
df.loc[df['Type 1'] == 'Flame', 'Type 1'] = 'Fire'

## add new column based on condition - all Fire pokemons are legendary
df.loc[df['Type 1'] == 'Fire', 'Legendary'] = True
df.loc[df['Total'] > 500, ['Generation', 'Legendary']] = [5, True]      # list or tuple

## drop the new column OR load data again from a file (modified.csv - our checkpoint)
df = df.drop(columns=['Legendary'])
# df = pd.read_csv(./Pokemon_Data/modified.csv)

''' Aggregate statistics using Groupby (Sum, Mean, Counting) '''
## group pokemon e.g. by type
# mean
df.groupby(['Type 1']).mean().sort_values('Defense', ascending=False)
df.groupby(['Type 1']).mean().sort_values('Attack', ascending=False)
df.groupby(['Type 1']).mean().sort_values('HP', ascending=False)

# sum
df.groupby(['Type 1']).sum()

# count - see how many pokemons there is by Type and then by Type 2 within those Types
df['Count'] = 1 # add a column to count them
df.groupby(['Type 1', 'Type 2']).count()['Count']


''' Working with large amounts of data (setting chunksize) '''
## e.g 20 GB - read it in chunks  (you cannot load all 20GB into memory)
new_df = pd.DataFrame(columns=df.columns)

for df in pd.read_csv('Pokemon_Data/modified.csv', chunksize=5):
    # each df is 5 rows
    results = df.groupby(['Type 1']).count()
    new_df = pd.concat([new_df, results])

