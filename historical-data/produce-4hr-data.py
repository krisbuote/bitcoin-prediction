import pandas as pd

''' This script takes 1-minute bitcoin data in a csv and deletes all data except
for data on 4 hour intervals.
'''
path = './coinbaseUSD_1-min_data_2014-12-01_to_2018-06-27.csv'

df = pd.read_csv(path)

df.describe()

#Want 4 hour data - time steps between is:
gap = 4*60 # Time in minutes
print("Entries reducing from {0} to {1}".format(len(df), len(df)/gap))


i = 0
## BUG: THIS while LOOP LEAVES LAST 'GAP' Entries####
while i + gap < df.shape[0]:
    print(i)
    df = df.drop(df.index[i+1:i+gap])
    i += 1

## Drop the last n = gap entries ##

print("Dropping last {0} entries.".format(gap))
df.drop(df.tail(gap).index,inplace=True)

df.to_csv('./coinbaseUSD_4-hr_data_2014-12-01_to_2018-06-27.csv', index=False)

print("Done.")
