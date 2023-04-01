def keep_regs(df, regs):
    """ Example function. Keep only the subset regs of regions in data.

    Args:
        df (pd.DataFrame): pandas dataframe 

    Returns:
        df (pd.DataFrame): pandas dataframe

    """ 
    
    for r in regs:
        I = df.reg.str.contains(r)
        df = df.loc[I == False] # keep everything else
    
    return df
    
#DATA PROJECT

#Import all necessary libraries

import pandas as pd
import numpy as np

#Import table functions
import pandas.io.formats.style as style
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tabulate import tabulate

#Import all neccesarry data files inCSV format

AAPL_data = pd.read_csv('AAPL.csv', sep=',')  
AMZN_data = pd.read_csv('AMZN.csv', sep=',')   
IBM_data = pd.read_csv('IBM.csv', sep=',')  
INTC_data = pd.read_csv('INTC.csv', sep=',')  
MSFT_data = pd.read_csv('MSFT.csv', sep=',')   

#1: TABLE OVER APPLE STOCK SHOWING yearly prices and volume from 31 dec. 2000 to 31 dec. 2022

# Convert the 'Date' column to datetime format
AAPL_data['Date'] = pd.to_datetime(AAPL_data['Date'])

# Filter the data for the past two years
AAPL_yearly__data = AAPL_data[AAPL_data['Date'] >= '2000-01-01']

# Create a new DataFrame with only the desired columns
AAPL_table = AAPL_yearly__data.iloc[:, [0, 1, 2, 3, 4,6]]

# Print the resulting table
print(AAPL_table)

#2: Converting the data to yearly data

# Convert the 'Date' column to datetime format and set it as the index
AAPL_data['Date'] = pd.to_datetime(AAPL_data['Date'])
AAPL_data.set_index('Date', inplace=True)

# Resample the data at a yearly frequency thus select the last value of each year
AAPL_yearly_data = AAPL_data.resample('Y').last() #'Y' is a frquency ALIAS for YEAR. The frequency ALIAS for monthly is 'M', daily is 'D'...

# Drop the 'Adj Close' column
AAPL_yearly_data = AAPL_yearly_data.drop(columns=['Adj Close'])

# Reset the index to include the 'Date' column as a regular column
AAPL_yearly_data.reset_index(inplace=True)

# Deleting las row in table output since we don't have data for the whole year of 2023 yet
AAPL_yearly_data=AAPL_yearly_data.iloc[:-1]

print(AAPL_yearly_data)

#3: Formatting the Data

#Formate the data to only include two decimals i.e 2f
AAPL_yearly_data['Open'] = AAPL_yearly_data['Open'].round(2)
AAPL_yearly_data['High'] = AAPL_yearly_data['High'].round(2)
AAPL_yearly_data['Low'] = AAPL_yearly_data['Low'].round(2)
AAPL_yearly_data['Close'] = AAPL_yearly_data['Close'].round(2)
AAPL_yearly_data['Volume'] = AAPL_yearly_data['Volume'].apply(lambda x: '{:,}'.format(x))

# Renaming the columns
AAPL_yearly_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# Table formatting and printing
table = tabulate(AAPL_yearly_data, headers='keys', tablefmt='grid', showindex=False)
print(table)

#4: Graph illustrating the APPLE stock price over time using the yearly close price

# Resample again the data at a yearly frequency and select the closing price for each year
AAPL_yearly_close = AAPL_data['Close'].resample('Y').last()

# Use the plot librabry
fig, ax = plt.subplots()
ax.plot(AAPL_yearly_close.index, AAPL_yearly_close.values)

# Format the x-axis labels to show only the year
date_fmt = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(date_fmt)

# Set the axis labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Closing Stock Price ($)')
ax.set_title('Yearly Apple Stock Price')

# Show the plot
plt.show()

#5: Making the graph more stylish

# Choose a style theme
plt.style.use('seaborn-bright')

# There are many other styles availble that can be found by printing the below
print(plt.style.available)

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 8)) #If desired the graph size can be smaller

# Plot the data with a line of a different color and thickness
ax.plot(AAPL_yearly_close.index, AAPL_yearly_close.values, color='#0077C8', linewidth=2)

# Set the background color of the plot area
ax.set_facecolor('#E8E8E8') #Code for dark grey color

# Set the tick label size and color
ax.tick_params(axis='both', which='major', labelsize=12, labelcolor='#666666') #Hex color, respresenting a dark grey color

# Format the x-axis labels to show only the year
date_fmt = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(date_fmt)

# Set the axis labels and title
ax.set_xlabel('Year', fontsize=14, fontweight='bold', color='#666666')
ax.set_ylabel('Closing Stock Price ($)', fontsize=14, fontweight='bold', color='#666666')
ax.set_title('Yearly Apple Stock Price (2000-2022)', fontsize=16, fontweight='bold', color='#333333')

# Add a horizontal grid line and adjust the y-axis limits
ax.grid(axis='y', linestyle='--', color='#999999', alpha=0.5)
ax.set_ylim(bottom=0)

# Add a legend to the plot
ax.legend(['APPL'], loc='upper left', fontsize=12)

# Add annotation to the plot
note = 'Source: Yahoo Finance.'
ax.annotate(note, xy=(0.5, -0.1), xycoords='axes fraction', fontsize=12, ha='center', va='center')

#save the figure as a file in the directory
fig.savefig('apple_stock_price.png', dpi=300)

# Show the plot
plt.show()

# 6: Making a graph that illustrates the monthly volume on y1 axis and closing price at y2
# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 8))

AAPL_monthly_volume = AAPL_data['Volume'].resample('M').last() #Not necesaary as data is already monthly data, but just for formality

# Plot the closing price with a line of a different color and thickness
ax.plot(AAPL_monthly_volume.index, AAPL_monthly_volume.values, color='#0077C8', linewidth=2)

# Create a twin axis for the volume data
ax2 = ax.twinx()

# Plot the volume as a bar chart on the twin axis
ax2.bar(AAPL_yearly_close.index, AAPL_yearly_close.values, color='#70B8FF', width=80)

# Set the background color of the plot area
ax.set_facecolor('#E8E8E8')

# Set the tick label size and color
ax.tick_params(axis='both', which='major', labelsize=12, labelcolor='#666666')
ax2.tick_params(axis='both', which='major', labelsize=12, labelcolor='#666666')

# Format the x-axis labels to show only the year
date_fmt = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(date_fmt)
ax2.xaxis.set_major_formatter(date_fmt)

# Set the axis labels and title
ax.set_xlabel('Year', fontsize=14, fontweight='bold', color='#666666')
ax2.set_ylabel('Closing Stock Price ($)', fontsize=14, fontweight='bold', color='#666666')
ax.set_ylabel('Volume', fontsize=14, fontweight='bold', color='#666666')
ax.set_title('Yearly Apple Stock Price (2000-2022)', fontsize=16, fontweight='bold', color='#333333')

# Add a horizontal grid line and adjust the y-axis limits
ax.grid(axis='y', linestyle='--', color='#999999', alpha=0.5)
ax.set_ylim(bottom=0)

# Add a legend to the plot
ax2.legend(['AAPL Close'], loc='upper left', fontsize=12)
ax.legend(['AAPL Volume'], loc='upper right', fontsize=12)

# If desired this can be saved in the directory with the code with the code for the previous figure

plt.show()

#7: Compare with other stocks in the same sector

# Convert date column to datetime format and set as index
AAPL_data['Date'] = pd.to_datetime(AAPL_data['Date'])
AAPL_data.set_index('Date', inplace=True)
AMZN_data['Date'] = pd.to_datetime(AMZN_data['Date'])
AMZN_data.set_index('Date', inplace=True)
MSFT_data['Date'] = pd.to_datetime(MSFT_data['Date'])
MSFT_data.set_index('Date', inplace=True)
IBM_data['Date'] = pd.to_datetime(IBM_data['Date'])
IBM_data.set_index('Date', inplace=True)
INTC_data['Date'] = pd.to_datetime(INTC_data['Date'])
INTC_data.set_index('Date', inplace=True)

# Resample data at a yearly frequency and select the last value of each year (WE illustrate by using dec. 31)
AAPL_yearly_data = AAPL_data.resample('Y').last()['Close']
AMZN_yearly_data = AMZN_data.resample('Y').last()['Close']
MSFT_yearly_data = MSFT_data.resample('Y').last()['Close']
INTC_yearly_data = INTC_data.resample('Y').last()['Close']
IBM_yearly_data = IBM_data.resample('Y').last()['Close']

# Create a plot with multiple lines
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(AAPL_yearly_data.index, AAPL_yearly_data.values, color='#0077C8', linewidth=2, label='Apple')
ax.plot(AMZN_yearly_data.index, AMZN_yearly_data.values, color='#FF9900', linewidth=2, label='Amazon')
ax.plot(MSFT_yearly_data.index, MSFT_yearly_data.values, color='#00CC66', linewidth=2, label='Microsoft')
ax.plot(IBM_yearly_data.index, IBM_yearly_data.values, color='#FFC0CB', linewidth=2, label='IBM')
ax.plot(INTC_yearly_data.index, INTC_yearly_data.values, color='#00FF00', linewidth=2, label='Intel Corporation')

# Set the background color of the plot area
ax.set_facecolor('#E8E8E8')

# Format the x-axis labels to show only the year
date_fmt = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(date_fmt)

# Set the axis labels and title
ax.set_xlabel('Year', fontsize=14, fontweight='bold', color='#666666')
ax.set_ylabel('Closing Stock Price ($)', fontsize=14, fontweight='bold', color='#666666')
ax.set_title('Yearly Stock Prices of Apple, Amazon, Intel, IBM and Microsoft (2000-2022)', fontsize=15, fontweight='bold', color='#333333')

# Add a horizontal grid line and adjust the y-axis limits
ax.grid(axis='y', linestyle='--', color='#999999', alpha=0.5)
ax.set_ylim(bottom=0)

# Add a legend to the plot
ax.legend(loc='upper left', fontsize=12)

# Add annotation to the plot
note = 'Source: Yahoo Finance.'
ax.annotate(note, xy=(0.5, -0.1), xycoords='axes fraction', fontsize=12, ha='center', va='center')

# Show the plot
plt.show()
