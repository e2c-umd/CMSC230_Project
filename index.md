# The Relationship Between Crime Rates and External Factors In Chicago
Contributor: 
* Evan Chen

Contributions:
* Evan Chen: all


### Introduction
In the constant effort to keep our living spaces safer, we are always looking for ways to mitigate crime rates. In order to develop effective ways to do so, we need to look at some of the root causes of crime itself; the factors that influence crime rates. This will allow us to develop preventative measures and take legal action to stop these crimes before they happen.

This project will take us through the data science life cycle, and allow us to see how data science and statistics can let us develop meaningful conclusions that have the potential to improve or even save the lives of millions of people



```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

```

### Datasets and Data Cleaning

In the early stages of the data science life cycle, we need to get our datatsets and filter them for the data we want.
In this case I have opted to use data from Kaggle:
* https://www.kaggle.com/datasets/alistairking/electricity-prices 
* https://www.kaggle.com/datasets/middlehigh/los-angeles-crime-data-from-2000 
* https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities 
* https://www.kaggle.com/datasets/varpit94/us-inflation-data-updated-till-may-2021 


For this project, we are focusing on the city of Chicago to do our data analysis. 
The external factors we are going to be considering for this project are:
* Temperature
* Consumer Price Index (general metric for cost of living)
* Electricity Prices

In the real world, we could consider dozens or even hundreds of different factors. However, these 3 factors will let us perform sufficient analysis for our project.

We are going to filter out the data we don't want, and save it as a new CSV for easy access. 

**Only run the cell below[3] ONE time!!**



```python


#Data Cleaning

#only get national CPI from 1995 onwards
#I dont need CPI from like 1904
df = pd.read_csv('US CPI.csv')
df['Yearmon'] = pd.to_datetime(df['Yearmon'], format='%d-%m-%Y')
df = df[df['Yearmon'] >= '1995-01-01']
df.to_csv('US CPI filtered.csv', index = False)

#filter for chicago
df = pd.read_csv('city_temperature.csv')
df = df[df['City'] == 'Chicago']
df.to_csv('city_temp_chicago.csv', index=False)

#Clean electricity prices to filter for the state 
#of Illinois
df = pd.read_csv('US_Electricity_Prices.csv')
df = df[df['stateDescription'] == 'Illinois']
df.to_csv('US_Electricity_Illinois.csv', index = False)


```

    C:\Users\evan3\AppData\Local\Temp\ipykernel_15940\2352892721.py:11: DtypeWarning: Columns (2) have mixed types. Specify dtype option on import or set low_memory=False.
      df = pd.read_csv('city_temperature.csv')
    

Run this next cell to import the data we are going to use for the rest of this project.

Might take 30-40 seconds since one of our datasets is very large


```python
#immport our filtered datasets
cpi = pd.read_csv('US CPI filtered.csv')
temperature = pd.read_csv('city_temp_chicago.csv')
electricity = pd.read_csv('US_Electricity_Illinois.csv')

#chicago crime data cleaning. 
#this file is massive so i dont want to make a copy of it 
df = pd.read_csv('Chicage Crime Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

#this is a function we use later
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())
```

    C:\Users\evan3\AppData\Local\Temp\ipykernel_15940\2496311217.py:8: DtypeWarning: Columns (8,9) have mixed types. Specify dtype option on import or set low_memory=False.
      df = pd.read_csv('Chicage Crime Data.csv')
    

Lets have a look at our data:


```python
cpi.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Yearmon</th>
      <th>CPI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1995-01-01</td>
      <td>150.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1995-02-01</td>
      <td>150.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1995-03-01</td>
      <td>151.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1995-04-01</td>
      <td>151.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1995-05-01</td>
      <td>152.2</td>
    </tr>
  </tbody>
</table>
</div>



'CPI' is the Consumer Price Index for the year and month given by 'Yearmon'.
However, this metric is for the US as a whole, not Chicago specifically.
For more info on CPI: https://www.investopedia.com/terms/c/consumerpriceindex.asp 


```python
temperature.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Country</th>
      <th>State</th>
      <th>City</th>
      <th>Month</th>
      <th>Day</th>
      <th>Year</th>
      <th>AvgTemperature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>North America</td>
      <td>US</td>
      <td>Illinois</td>
      <td>Chicago</td>
      <td>1</td>
      <td>1</td>
      <td>1995</td>
      <td>23.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>North America</td>
      <td>US</td>
      <td>Illinois</td>
      <td>Chicago</td>
      <td>1</td>
      <td>2</td>
      <td>1995</td>
      <td>13.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>North America</td>
      <td>US</td>
      <td>Illinois</td>
      <td>Chicago</td>
      <td>1</td>
      <td>3</td>
      <td>1995</td>
      <td>14.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>North America</td>
      <td>US</td>
      <td>Illinois</td>
      <td>Chicago</td>
      <td>1</td>
      <td>4</td>
      <td>1995</td>
      <td>3.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>North America</td>
      <td>US</td>
      <td>Illinois</td>
      <td>Chicago</td>
      <td>1</td>
      <td>5</td>
      <td>1995</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



Our 'temperature' dataset gives us the average temperature for every day since January 1, 1995, until 2020, in the city of Chicago


```python
electricity.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>month</th>
      <th>stateDescription</th>
      <th>sectorName</th>
      <th>customers</th>
      <th>price</th>
      <th>revenue</th>
      <th>sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>1</td>
      <td>Illinois</td>
      <td>all sectors</td>
      <td>NaN</td>
      <td>5.87</td>
      <td>727.44186</td>
      <td>12382.56439</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>1</td>
      <td>Illinois</td>
      <td>commercial</td>
      <td>NaN</td>
      <td>5.81</td>
      <td>214.59817</td>
      <td>3692.91148</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>1</td>
      <td>Illinois</td>
      <td>industrial</td>
      <td>NaN</td>
      <td>4.03</td>
      <td>140.61220</td>
      <td>3489.34187</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>1</td>
      <td>Illinois</td>
      <td>other</td>
      <td>NaN</td>
      <td>5.44</td>
      <td>50.24250</td>
      <td>924.09617</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>1</td>
      <td>Illinois</td>
      <td>residential</td>
      <td>NaN</td>
      <td>7.53</td>
      <td>321.98899</td>
      <td>4276.21487</td>
    </tr>
  </tbody>
</table>
</div>



Our 'electricity' dataset gives us the the prices, revenue, and sales of the different sectors in the state of Illinois.
* Price is in cents per Kilowatt-hour (kWh)
* Revenue is in millions of dollars
* Sales is in millions of Kilowatt-hours (kWh)


```python
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Case Number</th>
      <th>Date</th>
      <th>Block</th>
      <th>IUCR</th>
      <th>Primary Type</th>
      <th>Description</th>
      <th>Location Description</th>
      <th>Arrest</th>
      <th>Domestic</th>
      <th>...</th>
      <th>Community Area</th>
      <th>FBI Code</th>
      <th>X Coordinate</th>
      <th>Y Coordinate</th>
      <th>Year</th>
      <th>Updated On</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Location</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5741943</td>
      <td>HN549294</td>
      <td>2007-08-25 09:22:18</td>
      <td>074XX N ROGERS AVE</td>
      <td>0560</td>
      <td>ASSAULT</td>
      <td>SIMPLE</td>
      <td>OTHER</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>1.0</td>
      <td>08A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2007.0</td>
      <td>08/17/2015 03:03:40 PM</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25953</td>
      <td>JE240540</td>
      <td>2021-05-24 15:06:00</td>
      <td>020XX N LARAMIE AVE</td>
      <td>0110</td>
      <td>HOMICIDE</td>
      <td>FIRST DEGREE MURDER</td>
      <td>STREET</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>19.0</td>
      <td>01A</td>
      <td>1141387.0</td>
      <td>1913179.0</td>
      <td>2021.0</td>
      <td>11/18/2023 03:39:49 PM</td>
      <td>41.917838</td>
      <td>-87.755969</td>
      <td>(41.917838056, -87.755968972)</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26038</td>
      <td>JE279849</td>
      <td>2021-06-26 09:24:00</td>
      <td>062XX N MC CORMICK RD</td>
      <td>0110</td>
      <td>HOMICIDE</td>
      <td>FIRST DEGREE MURDER</td>
      <td>PARKING LOT</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>13.0</td>
      <td>01A</td>
      <td>1152781.0</td>
      <td>1941458.0</td>
      <td>2021.0</td>
      <td>11/18/2023 03:39:49 PM</td>
      <td>41.995219</td>
      <td>-87.713355</td>
      <td>(41.995219444, -87.713354912)</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13279676</td>
      <td>JG507211</td>
      <td>2023-11-09 07:30:00</td>
      <td>019XX W BYRON ST</td>
      <td>0620</td>
      <td>BURGLARY</td>
      <td>UNLAWFUL ENTRY</td>
      <td>APARTMENT</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>5.0</td>
      <td>05</td>
      <td>1162518.0</td>
      <td>1925906.0</td>
      <td>2023.0</td>
      <td>11/18/2023 03:39:49 PM</td>
      <td>41.952345</td>
      <td>-87.677975</td>
      <td>(41.952345086, -87.677975059)</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13274752</td>
      <td>JG501049</td>
      <td>2023-11-12 07:59:00</td>
      <td>086XX S COTTAGE GROVE AVE</td>
      <td>0454</td>
      <td>BATTERY</td>
      <td>AGGRAVATED P.O. - HANDS, FISTS, FEET, NO / MIN...</td>
      <td>SMALL RETAIL STORE</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>44.0</td>
      <td>08B</td>
      <td>1183071.0</td>
      <td>1847869.0</td>
      <td>2023.0</td>
      <td>12/09/2023 03:41:24 PM</td>
      <td>41.737751</td>
      <td>-87.604856</td>
      <td>(41.737750767, -87.604855911)</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



Our 'df' dataframe is the main focus of this project. It contains 7,391,187 criminal cases as well as a bunch of information associated with each case. 

We will mostly focus on the following columns:
* Primary Type: the category of the crime
* Arrest: whether or not an arrest was made
* Domestic: whether or not the case was a domestic crime or not
* Date: the day the crime occurred 



## Data Visualization
Now that we have cleaned our datasets, lets see what our datasets really look like.


### Part 1


To start, lets look at the relationships between our finance-related datasets. 
We expect the general trend for CPI and electricity prices to be increasing due to inflation.

We are going to plot electricty prices, revenue, and CPI over time.
To do this, we have to do a little bit of pre-processing to make sure our plot comes out how we want


```python

#initialize a date range for the plot
full_date_range = pd.date_range(start='2001-01-01', end=pd.to_datetime('today'), freq='M')

#make sure our dates are in datetime format
cpi['YearMonth'] = pd.to_datetime(cpi['Yearmon'], errors='coerce').dt.to_period('M')

electricity['YearMonth'] = pd.to_datetime(electricity[['year', 'month']].assign(day=1))
electricity['YearMonth'] = electricity['YearMonth'].dt.to_period('M')

#make sure our datasets are ordered in ascending order by date
#we need to groupby yearmonth because of the formatting differences of some of our datasets

avg_cpi = cpi.groupby('YearMonth')['CPI'].mean()
avg_cpi = avg_cpi.reindex(full_date_range.to_period('M'))
 
avg_electricity_price = electricity.groupby('YearMonth')['price'].mean()
avg_electricity_price = avg_electricity_price.reindex(full_date_range.to_period('M'))


avg_electricity_rev = electricity.groupby('YearMonth')['revenue'].mean()
avg_electricity_rev = avg_electricity_rev.reindex(full_date_range.to_period('M'))


```

    C:\Users\evan3\AppData\Local\Temp\ipykernel_15940\558707268.py:2: FutureWarning: 'M' is deprecated and will be removed in a future version, please use 'ME' instead.
      full_date_range = pd.date_range(start='2001-01-01', end=pd.to_datetime('today'), freq='M')
    

There are differences in the orders of magnitude of our datasets. If we aren't careful, our visualization will look like a child's depiction of a landscape. A green line at the bottom for the grass and a blue line at the top for the sky. 

To get around this, we are going to normalize everything so that we can see the overall trend over time more clearly.



```python

normalized_cpi = normalize(avg_cpi)
normalized_electricity_price = normalize(avg_electricity_price)
normalized_electricity_rev = normalize(avg_electricity_rev)


#time to plot
plt.figure(figsize=(12, 6))

plt.plot(full_date_range, normalized_cpi.values, label='CPI')
plt.plot(full_date_range, normalized_electricity_price.values, label='Electricity Price')
plt.plot(full_date_range, normalized_electricity_rev.values, label='Electricity Revenue')

# Customize the plot
plt.title('CPI, Electricity Price, Electricity Revenue Over Time')
plt.xticks(ticks=pd.date_range(start='2001-01-01', end=pd.to_datetime('today'), freq='YS'), 
           labels=pd.date_range(start='2001-01-01', end=pd.to_datetime('today'), freq='YS').strftime('%Y'), rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

```


    
![png](FinalProject_files/FinalProject_20_0.png)
    


Cool! It looks like all the curves have a general increase like we expected. It makes sense to see electricity prices follow a similar trend as CPI because we expect electricity prices to be a factor when calculating cost of living. 

Look at the spikes that happen every year around summer time. This might tell us that air conditioning takes a lot more energy than heating, which drives energy prices up. We can't say for sure, but it's the first guess that comes to mind.

If you want to learn how to make beautiful plots  like these, check out MatPlotLib: https://matplotlib.org/stable/index.html


## Part 2

Since we have observed a sudden jump in electricity prices during the summer time, we can hypothesize that these higher prices could make daily life more difficult, which could increase the likelihood of crimes being committed. So lets see if our guess has any merit.



```python




plt.figure(figsize=(10, 6))

monthly_counts = normalize(df['Month'].value_counts().reindex(range(1, 13), fill_value=0))

monthly_counts.plot(kind='bar', color='skyblue')
plt.title('Crime Frequency by Month')
plt.xlabel('Month')
plt.xticks(ticks=range(12), 
           labels=['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December'], 
           rotation=45)

plt.ylabel('Number of Entries (normalized)')
plt.grid(axis='y')

```


    
![png](FinalProject_files/FinalProject_23_0.png)
    


Interesting! It looks like our hypothesis isn't completely wrong.

However, we are good data scientists, and know that correlation is not necessarily causation.

The spike in crime rates during the summer months could be explained by many other factors we havent considered. 
I read online somewhere that when temperatures rise, people become irritable and more violent, which causes crimes related to temperament to increase. Lets test this hypothesis.
 
We are going to compare the distribution of homicide frequencies with other crimes that are less related to temperament. I've chosen narcotics and motor vehicle theft. 


```python
plt.figure(figsize=(10, 6))



crime_types = ['HOMICIDE', 'NARCOTICS', 'MOTOR VEHICLE THEFT']  # Ive experimented with a few more, but these three look the best
num_crimes = len(crime_types)
fig, axes = plt.subplots(nrows=1, ncols=num_crimes, figsize=(15, 6), sharey=True)

for ax, crime in zip(axes, crime_types):
    normalized_counts = normalize(df[df['Primary Type'] == crime]['Month'].value_counts().reindex(range(1, 13), fill_value=0))
    normalized_counts.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title(f'Number of Entries by Month for {crime}')
    ax.set_xticks(range(12))
    ax.set_xticklabels(['January', 'February', 'March', 'April', 'May', 'June', 
                        'July', 'August', 'September', 'October', 'November', 'December'], 
                       rotation=45)
    ax.grid(axis='y')

plt.ylabel('Number of Entries')
plt.tight_layout()
plt.show()



```


    <Figure size 1000x600 with 0 Axes>



    
![png](FinalProject_files/FinalProject_25_1.png)
    


Wow! Looks like there is a pretty clear difference in the distribution of crimes throughout the months! 

Notice how the y axis is between 0 and 1. We have normalized our data once again because of the difference in frequencies across different crime categories. There are going to be a lot less homicides than drug related crimes. What we care about is how the frequencies of these crimes change throughout different times of the year.

Now that we've had a good look at all our datasets, lets see how all of them might be related. 
Lets plot out our temperature, CPI, and electricity price curves over top our homicide frequency bar graph.




```python

#Since cpi, and electricity price are recorded over many years, 
# I will just get the average of each month over all the years available.


# Convert 'Year', 'Month', and 'Day' to datetime for temperature data
temperature['Date'] = pd.to_datetime(temperature[['Year', 'Month', 'Day']])

# Group by month and calculate average temperature
temperature['Month'] = temperature['Date'].dt.month
avg_temperature = temperature.groupby('Month')['AvgTemperature'].mean()

# need to xtract month from CPI data
cpi['Yearmon'] = pd.to_datetime(cpi['Yearmon'])
cpi['Month'] = cpi['Yearmon'].dt.month

avg_cpi = cpi.groupby(cpi['Yearmon'].dt.month)['CPI'].mean()
avg_electricity_price = electricity.groupby(['year', 'month'])['price'].mean().reset_index()
avg_electricity_price = avg_electricity_price.groupby('month')['price'].mean()



#I am going to normalize the datasets again because the actual numeric values
#in each category are vastly different. 
#I only care about the changes between months here.

normalized_temperature = normalize(avg_temperature)
normalized_cpi = normalize(avg_cpi)
normalized_electricity_price = normalize(avg_electricity_price)

#Plot all the categories as curves
plt.figure(figsize=(12, 6))
plt.plot(normalized_temperature.index, normalized_temperature.values, label='Average Temperature', marker='o')
plt.plot(normalized_cpi.index, normalized_cpi.values, label='CPI', marker='o')
plt.plot(normalized_electricity_price.index, normalized_electricity_price.values, label='Electricity Price', marker='o')

# Plot HOMICIDE as a bar graph because its sort of the main character here
homicide_counts = normalize(df[df['Primary Type'] == 'HOMICIDE']['Month'].value_counts().reindex(range(1, 13), fill_value=0))
plt.bar(homicide_counts.index, homicide_counts.values, label='Homicide Counts', alpha=0.5)

#add some nice labels 
plt.title('CPI, Temp, Electricity Price, compared to Homicide counts by month')
plt.xlabel('Month')
plt.xticks(normalized_temperature.index, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.grid()
plt.show()

```


    
![png](FinalProject_files/FinalProject_27_0.png)
    


Normalization is awesome!
It looks like on average, the maximum temperature, CPI, electricity price, and homicide rates all hit their maximums in July! 

Now that we have a good idea of the relationship between our datasets, we want to see which statistics are most related to which statistics. A correlation matrix will be perfect for this job.




```python
#Since we have seen a very nice correlation between our categories above, 
#lets make a correlation matrix 

#cool dataframe trick
data = {
    'temperature': normalized_temperature.values,
    'CPI': normalized_cpi.values,
    'electricity price': normalized_electricity_price.values,
    'homicide frequency': homicide_counts.values
}

correlation_df = pd.DataFrame(data)

#correlation matrix
correlation_matrix = correlation_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()



```


    
![png](FinalProject_files/FinalProject_29_0.png)
    


How do we read this??

**How to Read a Correlation Matrix**:
Notice that the labels on the X and Y axis contain the same elements. The square (x,y) corresponds to the correlation between x and y. Notice that all values on the diagonal are 1. This is because every statistic has perfect correlation with itself, so these squares are trivial. Also notice that the matrix is symmetric about  the diagonal. This is because the correlation between x and y is equal to the correlation between y and x.

It looks like out of all of our factors, the most correlated pair is temperature to homicide frequency, which supports the hypothesis that high temperatures are related to higher violent crime rates. Another highly correlated pair is temperature and electricity price, which supports our hypothesis we made earlier about air conditioning and energy prices.

## Machine Learning

Visualizations are cool and all, but we are data scientists, and believe it or not, we don't want to read and interpret charts all day. We want the computer to do that for us. By using the data we have collected, we want to train the computer to make predictions about new data. The computer might even be able to recognize patterns we can't see.

### Part 1A: Regression

Before we start, we want our datasets to be easily 'digestible' by our machine learning models.

We are going to combine the features and labels that we want into one nice dataframe that can be easily split into testing and training sets.



```python


#make a new column called 'date' 
#we will use this column to join our datasets together
temperature['date'] = pd.to_datetime(temperature[['Year', 'Month', 'Day']])

#crimes is the dataset we are going to join on
crimes = temperature[['date', 'AvgTemperature']].copy()

#rename so that this column is consistent across our many dataframes
crimes.rename(columns={'date': 'Date'}, inplace=True)

crimes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>AvgTemperature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1995-01-01</td>
      <td>23.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1995-01-02</td>
      <td>13.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1995-01-03</td>
      <td>14.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1995-01-04</td>
      <td>3.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1995-01-05</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#crime_counts is a dataframe that gives us the number of each crime that has occured on each date
#We take all the crimes that have occured on the same day, 
# and count the number of each type of crime that occured on that day
#then, we make each category into its own column
crime_counts = df.groupby(df['Date'].dt.date)['Primary Type'].value_counts().unstack(fill_value=0)

#since we have grouped by date, the our crime_counts dataframe is indexed by 'Date' 
#we will convert our date to pandas DateTime
crime_counts.index = pd.to_datetime(crime_counts.index)

crime_counts.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Primary Type</th>
      <th>ARSON</th>
      <th>ASSAULT</th>
      <th>BATTERY</th>
      <th>BURGLARY</th>
      <th>CONCEALED CARRY LICENSE VIOLATION</th>
      <th>CRIM SEXUAL ASSAULT</th>
      <th>CRIMINAL DAMAGE</th>
      <th>CRIMINAL SEXUAL ASSAULT</th>
      <th>CRIMINAL TRESPASS</th>
      <th>DECEPTIVE PRACTICE</th>
      <th>...</th>
      <th>OTHER OFFENSE</th>
      <th>PROSTITUTION</th>
      <th>PUBLIC INDECENCY</th>
      <th>PUBLIC PEACE VIOLATION</th>
      <th>RITUALISM</th>
      <th>ROBBERY</th>
      <th>SEX OFFENSE</th>
      <th>STALKING</th>
      <th>THEFT</th>
      <th>WEAPONS VIOLATION</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001-01-01</th>
      <td>0</td>
      <td>70</td>
      <td>296</td>
      <td>66</td>
      <td>0</td>
      <td>38</td>
      <td>233</td>
      <td>3</td>
      <td>29</td>
      <td>93</td>
      <td>...</td>
      <td>167</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>41</td>
      <td>65</td>
      <td>1</td>
      <td>412</td>
      <td>32</td>
    </tr>
    <tr>
      <th>2001-01-02</th>
      <td>0</td>
      <td>66</td>
      <td>143</td>
      <td>68</td>
      <td>0</td>
      <td>2</td>
      <td>118</td>
      <td>0</td>
      <td>35</td>
      <td>78</td>
      <td>...</td>
      <td>101</td>
      <td>11</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>35</td>
      <td>4</td>
      <td>0</td>
      <td>221</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2001-01-03</th>
      <td>1</td>
      <td>79</td>
      <td>165</td>
      <td>57</td>
      <td>0</td>
      <td>7</td>
      <td>136</td>
      <td>0</td>
      <td>35</td>
      <td>49</td>
      <td>...</td>
      <td>96</td>
      <td>16</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>51</td>
      <td>5</td>
      <td>1</td>
      <td>226</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2001-01-04</th>
      <td>1</td>
      <td>57</td>
      <td>173</td>
      <td>55</td>
      <td>0</td>
      <td>2</td>
      <td>133</td>
      <td>0</td>
      <td>29</td>
      <td>42</td>
      <td>...</td>
      <td>96</td>
      <td>19</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>243</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2001-01-05</th>
      <td>3</td>
      <td>68</td>
      <td>178</td>
      <td>55</td>
      <td>0</td>
      <td>6</td>
      <td>142</td>
      <td>0</td>
      <td>29</td>
      <td>53</td>
      <td>...</td>
      <td>90</td>
      <td>16</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>55</td>
      <td>2</td>
      <td>0</td>
      <td>265</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
# Merge the temperature data with the crime counts on the date
crimes = crimes.merge(crime_counts, left_on='Date', right_index=True, how='left')

#drop missing values because they cause indigestion for our ML models
crimes = crimes.dropna()

crimes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>AvgTemperature</th>
      <th>ARSON</th>
      <th>ASSAULT</th>
      <th>BATTERY</th>
      <th>BURGLARY</th>
      <th>CONCEALED CARRY LICENSE VIOLATION</th>
      <th>CRIM SEXUAL ASSAULT</th>
      <th>CRIMINAL DAMAGE</th>
      <th>CRIMINAL SEXUAL ASSAULT</th>
      <th>...</th>
      <th>OTHER OFFENSE</th>
      <th>PROSTITUTION</th>
      <th>PUBLIC INDECENCY</th>
      <th>PUBLIC PEACE VIOLATION</th>
      <th>RITUALISM</th>
      <th>ROBBERY</th>
      <th>SEX OFFENSE</th>
      <th>STALKING</th>
      <th>THEFT</th>
      <th>WEAPONS VIOLATION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2192</th>
      <td>2001-01-01</td>
      <td>15.2</td>
      <td>0.0</td>
      <td>70.0</td>
      <td>296.0</td>
      <td>66.0</td>
      <td>0.0</td>
      <td>38.0</td>
      <td>233.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>167.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>41.0</td>
      <td>65.0</td>
      <td>1.0</td>
      <td>412.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>2193</th>
      <td>2001-01-02</td>
      <td>11.8</td>
      <td>0.0</td>
      <td>66.0</td>
      <td>143.0</td>
      <td>68.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>118.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>101.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>221.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2194</th>
      <td>2001-01-03</td>
      <td>15.5</td>
      <td>1.0</td>
      <td>79.0</td>
      <td>165.0</td>
      <td>57.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>136.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>96.0</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>51.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>226.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2195</th>
      <td>2001-01-04</td>
      <td>24.6</td>
      <td>1.0</td>
      <td>57.0</td>
      <td>173.0</td>
      <td>55.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>133.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>96.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>55.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>243.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2196</th>
      <td>2001-01-05</td>
      <td>32.1</td>
      <td>3.0</td>
      <td>68.0</td>
      <td>178.0</td>
      <td>55.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>142.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>90.0</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>55.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>265.0</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




Right now the 'crimes' dataframe contains the daily average temperature and the number of crimes committed on that day for all crime categories. 
Since our CPI and electricity price datasets are dated by month, we will make some adjustments to accomodate:
* Sum up the total number of committed crimes of all categories in the month
* Get the month's median temperature.  

We could have filled in the monthly CPI and electricity price for each day in the month, but this could keep our data very 'noisy' since there might be high variation between the number of crimes committed per day. Additionally, the large amount of rows could make model training take longer than we have patience for.


```python


# Resample the data by month and calculate the median temperature for AvgTemperature
crime_months = crimes.resample('M', on='Date').agg({'AvgTemperature': 'median'})

# Sum the counts of all columns except the first (AvgTemperature) for the month
crime_months = crime_months.join(crimes.resample('M', on='Date').sum().iloc[:, 1:], how='left')

#Change the 'Date' column to 'YearMonth' to matc hthe names of the other columns
crime_months['YearMonth'] = crime_months.index.to_period('M').astype(str)


#for some reason it doesn't work if I don't convert the dates to strings first
cpi['YearMonth'] = cpi['YearMonth'].astype(str)
crime_months['YearMonth'] = crime_months['YearMonth'].astype(str)
electricity['YearMonth'] = electricity['YearMonth'].astype(str)

# Ensure the YearMonth column in all DataFrames are of the same type
cpi['YearMonth'] = pd.to_datetime(cpi['YearMonth'], format='%Y-%m')  # Convert to datetime
crime_months['YearMonth'] = pd.to_datetime(crime_months['YearMonth'], format='%Y-%m')  # Convert to datetime
electricity['YearMonth'] = pd.to_datetime(electricity['YearMonth'], format='%Y-%m')  # Convert to datetime




```

    C:\Users\evan3\AppData\Local\Temp\ipykernel_15940\3819388843.py:2: FutureWarning: 'M' is deprecated and will be removed in a future version, please use 'ME' instead.
      crime_months = crimes.resample('M', on='Date').agg({'AvgTemperature': 'median'})
    C:\Users\evan3\AppData\Local\Temp\ipykernel_15940\3819388843.py:5: FutureWarning: 'M' is deprecated and will be removed in a future version, please use 'ME' instead.
      crime_months = crime_months.join(crimes.resample('M', on='Date').sum().iloc[:, 1:], how='left')
    


```python
cpi.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Yearmon</th>
      <th>CPI</th>
      <th>YearMonth</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1995-01-01</td>
      <td>150.3</td>
      <td>1995-01-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1995-02-01</td>
      <td>150.9</td>
      <td>1995-02-01</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1995-03-01</td>
      <td>151.4</td>
      <td>1995-03-01</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1995-04-01</td>
      <td>151.9</td>
      <td>1995-04-01</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1995-05-01</td>
      <td>152.2</td>
      <td>1995-05-01</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
electricity.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>month</th>
      <th>stateDescription</th>
      <th>sectorName</th>
      <th>customers</th>
      <th>price</th>
      <th>revenue</th>
      <th>sales</th>
      <th>YearMonth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>1</td>
      <td>Illinois</td>
      <td>all sectors</td>
      <td>NaN</td>
      <td>5.87</td>
      <td>727.44186</td>
      <td>12382.56439</td>
      <td>2001-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>1</td>
      <td>Illinois</td>
      <td>commercial</td>
      <td>NaN</td>
      <td>5.81</td>
      <td>214.59817</td>
      <td>3692.91148</td>
      <td>2001-01-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>1</td>
      <td>Illinois</td>
      <td>industrial</td>
      <td>NaN</td>
      <td>4.03</td>
      <td>140.61220</td>
      <td>3489.34187</td>
      <td>2001-01-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>1</td>
      <td>Illinois</td>
      <td>other</td>
      <td>NaN</td>
      <td>5.44</td>
      <td>50.24250</td>
      <td>924.09617</td>
      <td>2001-01-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>1</td>
      <td>Illinois</td>
      <td>residential</td>
      <td>NaN</td>
      <td>7.53</td>
      <td>321.98899</td>
      <td>4276.21487</td>
      <td>2001-01-01</td>
    </tr>
  </tbody>
</table>
</div>




```python
crime_months.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AvgTemperature</th>
      <th>ARSON</th>
      <th>ASSAULT</th>
      <th>BATTERY</th>
      <th>BURGLARY</th>
      <th>CONCEALED CARRY LICENSE VIOLATION</th>
      <th>CRIM SEXUAL ASSAULT</th>
      <th>CRIMINAL DAMAGE</th>
      <th>CRIMINAL SEXUAL ASSAULT</th>
      <th>CRIMINAL TRESPASS</th>
      <th>...</th>
      <th>PROSTITUTION</th>
      <th>PUBLIC INDECENCY</th>
      <th>PUBLIC PEACE VIOLATION</th>
      <th>RITUALISM</th>
      <th>ROBBERY</th>
      <th>SEX OFFENSE</th>
      <th>STALKING</th>
      <th>THEFT</th>
      <th>WEAPONS VIOLATION</th>
      <th>YearMonth</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2001-01-31</th>
      <td>24.6</td>
      <td>67.0</td>
      <td>2123.0</td>
      <td>6525.0</td>
      <td>1934.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>3966.0</td>
      <td>3.0</td>
      <td>1191.0</td>
      <td>...</td>
      <td>563.0</td>
      <td>0.0</td>
      <td>161.0</td>
      <td>2.0</td>
      <td>1396.0</td>
      <td>218.0</td>
      <td>26.0</td>
      <td>7865.0</td>
      <td>337.0</td>
      <td>2001-01-01</td>
    </tr>
    <tr>
      <th>2001-02-28</th>
      <td>28.6</td>
      <td>57.0</td>
      <td>2029.0</td>
      <td>6041.0</td>
      <td>1666.0</td>
      <td>0.0</td>
      <td>134.0</td>
      <td>3664.0</td>
      <td>1.0</td>
      <td>1063.0</td>
      <td>...</td>
      <td>426.0</td>
      <td>1.0</td>
      <td>173.0</td>
      <td>0.0</td>
      <td>1159.0</td>
      <td>149.0</td>
      <td>13.0</td>
      <td>6669.0</td>
      <td>301.0</td>
      <td>2001-02-01</td>
    </tr>
    <tr>
      <th>2001-03-31</th>
      <td>34.3</td>
      <td>93.0</td>
      <td>2824.0</td>
      <td>7659.0</td>
      <td>1832.0</td>
      <td>0.0</td>
      <td>149.0</td>
      <td>4617.0</td>
      <td>5.0</td>
      <td>1141.0</td>
      <td>...</td>
      <td>550.0</td>
      <td>0.0</td>
      <td>267.0</td>
      <td>2.0</td>
      <td>1399.0</td>
      <td>184.0</td>
      <td>17.0</td>
      <td>7765.0</td>
      <td>344.0</td>
      <td>2001-03-01</td>
    </tr>
    <tr>
      <th>2001-04-30</th>
      <td>51.6</td>
      <td>89.0</td>
      <td>2747.0</td>
      <td>8326.0</td>
      <td>1931.0</td>
      <td>0.0</td>
      <td>132.0</td>
      <td>4922.0</td>
      <td>2.0</td>
      <td>1133.0</td>
      <td>...</td>
      <td>564.0</td>
      <td>1.0</td>
      <td>229.0</td>
      <td>1.0</td>
      <td>1341.0</td>
      <td>169.0</td>
      <td>29.0</td>
      <td>7702.0</td>
      <td>321.0</td>
      <td>2001-04-01</td>
    </tr>
    <tr>
      <th>2001-05-31</th>
      <td>59.5</td>
      <td>94.0</td>
      <td>2903.0</td>
      <td>8888.0</td>
      <td>1997.0</td>
      <td>0.0</td>
      <td>155.0</td>
      <td>4756.0</td>
      <td>2.0</td>
      <td>1067.0</td>
      <td>...</td>
      <td>503.0</td>
      <td>1.0</td>
      <td>239.0</td>
      <td>1.0</td>
      <td>1491.0</td>
      <td>226.0</td>
      <td>11.0</td>
      <td>8419.0</td>
      <td>390.0</td>
      <td>2001-05-01</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>



Now that all dataframes have a matching 'YearMonth' column, we can join them together by this column

**ONLY RUN THIS CELL ONCE**


```python

crime_months = crime_months.merge(cpi[['YearMonth', 'CPI']], on='YearMonth', how='left')


electricity_filtered = electricity[electricity['sectorName'] == 'all sectors']
crime_months = crime_months.merge(electricity_filtered[['YearMonth', 'price']], on='YearMonth', how='left')  # Use merge instead of join for consistency
crime_months = crime_months.dropna()  # Remove any rows with missing values




```


```python
#run this cell to see what our end result is
crime_months.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AvgTemperature</th>
      <th>ARSON</th>
      <th>ASSAULT</th>
      <th>BATTERY</th>
      <th>BURGLARY</th>
      <th>CONCEALED CARRY LICENSE VIOLATION</th>
      <th>CRIM SEXUAL ASSAULT</th>
      <th>CRIMINAL DAMAGE</th>
      <th>CRIMINAL SEXUAL ASSAULT</th>
      <th>CRIMINAL TRESPASS</th>
      <th>...</th>
      <th>PUBLIC PEACE VIOLATION</th>
      <th>RITUALISM</th>
      <th>ROBBERY</th>
      <th>SEX OFFENSE</th>
      <th>STALKING</th>
      <th>THEFT</th>
      <th>WEAPONS VIOLATION</th>
      <th>YearMonth</th>
      <th>CPI</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24.6</td>
      <td>67.0</td>
      <td>2123.0</td>
      <td>6525.0</td>
      <td>1934.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>3966.0</td>
      <td>3.0</td>
      <td>1191.0</td>
      <td>...</td>
      <td>161.0</td>
      <td>2.0</td>
      <td>1396.0</td>
      <td>218.0</td>
      <td>26.0</td>
      <td>7865.0</td>
      <td>337.0</td>
      <td>2001-01-01</td>
      <td>175.1</td>
      <td>5.87</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28.6</td>
      <td>57.0</td>
      <td>2029.0</td>
      <td>6041.0</td>
      <td>1666.0</td>
      <td>0.0</td>
      <td>134.0</td>
      <td>3664.0</td>
      <td>1.0</td>
      <td>1063.0</td>
      <td>...</td>
      <td>173.0</td>
      <td>0.0</td>
      <td>1159.0</td>
      <td>149.0</td>
      <td>13.0</td>
      <td>6669.0</td>
      <td>301.0</td>
      <td>2001-02-01</td>
      <td>175.8</td>
      <td>6.17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34.3</td>
      <td>93.0</td>
      <td>2824.0</td>
      <td>7659.0</td>
      <td>1832.0</td>
      <td>0.0</td>
      <td>149.0</td>
      <td>4617.0</td>
      <td>5.0</td>
      <td>1141.0</td>
      <td>...</td>
      <td>267.0</td>
      <td>2.0</td>
      <td>1399.0</td>
      <td>184.0</td>
      <td>17.0</td>
      <td>7765.0</td>
      <td>344.0</td>
      <td>2001-03-01</td>
      <td>176.2</td>
      <td>6.47</td>
    </tr>
    <tr>
      <th>3</th>
      <td>51.6</td>
      <td>89.0</td>
      <td>2747.0</td>
      <td>8326.0</td>
      <td>1931.0</td>
      <td>0.0</td>
      <td>132.0</td>
      <td>4922.0</td>
      <td>2.0</td>
      <td>1133.0</td>
      <td>...</td>
      <td>229.0</td>
      <td>1.0</td>
      <td>1341.0</td>
      <td>169.0</td>
      <td>29.0</td>
      <td>7702.0</td>
      <td>321.0</td>
      <td>2001-04-01</td>
      <td>176.9</td>
      <td>6.61</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59.5</td>
      <td>94.0</td>
      <td>2903.0</td>
      <td>8888.0</td>
      <td>1997.0</td>
      <td>0.0</td>
      <td>155.0</td>
      <td>4756.0</td>
      <td>2.0</td>
      <td>1067.0</td>
      <td>...</td>
      <td>239.0</td>
      <td>1.0</td>
      <td>1491.0</td>
      <td>226.0</td>
      <td>11.0</td>
      <td>8419.0</td>
      <td>390.0</td>
      <td>2001-05-01</td>
      <td>177.7</td>
      <td>6.88</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>



Great! We have created a clean dataset full of potential features and labels. 

If we wanted to, we could use any of combination of the columns in this dataframe to predict the values in any other column.
For simplicity of this project, we will just focus on violent crime specifically. The crimes we will consider to be 'violent' are assault, battery, homicide, and domestic violence. And we will select Temperature, CPI, and electricity price as our features.


```python
violent_crimes = crime_months[['AvgTemperature', 'YearMonth', 'CPI', 'price']].copy()
violent_crimes['VIOLENT_CRIMES'] = crime_months[['ASSAULT', 'BATTERY', 'HOMICIDE', 'DOMESTIC VIOLENCE']].sum(axis=1)
violent_crimes = violent_crimes.dropna()  # Optionally drop any rows with missing values

violent_crimes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AvgTemperature</th>
      <th>YearMonth</th>
      <th>CPI</th>
      <th>price</th>
      <th>VIOLENT_CRIMES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24.6</td>
      <td>2001-01-01</td>
      <td>175.1</td>
      <td>5.87</td>
      <td>8691.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28.6</td>
      <td>2001-02-01</td>
      <td>175.8</td>
      <td>6.17</td>
      <td>8097.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34.3</td>
      <td>2001-03-01</td>
      <td>176.2</td>
      <td>6.47</td>
      <td>10520.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>51.6</td>
      <td>2001-04-01</td>
      <td>176.9</td>
      <td>6.61</td>
      <td>11132.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59.5</td>
      <td>2001-05-01</td>
      <td>177.7</td>
      <td>6.88</td>
      <td>11833.0</td>
    </tr>
  </tbody>
</table>
</div>



Now lets have some fun!

Lets look at the relationship between each the label and each feature individually. For this, we will use Linear Regression from sklearn. https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LinearRegression.html


```python



plt.figure(figsize=(15, 5))




#------------------------------------------------
# Scatterplot for AvgTemperature vs VIOLENT_CRIMES
plt.subplot(1, 3, 1)
plt.scatter(violent_crimes['AvgTemperature'], violent_crimes['VIOLENT_CRIMES'], alpha=0.5)

model_temp = LinearRegression()
model_temp.fit(violent_crimes[['AvgTemperature']], violent_crimes['VIOLENT_CRIMES'])
plt.plot(violent_crimes['AvgTemperature'], model_temp.predict(violent_crimes[['AvgTemperature']]), color='red')
plt.title('AvgTemperature vs VIOLENT_CRIMES')
plt.xlabel('AvgTemperature')
plt.ylabel('VIOLENT_CRIMES')
#------------------------------------------------

#------------------------------------------------
# Scatterplot for CPI vs VIOLENT_CRIMES
plt.subplot(1, 3, 2)
plt.scatter(violent_crimes['CPI'], violent_crimes['VIOLENT_CRIMES'], alpha=0.5, color='orange')

model_cpi = LinearRegression()
model_cpi.fit(violent_crimes[['CPI']], violent_crimes['VIOLENT_CRIMES'])
plt.plot(violent_crimes['CPI'], model_cpi.predict(violent_crimes[['CPI']]), color='red')
plt.title('CPI vs VIOLENT_CRIMES')
plt.xlabel('CPI')
plt.ylabel('VIOLENT_CRIMES')
#------------------------------------------------

#------------------------------------------------
# Scatterplot for price vs VIOLENT_CRIMES
plt.subplot(1, 3, 3)
plt.scatter(violent_crimes['price'], violent_crimes['VIOLENT_CRIMES'], alpha=0.5, color='green')

model_price = LinearRegression()
model_price.fit(violent_crimes[['price']], violent_crimes['VIOLENT_CRIMES'])
plt.plot(violent_crimes['price'], model_price.predict(violent_crimes[['price']]), color='red')
plt.title('Price vs VIOLENT_CRIMES')
plt.xlabel('Price')
plt.ylabel('VIOLENT_CRIMES')
#------------------------------------------------


plt.tight_layout()
plt.show()



```


    
![png](FinalProject_files/FinalProject_47_0.png)
    



```python
temp_predictions = model_temp.predict(violent_crimes[['AvgTemperature']])
cpi_predictions = model_cpi.predict(violent_crimes[['CPI']])
price_predictions = model_price.predict(violent_crimes[['price']])

# Calculate accuracy metrics
temp_mse = mean_squared_error(violent_crimes['VIOLENT_CRIMES'], temp_predictions)
temp_r2 = r2_score(violent_crimes['VIOLENT_CRIMES'], temp_predictions)

cpi_mse = mean_squared_error(violent_crimes['VIOLENT_CRIMES'], cpi_predictions)
cpi_r2 = r2_score(violent_crimes['VIOLENT_CRIMES'], cpi_predictions)

price_mse = mean_squared_error(violent_crimes['VIOLENT_CRIMES'], price_predictions)
price_r2 = r2_score(violent_crimes['VIOLENT_CRIMES'], price_predictions)

print(f"AvgTemperature Model - MSE: {temp_mse:.2f}, R^2: {temp_r2:.2f}\n")
print(f"CPI Model - MSE: {cpi_mse:.2f}, R^2: {cpi_r2:.2f}\n")
print(f"Price Model - MSE: {price_mse:.2f}, R^2: {price_r2:.2f}\n")
```

    AvgTemperature Model - MSE: 3601789.81, R^2: 0.19
    
    CPI Model - MSE: 1460678.55, R^2: 0.67
    
    Price Model - MSE: 2722418.33, R^2: 0.38
    
    

Interesting. It looks like CPI is the best predictor for the frequency of violent crime.

More info on MSE: https://en.wikipedia.org/wiki/Mean_squared_error

More info on R^2: https://en.wikipedia.org/wiki/Coefficient_of_determination

Lets see if considering all the features at once will make our predictions any more accurate. 
This looks like a job for KNN!

### Part 1B: K Nearest Neighbors



```python


# Define features and label
X = violent_crimes[['AvgTemperature', 'CPI', 'price']]
y = violent_crimes['VIOLENT_CRIMES']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


# Calculate average error
average_error = abs(y_pred - y_test).mean()
print(f"Average Error: {average_error:.2f}")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)
print(f"{mse=}")
print(f"{r2=}")


```

    Average Error: 809.38
    mse=1183105.0
    r2=0.7479375511387085
    

Looks like considering all features using KNN gave us the most accurate predictions!


A drawback I forgot to mention before conducting these tests is that CPI and electricity prices have continually increased (due to inflation) and crime rates have continually decreased. This shift in crime rates over time is larger than the change in crime rates throughout the year, which inidicates to the ML model that the correlation between inflation and crime rates is stronger than the correlation between the seasonal changes in cost of living and crime rates.

This is why it is important to be aware of the potential existence of factors you have not considered to prevent us from making inaccurate conclusions.


```python
# Plot the number of violent crimes over time
plt.figure(figsize=(10, 6))
plt.plot(violent_crimes['YearMonth'], violent_crimes['VIOLENT_CRIMES'])
plt.title('Number of Violent Crimes Over Time')
plt.xlabel('YearMonth')
plt.ylabel('Number of Violent Crimes')
plt.grid(True)
plt.show()
```


    
![png](FinalProject_files/FinalProject_54_0.png)
    


### Part 2: Decision Trees, Random Forest, Logistic Regression

Now, lets take a look at classification. 
Instead of predicting numerical values, we will predict what category to put things in.


**Next Objective**: we will predict whether or not an arrest was made given the type of crime, the month, and whether or not it was a domestic crime. We will start by filtering out a few columns.


```python

df_filtered = df[['Date', 'Primary Type', 'Location Description', 'Arrest', 'Domestic', 'Month']]
df_filtered['Arrest'] = df_filtered['Arrest'].fillna(False)
df_filtered['Domestic'] = df_filtered['Domestic'].fillna(False)
df_filtered['Arrest'] = df_filtered['Arrest'].astype(int)
df_filtered['Domestic'] = df_filtered['Domestic'].astype(int)

df_filtered.head()


```

    C:\Users\evan3\AppData\Local\Temp\ipykernel_15940\169052729.py:2: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      df_filtered['Arrest'] = df_filtered['Arrest'].fillna(False)
    C:\Users\evan3\AppData\Local\Temp\ipykernel_15940\169052729.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_filtered['Arrest'] = df_filtered['Arrest'].fillna(False)
    C:\Users\evan3\AppData\Local\Temp\ipykernel_15940\169052729.py:3: FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version. Call result.infer_objects(copy=False) instead. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      df_filtered['Domestic'] = df_filtered['Domestic'].fillna(False)
    C:\Users\evan3\AppData\Local\Temp\ipykernel_15940\169052729.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_filtered['Domestic'] = df_filtered['Domestic'].fillna(False)
    C:\Users\evan3\AppData\Local\Temp\ipykernel_15940\169052729.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_filtered['Arrest'] = df_filtered['Arrest'].astype(int)
    C:\Users\evan3\AppData\Local\Temp\ipykernel_15940\169052729.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_filtered['Domestic'] = df_filtered['Domestic'].astype(int)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Primary Type</th>
      <th>Location Description</th>
      <th>Arrest</th>
      <th>Domestic</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2007-08-25 09:22:18</td>
      <td>ASSAULT</td>
      <td>OTHER</td>
      <td>0</td>
      <td>0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-05-24 15:06:00</td>
      <td>HOMICIDE</td>
      <td>STREET</td>
      <td>1</td>
      <td>0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-06-26 09:24:00</td>
      <td>HOMICIDE</td>
      <td>PARKING LOT</td>
      <td>1</td>
      <td>0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-11-09 07:30:00</td>
      <td>BURGLARY</td>
      <td>APARTMENT</td>
      <td>0</td>
      <td>0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-11-12 07:59:00</td>
      <td>BATTERY</td>
      <td>SMALL RETAIL STORE</td>
      <td>1</td>
      <td>0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>



Look at Primary Type and Location Description. These are non-numerical values. We want to make sure there aren't **too** many categories, because it could really slow down the training speed of our models. Lets see how many categories we have to deal with:



```python
pd.set_option('display.max_rows', None)
print(df_filtered['Location Description'].value_counts())
pd.reset_option('display.max_rows')
```

    Location Description
    STREET                                                   1923555
    RESIDENCE                                                1243468
    APARTMENT                                                 799520
    SIDEWALK                                                  707612
    OTHER                                                     269998
    PARKING LOT/GARAGE(NON.RESID.)                            202960
    ALLEY                                                     164396
    SCHOOL, PUBLIC, BUILDING                                  146365
    SMALL RETAIL STORE                                        137372
    RESIDENCE-GARAGE                                          135521
    RESIDENCE PORCH/HALLWAY                                   124184
    RESTAURANT                                                120703
    VEHICLE NON-COMMERCIAL                                    118938
    GROCERY FOOD STORE                                         94992
    DEPARTMENT STORE                                           94696
    GAS STATION                                                80936
    RESIDENTIAL YARD (FRONT/BACK)                              75146
    CHA PARKING LOT/GROUNDS                                    56101
    PARK PROPERTY                                              55835
    COMMERCIAL / BUSINESS OFFICE                               55016
    BAR OR TAVERN                                              40633
    CTA PLATFORM                                               38581
    CHA APARTMENT                                              37838
    DRUG STORE                                                 33790
    SCHOOL, PUBLIC, GROUNDS                                    30250
    BANK                                                       30028
    HOTEL/MOTEL                                                29738
    CTA TRAIN                                                  28490
    CHA HALLWAY/STAIRWELL/ELEVATOR                             25021
    VACANT LOT/LAND                                            24683
    CTA BUS                                                    24001
    TAVERN/LIQUOR STORE                                        22460
    HOSPITAL BUILDING/GROUNDS                                  22191
    DRIVEWAY - RESIDENTIAL                                     21358
    CONVENIENCE STORE                                          21224
    POLICE FACILITY/VEH PARKING LOT                            18565
    AIRPORT/AIRCRAFT                                           16210
    CHURCH/SYNAGOGUE/PLACE OF WORSHIP                          15491
    GOVERNMENT BUILDING/PROPERTY                               14757
    NURSING HOME/RETIREMENT HOME                               14656
    SCHOOL, PRIVATE, BUILDING                                  14181
    CONSTRUCTION SITE                                          13597
    ABANDONED BUILDING                                         11661
    CURRENCY EXCHANGE                                          11573
    CTA GARAGE / OTHER PROPERTY                                10277
    WAREHOUSE                                                   9962
    PARKING LOT / GARAGE (NON RESIDENTIAL)                      9763
    ATHLETIC CLUB                                               9011
    ATM (AUTOMATIC TELLER MACHINE)                              8250
    BARBERSHOP                                                  8192
    TAXICAB                                                     7631
    MEDICAL/DENTAL OFFICE                                       7430
    CTA BUS STOP                                                7349
    OTHER (SPECIFY)                                             6980
    FACTORY/MANUFACTURING BUILDING                              6894
    LIBRARY                                                     6493
    CTA STATION                                                 6066
    OTHER RAILROAD PROP / TRAIN DEPOT                           5932
    COLLEGE/UNIVERSITY GROUNDS                                  5786
    VEHICLE-COMMERCIAL                                          5615
    SPORTS ARENA/STADIUM                                        5290
    CLEANING STORE                                              4941
    RESIDENCE - PORCH / HALLWAY                                 4650
    AIRPORT TERMINAL UPPER LEVEL - SECURE AREA                  4410
    SCHOOL, PRIVATE, GROUNDS                                    4292
    RESIDENCE - YARD (FRONT / BACK)                             4231
    RESIDENCE - GARAGE                                          3543
    OTHER COMMERCIAL TRANSPORTATION                             3190
    DAY CARE CENTER                                             3162
    CAR WASH                                                    3062
    MOVIE HOUSE/THEATER                                         2732
    APPLIANCE STORE                                             2118
    SCHOOL - PUBLIC BUILDING                                    1979
    AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA              1949
    HOSPITAL BUILDING / GROUNDS                                 1721
    SCHOOL - PUBLIC GROUNDS                                     1645
    HOTEL / MOTEL                                               1605
    COLLEGE/UNIVERSITY RESIDENCE HALL                           1400
    AUTO                                                        1375
    LAKEFRONT/WATERFRONT/RIVERBANK                              1181
    JAIL / LOCK-UP FACILITY                                     1168
    POLICE FACILITY / VEHICLE PARKING LOT                       1134
    FIRE STATION                                                1124
    AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA             1108
    COIN OPERATED MACHINE                                       1097
    HIGHWAY/EXPRESSWAY                                          1085
    AIRPORT PARKING LOT                                         1018
    NURSING / RETIREMENT HOME                                    968
    POOL ROOM                                                    962
    DELIVERY TRUCK                                               962
    AIRPORT EXTERIOR - NON-SECURE AREA                           945
    AIRPORT VENDING ESTABLISHMENT                                934
    VACANT LOT / LAND                                            928
    FEDERAL BUILDING                                             887
    AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA               873
    ANIMAL HOSPITAL                                              851
    AUTO / BOAT / RV DEALERSHIP                                  807
    AIRPORT TERMINAL LOWER LEVEL - SECURE AREA                   777
    AIRCRAFT                                                     769
    GOVERNMENT BUILDING / PROPERTY                               744
    BOWLING ALLEY                                                733
    AIRPORT BUILDING NON-TERMINAL - SECURE AREA                  705
    BOAT/WATERCRAFT                                              698
    TAVERN / LIQUOR STORE                                        675
    HOUSE                                                        674
    PAWN SHOP                                                    655
    CHURCH / SYNAGOGUE / PLACE OF WORSHIP                        575
    CHA PARKING LOT / GROUNDS                                    562
    CREDIT UNION                                                 542
    VEHICLE - OTHER RIDE SHARE SERVICE (E.G., UBER, LYFT)        467
    BRIDGE                                                       426
    FOREST PRESERVE                                              413
    MEDICAL / DENTAL OFFICE                                      401
    PORCH                                                        398
    SAVINGS AND LOAN                                             385
    CEMETARY                                                     385
    AIRPORT EXTERIOR - SECURE AREA                               370
    VEHICLE - OTHER RIDE SERVICE                                 331
    SCHOOL - PRIVATE GROUNDS                                     320
    SCHOOL - PRIVATE BUILDING                                    313
    YARD                                                         311
    VEHICLE - COMMERCIAL                                         304
    PARKING LOT                                                  263
    VEHICLE - DELIVERY TRUCK                                     242
    NEWSSTAND                                                    239
    CHA HALLWAY / STAIRWELL / ELEVATOR                           216
    CTA PARKING LOT / GARAGE / OTHER PROPERTY                    204
    OTHER RAILROAD PROPERTY / TRAIN DEPOT                        204
    SPORTS ARENA / STADIUM                                       178
    VEHICLE - OTHER RIDE SHARE SERVICE (LYFT, UBER, ETC.)        177
    FACTORY / MANUFACTURING BUILDING                             171
    CTA TRACKS - RIGHT OF WAY                                    169
    VACANT LOT                                                   136
    AIRPORT TRANSPORTATION SYSTEM (ATS)                          111
    HALLWAY                                                      106
    AIRPORT TERMINAL MEZZANINE - NON-SECURE AREA                 102
    RETAIL STORE                                                 101
    COLLEGE / UNIVERSITY - GROUNDS                               100
    MOVIE HOUSE / THEATER                                         93
    GANGWAY                                                       74
    LAKEFRONT / WATERFRONT / RIVERBANK                            74
    GARAGE                                                        73
    GAS STATION DRIVE/PROP.                                       71
    CHA PARKING LOT                                               58
    HIGHWAY / EXPRESSWAY                                          54
    CHA GROUNDS                                                   47
    CHA HALLWAY                                                   38
    TAVERN                                                        37
    BASEMENT                                                      32
    COLLEGE / UNIVERSITY - RESIDENCE HALL                         29
    VESTIBULE                                                     28
    BOAT / WATERCRAFT                                             28
    HOTEL                                                         26
    DRIVEWAY                                                      26
    BARBER SHOP/BEAUTY SALON                                      26
    STAIRWELL                                                     24
    OFFICE                                                        20
    CLUB                                                          18
    SCHOOL YARD                                                   17
    RAILROAD PROPERTY                                             15
    HOSPITAL                                                      14
    LIQUOR STORE                                                  13
    GARAGE/AUTO REPAIR                                            11
    VEHICLE-COMMERCIAL - TROLLEY BUS                              10
    KENNEL                                                        10
    VEHICLE-COMMERCIAL - ENTERTAINMENT/PARTY BUS                  10
    CTA "L" TRAIN                                                 10
    TRUCK                                                          9
    CHA LOBBY                                                      9
    CHA STAIRWELL                                                  9
    CTA "L" PLATFORM                                               9
    CTA PROPERTY                                                   9
    WOODED AREA                                                    7
    MOTEL                                                          7
    DUMPSTER                                                       7
    NURSING HOME                                                   6
    CHURCH                                                         6
    TAXI CAB                                                       6
    VEHICLE - COMMERCIAL: ENTERTAINMENT / PARTY BUS                5
    VEHICLE - COMMERCIAL: TROLLEY BUS                              5
    RIVER BANK                                                     5
    FARM                                                           5
    LAKE                                                           4
    TRAILER                                                        4
    CHA PLAY LOT                                                   4
    RIVER                                                          4
    CHA BREEZEWAY                                                  3
    HORSE STABLE                                                   3
    CHA ELEVATOR                                                   3
    COACH HOUSE                                                    3
    SEWER                                                          3
    YMCA                                                           3
    PRAIRIE                                                        2
    PUBLIC HIGH SCHOOL                                             2
    LAUNDRY ROOM                                                   2
    PUBLIC GRAMMAR SCHOOL                                          2
    CHURCH PROPERTY                                                2
    BANQUET HALL                                                   2
    ELEVATOR                                                       2
    FACTORY                                                        2
    COUNTY JAIL                                                    2
    LIVERY STAND OFFICE                                            2
    ROOMING HOUSE                                                  2
    GOVERNMENT BUILDING                                            2
    CTA SUBWAY STATION                                             2
    EXPRESSWAY EMBANKMENT                                          1
    BEACH                                                          1
    LOADING DOCK                                                   1
    TRUCKING TERMINAL                                              1
    LIVERY AUTO                                                    1
    FUNERAL PARLOR                                                 1
    POLICE FACILITY                                                1
    CLEANERS/LAUNDROMAT                                            1
    POOLROOM                                                       1
    ROOF                                                           1
    LAGOON                                                         1
    JUNK YARD/GARBAGE DUMP                                         1
    Name: count, dtype: int64
    


```python
pd.set_option('display.max_rows', None)
print(df_filtered['Location Description'].value_counts())

pd.reset_option('display.max_rows')
```

    Location Description
    STREET                                                   1923555
    RESIDENCE                                                1243468
    APARTMENT                                                 799520
    SIDEWALK                                                  707612
    OTHER                                                     269998
    PARKING LOT/GARAGE(NON.RESID.)                            202960
    ALLEY                                                     164396
    SCHOOL, PUBLIC, BUILDING                                  146365
    SMALL RETAIL STORE                                        137372
    RESIDENCE-GARAGE                                          135521
    RESIDENCE PORCH/HALLWAY                                   124184
    RESTAURANT                                                120703
    VEHICLE NON-COMMERCIAL                                    118938
    GROCERY FOOD STORE                                         94992
    DEPARTMENT STORE                                           94696
    GAS STATION                                                80936
    RESIDENTIAL YARD (FRONT/BACK)                              75146
    CHA PARKING LOT/GROUNDS                                    56101
    PARK PROPERTY                                              55835
    COMMERCIAL / BUSINESS OFFICE                               55016
    BAR OR TAVERN                                              40633
    CTA PLATFORM                                               38581
    CHA APARTMENT                                              37838
    DRUG STORE                                                 33790
    SCHOOL, PUBLIC, GROUNDS                                    30250
    BANK                                                       30028
    HOTEL/MOTEL                                                29738
    CTA TRAIN                                                  28490
    CHA HALLWAY/STAIRWELL/ELEVATOR                             25021
    VACANT LOT/LAND                                            24683
    CTA BUS                                                    24001
    TAVERN/LIQUOR STORE                                        22460
    HOSPITAL BUILDING/GROUNDS                                  22191
    DRIVEWAY - RESIDENTIAL                                     21358
    CONVENIENCE STORE                                          21224
    POLICE FACILITY/VEH PARKING LOT                            18565
    AIRPORT/AIRCRAFT                                           16210
    CHURCH/SYNAGOGUE/PLACE OF WORSHIP                          15491
    GOVERNMENT BUILDING/PROPERTY                               14757
    NURSING HOME/RETIREMENT HOME                               14656
    SCHOOL, PRIVATE, BUILDING                                  14181
    CONSTRUCTION SITE                                          13597
    ABANDONED BUILDING                                         11661
    CURRENCY EXCHANGE                                          11573
    CTA GARAGE / OTHER PROPERTY                                10277
    WAREHOUSE                                                   9962
    PARKING LOT / GARAGE (NON RESIDENTIAL)                      9763
    ATHLETIC CLUB                                               9011
    ATM (AUTOMATIC TELLER MACHINE)                              8250
    BARBERSHOP                                                  8192
    TAXICAB                                                     7631
    MEDICAL/DENTAL OFFICE                                       7430
    CTA BUS STOP                                                7349
    OTHER (SPECIFY)                                             6980
    FACTORY/MANUFACTURING BUILDING                              6894
    LIBRARY                                                     6493
    CTA STATION                                                 6066
    OTHER RAILROAD PROP / TRAIN DEPOT                           5932
    COLLEGE/UNIVERSITY GROUNDS                                  5786
    VEHICLE-COMMERCIAL                                          5615
    SPORTS ARENA/STADIUM                                        5290
    CLEANING STORE                                              4941
    RESIDENCE - PORCH / HALLWAY                                 4650
    AIRPORT TERMINAL UPPER LEVEL - SECURE AREA                  4410
    SCHOOL, PRIVATE, GROUNDS                                    4292
    RESIDENCE - YARD (FRONT / BACK)                             4231
    RESIDENCE - GARAGE                                          3543
    OTHER COMMERCIAL TRANSPORTATION                             3190
    DAY CARE CENTER                                             3162
    CAR WASH                                                    3062
    MOVIE HOUSE/THEATER                                         2732
    APPLIANCE STORE                                             2118
    SCHOOL - PUBLIC BUILDING                                    1979
    AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA              1949
    HOSPITAL BUILDING / GROUNDS                                 1721
    SCHOOL - PUBLIC GROUNDS                                     1645
    HOTEL / MOTEL                                               1605
    COLLEGE/UNIVERSITY RESIDENCE HALL                           1400
    AUTO                                                        1375
    LAKEFRONT/WATERFRONT/RIVERBANK                              1181
    JAIL / LOCK-UP FACILITY                                     1168
    POLICE FACILITY / VEHICLE PARKING LOT                       1134
    FIRE STATION                                                1124
    AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA             1108
    COIN OPERATED MACHINE                                       1097
    HIGHWAY/EXPRESSWAY                                          1085
    AIRPORT PARKING LOT                                         1018
    NURSING / RETIREMENT HOME                                    968
    POOL ROOM                                                    962
    DELIVERY TRUCK                                               962
    AIRPORT EXTERIOR - NON-SECURE AREA                           945
    AIRPORT VENDING ESTABLISHMENT                                934
    VACANT LOT / LAND                                            928
    FEDERAL BUILDING                                             887
    AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA               873
    ANIMAL HOSPITAL                                              851
    AUTO / BOAT / RV DEALERSHIP                                  807
    AIRPORT TERMINAL LOWER LEVEL - SECURE AREA                   777
    AIRCRAFT                                                     769
    GOVERNMENT BUILDING / PROPERTY                               744
    BOWLING ALLEY                                                733
    AIRPORT BUILDING NON-TERMINAL - SECURE AREA                  705
    BOAT/WATERCRAFT                                              698
    TAVERN / LIQUOR STORE                                        675
    HOUSE                                                        674
    PAWN SHOP                                                    655
    CHURCH / SYNAGOGUE / PLACE OF WORSHIP                        575
    CHA PARKING LOT / GROUNDS                                    562
    CREDIT UNION                                                 542
    VEHICLE - OTHER RIDE SHARE SERVICE (E.G., UBER, LYFT)        467
    BRIDGE                                                       426
    FOREST PRESERVE                                              413
    MEDICAL / DENTAL OFFICE                                      401
    PORCH                                                        398
    SAVINGS AND LOAN                                             385
    CEMETARY                                                     385
    AIRPORT EXTERIOR - SECURE AREA                               370
    VEHICLE - OTHER RIDE SERVICE                                 331
    SCHOOL - PRIVATE GROUNDS                                     320
    SCHOOL - PRIVATE BUILDING                                    313
    YARD                                                         311
    VEHICLE - COMMERCIAL                                         304
    PARKING LOT                                                  263
    VEHICLE - DELIVERY TRUCK                                     242
    NEWSSTAND                                                    239
    CHA HALLWAY / STAIRWELL / ELEVATOR                           216
    CTA PARKING LOT / GARAGE / OTHER PROPERTY                    204
    OTHER RAILROAD PROPERTY / TRAIN DEPOT                        204
    SPORTS ARENA / STADIUM                                       178
    VEHICLE - OTHER RIDE SHARE SERVICE (LYFT, UBER, ETC.)        177
    FACTORY / MANUFACTURING BUILDING                             171
    CTA TRACKS - RIGHT OF WAY                                    169
    VACANT LOT                                                   136
    AIRPORT TRANSPORTATION SYSTEM (ATS)                          111
    HALLWAY                                                      106
    AIRPORT TERMINAL MEZZANINE - NON-SECURE AREA                 102
    RETAIL STORE                                                 101
    COLLEGE / UNIVERSITY - GROUNDS                               100
    MOVIE HOUSE / THEATER                                         93
    GANGWAY                                                       74
    LAKEFRONT / WATERFRONT / RIVERBANK                            74
    GARAGE                                                        73
    GAS STATION DRIVE/PROP.                                       71
    CHA PARKING LOT                                               58
    HIGHWAY / EXPRESSWAY                                          54
    CHA GROUNDS                                                   47
    CHA HALLWAY                                                   38
    TAVERN                                                        37
    BASEMENT                                                      32
    COLLEGE / UNIVERSITY - RESIDENCE HALL                         29
    VESTIBULE                                                     28
    BOAT / WATERCRAFT                                             28
    HOTEL                                                         26
    DRIVEWAY                                                      26
    BARBER SHOP/BEAUTY SALON                                      26
    STAIRWELL                                                     24
    OFFICE                                                        20
    CLUB                                                          18
    SCHOOL YARD                                                   17
    RAILROAD PROPERTY                                             15
    HOSPITAL                                                      14
    LIQUOR STORE                                                  13
    GARAGE/AUTO REPAIR                                            11
    VEHICLE-COMMERCIAL - TROLLEY BUS                              10
    KENNEL                                                        10
    VEHICLE-COMMERCIAL - ENTERTAINMENT/PARTY BUS                  10
    CTA "L" TRAIN                                                 10
    TRUCK                                                          9
    CHA LOBBY                                                      9
    CHA STAIRWELL                                                  9
    CTA "L" PLATFORM                                               9
    CTA PROPERTY                                                   9
    WOODED AREA                                                    7
    MOTEL                                                          7
    DUMPSTER                                                       7
    NURSING HOME                                                   6
    CHURCH                                                         6
    TAXI CAB                                                       6
    VEHICLE - COMMERCIAL: ENTERTAINMENT / PARTY BUS                5
    VEHICLE - COMMERCIAL: TROLLEY BUS                              5
    RIVER BANK                                                     5
    FARM                                                           5
    LAKE                                                           4
    TRAILER                                                        4
    CHA PLAY LOT                                                   4
    RIVER                                                          4
    CHA BREEZEWAY                                                  3
    HORSE STABLE                                                   3
    CHA ELEVATOR                                                   3
    COACH HOUSE                                                    3
    SEWER                                                          3
    YMCA                                                           3
    PRAIRIE                                                        2
    PUBLIC HIGH SCHOOL                                             2
    LAUNDRY ROOM                                                   2
    PUBLIC GRAMMAR SCHOOL                                          2
    CHURCH PROPERTY                                                2
    BANQUET HALL                                                   2
    ELEVATOR                                                       2
    FACTORY                                                        2
    COUNTY JAIL                                                    2
    LIVERY STAND OFFICE                                            2
    ROOMING HOUSE                                                  2
    GOVERNMENT BUILDING                                            2
    CTA SUBWAY STATION                                             2
    EXPRESSWAY EMBANKMENT                                          1
    BEACH                                                          1
    LOADING DOCK                                                   1
    TRUCKING TERMINAL                                              1
    LIVERY AUTO                                                    1
    FUNERAL PARLOR                                                 1
    POLICE FACILITY                                                1
    CLEANERS/LAUNDROMAT                                            1
    POOLROOM                                                       1
    ROOF                                                           1
    LAGOON                                                         1
    JUNK YARD/GARBAGE DUMP                                         1
    Name: count, dtype: int64
    

We have so many different categories!!!!

This sucks because we have to somewhat manually decide which categories we want and how to reorganize everything. It also looks like there are some categories that are basically the same but are worded differently, or have trivial specifications. We need to consolidate.


**ONLY RUN THIS ONCE**


```python



# Consolidation map
# everything in the VALUE arrays will map to its corresponding KEY
consolidation_map = {
    'RESIDENCE': [
        'RESIDENCE', 'RESIDENTIAL YARD (FRONT/BACK)', 'RESIDENCE PORCH/HALLWAY', 
        'RESIDENCE-GARAGE', 'DRIVEWAY - RESIDENTIAL', 'RESIDENCE - PORCH / HALLWAY',
        'RESIDENCE - YARD (FRONT / BACK)', 'RESIDENCE - GARAGE', 'CHA APARTMENT'
    ],
    'STORE': [
        'SMALL RETAIL STORE', 'GROCERY FOOD STORE', 'DEPARTMENT STORE',
        'CONVENIENCE STORE', 'DRUG STORE', 'TAVERN/LIQUOR STORE', 'LIQUOR STORE',
        'APPLIANCE STORE', 'CURRENCY EXCHANGE', 'BARBERSHOP', 'CLEANING STORE'
    ],
    'VEHICLE': [
        'VEHICLE NON-COMMERCIAL', 'VEHICLE-COMMERCIAL', 'TAXICAB',
        'DELIVERY TRUCK', 'VEHICLE - OTHER RIDE SHARE SERVICE (E.G., UBER, LYFT)',
        'VEHICLE - DELIVERY TRUCK', 'VEHICLE - COMMERCIAL: ENTERTAINMENT / PARTY BUS',
        'VEHICLE - COMMERCIAL: TROLLEY BUS', 'AUTO', 
        'TRUCK', 'BOAT/WATERCRAFT'
    ],
    'SCHOOL (PUBLIC + PRIVATE, BUILDING/GROUNDS)': [
        'SCHOOL, PUBLIC, BUILDING', 'SCHOOL, PUBLIC, GROUNDS', 
        'SCHOOL, PRIVATE, BUILDING', 'SCHOOL, PRIVATE, GROUNDS',
        'PUBLIC HIGH SCHOOL', 'PUBLIC GRAMMAR SCHOOL', 'SCHOOL - PUBLIC BUILDING',
        'SCHOOL - PUBLIC GROUNDS', 'SCHOOL - PRIVATE BUILDING', 'SCHOOL - PRIVATE GROUNDS'
    ],
    'BAR/TAVERN/LIQUOR STORE': [
        'BAR OR TAVERN', 'TAVERN', 'TAVERN/LIQUOR STORE'
    ],
    'PARKING LOT/GARAGE (NON-RESIDENTIAL)': [
        'PARKING LOT/GARAGE(NON.RESID.)', 'PARKING LOT / GARAGE (NON RESIDENTIAL)',
        'CHA PARKING LOT/GROUNDS', 'CTA PARKING LOT / GARAGE / OTHER PROPERTY'
    ],
    'PUBLIC TRANSPORT': [
        'CTA PLATFORM', 'CTA TRAIN', 'CTA BUS', 'CTA BUS STOP',
        'CTA STATION', 
        'CTA "L" TRAIN', 'CTA "L" PLATFORM', 'CTA PROPERTY',
        'CTA SUBWAY STATION', 'CTA TRACKS - RIGHT OF WAY',
        'AIRPORT/AIRCRAFT', 'AIRPORT TERMINAL UPPER LEVEL - SECURE AREA',
        'AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA',
        'AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA',
        'AIRPORT TERMINAL LOWER LEVEL - SECURE AREA',
        'AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA',
        'AIRPORT BUILDING NON-TERMINAL - SECURE AREA',
        'AIRPORT PARKING LOT', 'AIRPORT EXTERIOR - NON-SECURE AREA',
        'AIRPORT EXTERIOR - SECURE AREA', 'AIRPORT VENDING ESTABLISHMENT',
        'AIRPORT TRANSPORTATION SYSTEM (ATS)'
    ]
}

#flatten the map to so we know which things should map to which consolidation
flattened_map = {loc: key for key, locations in consolidation_map.items() for loc in locations}

#Apply the consolidation
df_filtered['Location Description'] = df_filtered['Location Description'].replace(flattened_map)



#we still have some outlier categories 
#These are the top 16 categories with the most entries so we will keep these and set everything else to OTHER
valid_categories = [
    'STREET', 'RESIDENCE', 'APARTMENT', 'SIDEWALK', 'STORE', 'OTHER',
    'PARKING LOT/GARAGE (NON-RESIDENTIAL)', 'SCHOOL (PUBLIC + PRIVATE, BUILDING/GROUNDS)',
    'ALLEY', 'VEHICLE', 'PUBLIC TRANSPORT', 'RESTAURANT', 'GAS STATION',
    'BAR/TAVERN/LIQUOR STORE', 'PARK PROPERTY', 'COMMERCIAL / BUSINESS OFFICE'
]

# Anything not in the valid_categories will be put into OTHER
df_filtered['Location Description'] = df_filtered['Location Description'].apply(
    lambda x: x if x in valid_categories else 'OTHER'
)

```

    C:\Users\evan3\AppData\Local\Temp\ipykernel_15940\1733506933.py:55: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_filtered['Location Description'] = df_filtered['Location Description'].replace(flattened_map)
    C:\Users\evan3\AppData\Local\Temp\ipykernel_15940\1733506933.py:69: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_filtered['Location Description'] = df_filtered['Location Description'].apply(
    


```python
#run this cell to see the number of categories we have now
df_filtered['Location Description'].value_counts()
```




    Location Description
    STREET                                         1923555
    RESIDENCE                                      1649939
    APARTMENT                                       799520
    SIDEWALK                                        707612
    OTHER                                           623214
    STORE                                           408911
    PARKING LOT/GARAGE (NON-RESIDENTIAL)            269028
    SCHOOL (PUBLIC + PRIVATE, BUILDING/GROUNDS)     199349
    ALLEY                                           164396
    VEHICLE                                         135947
    PUBLIC TRANSPORT                                134096
    RESTAURANT                                      120703
    GAS STATION                                      80936
    BAR/TAVERN/LIQUOR STORE                          63130
    PARK PROPERTY                                    55835
    COMMERCIAL / BUSINESS OFFICE                     55016
    Name: count, dtype: int64




```python
df_filtered.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Primary Type</th>
      <th>Location Description</th>
      <th>Arrest</th>
      <th>Domestic</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2007-08-25 09:22:18</td>
      <td>ASSAULT</td>
      <td>OTHER</td>
      <td>0</td>
      <td>0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-05-24 15:06:00</td>
      <td>HOMICIDE</td>
      <td>STREET</td>
      <td>1</td>
      <td>0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-06-26 09:24:00</td>
      <td>HOMICIDE</td>
      <td>OTHER</td>
      <td>1</td>
      <td>0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-11-09 07:30:00</td>
      <td>BURGLARY</td>
      <td>APARTMENT</td>
      <td>0</td>
      <td>0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-11-12 07:59:00</td>
      <td>BATTERY</td>
      <td>STORE</td>
      <td>1</td>
      <td>0</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
</div>



Nice, we needed to do this because too many categories could make our models take longer to train.

To make this more digestible for our models, we will turn each category into its own column and have a 0 or 1 indicating whether or not each row is part of that category. Now it is more clear why we needed to consolidate so much. Otherwise we would just have way too many features.


**RUN THIS CELL ONCE ONLY**


```python
df_filtered = df_filtered.join(pd.get_dummies(df_filtered['Primary Type'], prefix='', prefix_sep='', dtype=int))
df_filtered = df_filtered.join(pd.get_dummies(df_filtered['Location Description'], prefix='', prefix_sep='', dtype=int))

df_filtered.drop('Primary Type', axis=1, inplace=True)
df_filtered.drop('Location Description', axis = 1, inplace=True)
df_filtered = df_filtered.dropna(subset=['Date'])  # Remove rows with missing dates

```


```python
#run this cell to see our dataset after all pre processing
df_filtered.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Arrest</th>
      <th>Domestic</th>
      <th>Month</th>
      <th>ARSON</th>
      <th>ASSAULT</th>
      <th>BATTERY</th>
      <th>BURGLARY</th>
      <th>CONCEALED CARRY LICENSE VIOLATION</th>
      <th>CRIM SEXUAL ASSAULT</th>
      <th>...</th>
      <th>PARK PROPERTY</th>
      <th>PARKING LOT/GARAGE (NON-RESIDENTIAL)</th>
      <th>PUBLIC TRANSPORT</th>
      <th>RESIDENCE</th>
      <th>RESTAURANT</th>
      <th>SCHOOL (PUBLIC + PRIVATE, BUILDING/GROUNDS)</th>
      <th>SIDEWALK</th>
      <th>STORE</th>
      <th>STREET</th>
      <th>VEHICLE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2007-08-25 09:22:18</td>
      <td>0</td>
      <td>0</td>
      <td>8.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-05-24 15:06:00</td>
      <td>1</td>
      <td>0</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-06-26 09:24:00</td>
      <td>1</td>
      <td>0</td>
      <td>6.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-11-09 07:30:00</td>
      <td>0</td>
      <td>0</td>
      <td>11.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-11-12 07:59:00</td>
      <td>1</td>
      <td>0</td>
      <td>11.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 56 columns</p>
</div>



Now that we have completed pre processing, lets see which ML algorithm will give us the best Arrest=yes/no predictions.
To do this, we will compare the precision, recall and f1 score between a few different ML algorithms. More on these terms later.



```python

X = df_filtered.drop(columns=['Date', 'Arrest'])
y = df_filtered['Arrest']



seed = 42 
test_size = 0.2

# Randomly sample a specific number of rows
max_samples = 100000  # Set the maximum number of samples you want. We will set it to 100k to save time
X_sampled = X.sample(n=max_samples, random_state=seed)  # Randomly sample max_samples rows
y_sampled = y.loc[X_sampled.index]  # Get the corresponding y values


# split the sampled dataset
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=test_size, random_state=seed, shuffle=True)
#scale so that months are between 0 and 1, not sure if this matters or not
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#output is already either 0 or 1, so no need to scale

models = {
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression()

}

for model_name, model in models.items():
    np.random.seed(seed)
    model.fit(X_train_scaled, y_train)



# Evaluate the performance of each model
for model_name, model in models.items():
    np.random.seed(seed)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of {model_name}: {accuracy:.2f}")
    print(classification_report(y_test, y_pred)) 

```

    Accuracy of DecisionTree: 0.86
                  precision    recall  f1-score   support
    
               0       0.87      0.96      0.91     14546
               1       0.85      0.61      0.71      5454
    
        accuracy                           0.86     20000
       macro avg       0.86      0.79      0.81     20000
    weighted avg       0.86      0.86      0.86     20000
    
    Accuracy of RandomForest: 0.86
                  precision    recall  f1-score   support
    
               0       0.87      0.96      0.91     14546
               1       0.84      0.62      0.71      5454
    
        accuracy                           0.86     20000
       macro avg       0.86      0.79      0.81     20000
    weighted avg       0.86      0.86      0.86     20000
    
    Accuracy of LogisticRegression: 0.86
                  precision    recall  f1-score   support
    
               0       0.86      0.97      0.91     14546
               1       0.89      0.56      0.69      5454
    
        accuracy                           0.86     20000
       macro avg       0.87      0.77      0.80     20000
    weighted avg       0.86      0.86      0.85     20000
    
    

It seems like all 3 have relatively similar performance. The precision of all the models is pretty high, but the recall is particularly low. Before we continue there are a few terms we need to know:
* **Class Imbalance**: We have a class imbalance because there are more Arrest = No entries than there are Arrest = Yes entries. 
* **Precision**: Out of the number of times I predicted X, how many times was the actual answer not X?
* **Recall**: Out of the number of times the actual answer was X, how many times did I correctly predict X?
* **F1 Score**: an indicator that accounts for both precision and recall

Since we have such a disproportionate amount of Arrest=No to Arrest=Yes, our recall can become very low. Since our model sees 'No' so much more often than 'Yes', it trains itself to just go with 'No' most of the time since that is more likely to be correct. 

In situations like these, we need to think about what is more important to us: do we care more about just getting the prediction right, or do we care more about predicting a certain outcome correctly?

To show the tradeoff relationship between precision and recall, we will choose to maximize **recall**. In the following cell, we will stick with using a Decision Tree, because it has more parameters to tune, and we will be able to see a cool looking image of our decision tree at the end.



We want to make the BEST decision tree we can, so we will try many combinations of hyperparameters. The explanation for the numbers I chose are a bit much to explain for this project so here is some more information on hyperparameter tuning: https://www.geeksforgeeks.org/how-to-tune-a-decision-tree-in-hyperparameter-tuning/

The bottom line is, different combinations of hyperparameters can yield vastly different decision trees. So we are using GridSearch from sklearn to try all combinations of reasonable hyperparameter values to optimize for recall.



```python


#lets tweak the parameters to see which combination gives us the best accuracy
param_grid = {
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [100, 1000],
    'min_samples_leaf': [100, 1000],
    'class_weight': ['balanced', {0: 1, 1: 3.5},{0: 1, 1: 4.5}]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=seed), param_grid, scoring='recall')
grid_search.fit(X_train_scaled, y_train)

# Use the best estimator
best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}") 
print(classification_report(y_test, y_pred)) 



```

    Accuracy: 0.65
                  precision    recall  f1-score   support
    
               0       0.94      0.55      0.69     14546
               1       0.43      0.91      0.58      5454
    
        accuracy                           0.65     20000
       macro avg       0.69      0.73      0.64     20000
    weighted avg       0.80      0.65      0.66     20000
    
    

By prioritizing recall, we have sacrificed precision and overall accuracy. The reason we see this 'tradeoff' tendency is because when we prioritized recall, we are essentially making our model say 'Yes' more often. And knowing that we have a class imbalance towards 'No', saying 'Yes' more often will generally decrease the chances we get the answer right. 

Most of the time, we want precision and recall to be as close togther and as high as possible. If both precision and recall are high, it is an indicator that our model has detected a meaningful pattern in the data.

Because of our limited training data and time, it could be the case that the features we selected don't have nearly enough correlation for our decision tree to pick up on.


```python
#Heres what our decision tree looks like
plt.figure(figsize=(160,80))
plot_tree(best_model, filled=True, feature_names=X.columns, class_names=['No Arrest', 'Arrest'], rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
```


    
![png](FinalProject_files/FinalProject_73_0.png)
    


### Insights


In this project we have learned some important principles for data scientists:
* The importance of data pre-processing and normalization
* Being aware of the impact of potential un-accounted factors 
* The impact that class imbalance has when training ML models
* Relationships between precision and recall
* The limitations of ML models, specifically, decision trees

We are also now able to predict how many violent crimes will be committed given the median monthly temperature, the CPI, and electricity prices with decent accuracy. 

We tried to predict whether or not an arrest will be made given where the crime was committed and what type of crime it was. The results were not the worst, but were not accurate enough to make firm conclusions.

### Conclusion

In the real world, these conclusions can be used as insight to policy-making to keep people safer. For example, if we found that domestic violence, or crimes committed in office spaces are less likely to lead to arrests, we could raise awareness of these potential issues or take other preventative actions in those areas. 

Understanding data science and its life cycle will allow us to approach real world issues that at a glance, seem too complex to tackle. By leveraging the massive amount of data on the internet, or collecting your own, you could be just one decision tree away from creating a solution that has a real impact on humanity.

