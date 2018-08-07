
# Pyber Ride Sharing

#Steps
1. Extracting Data from csv files
2. Mine the data to gather relevant information
3. Create graphs to visualize the data 
----

### Analysis
* Roughly 70% of all rides and total fare value come from Urban riders.
* Rural riders tend to take significantly higher cost, albeit fewer, trips than Urban or Suburban riders.
* While comprising of roughly 37% of all rides and fares, Suburban and Rural cities afford only 14% of Pyber's total driver population.


```python
# Import dependencies for calculations and charting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from statistics import mean, median, mode
import seaborn as sns
```


```python
#Import two data CSV files
city_data = pd.read_csv("./raw_data/city_data.csv", low_memory = False)
ride_data = pd.read_csv("./raw_data/ride_data.csv", low_memory = False)
```


```python
#Dropping duplicates in the city csv file
city_data_df = city_data.groupby(['city', 'type']).sum()

#Resetting index before merging datagroups
city_data_df = city_data_df.reset_index()
city_data_df.head()
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
      <th>city</th>
      <th>type</th>
      <th>driver_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alvarezhaven</td>
      <td>Urban</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alyssaberg</td>
      <td>Urban</td>
      <td>67</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anitamouth</td>
      <td>Suburban</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Antoniomouth</td>
      <td>Urban</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aprilchester</td>
      <td>Urban</td>
      <td>49</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Merge the files on common column: city
pyber_df = pd.merge(city_data_df, ride_data, how = 'inner', on = 'city')
pyber_df.head()
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
      <th>city</th>
      <th>type</th>
      <th>driver_count</th>
      <th>date</th>
      <th>fare</th>
      <th>ride_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alvarezhaven</td>
      <td>Urban</td>
      <td>21</td>
      <td>2016-04-18 20:51:29</td>
      <td>31.93</td>
      <td>4267015736324</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alvarezhaven</td>
      <td>Urban</td>
      <td>21</td>
      <td>2016-08-01 00:39:48</td>
      <td>6.42</td>
      <td>8394540350728</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alvarezhaven</td>
      <td>Urban</td>
      <td>21</td>
      <td>2016-09-01 22:57:12</td>
      <td>18.09</td>
      <td>1197329964911</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alvarezhaven</td>
      <td>Urban</td>
      <td>21</td>
      <td>2016-08-18 07:12:06</td>
      <td>20.74</td>
      <td>357421158941</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alvarezhaven</td>
      <td>Urban</td>
      <td>21</td>
      <td>2016-04-04 23:45:50</td>
      <td>14.25</td>
      <td>6431434271355</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculate Average Fare ($) Per City and resetting the index
   # group by city, find the average of the fare column

# average_fare = city_group['fare'].mean()
average_fare = ride_data.groupby("city").mean()

#Reset the index and drop irrelevant information
average_fare = average_fare.reset_index()
average_fare = average_fare.drop(['ride_id'], axis=1)

#Rename the columns
average_fare = average_fare.rename(columns = {'city': 'City'})
average_fare = average_fare.rename(columns = {'fare': 'Average Fare'})

#Print top outcomes
average_fare =round(average_fare,2)
average_fare.head()
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
      <th>City</th>
      <th>Average Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alvarezhaven</td>
      <td>23.93</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alyssaberg</td>
      <td>20.61</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anitamouth</td>
      <td>37.32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Antoniomouth</td>
      <td>23.62</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aprilchester</td>
      <td>21.98</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Calculate Total Number of Rides Per City

total_rides_df = ride_data.groupby("city").count()
total_rides = total_rides_df['ride_id']

#Reset the index
total_rides = total_rides.reset_index()

#Rename the columns
total_rides = total_rides.rename(columns = {'city': 'City'})
total_rides = total_rides.rename(columns = {'ride_id': 'Total Number of Rides'})

#Print top outcomes
total_rides.head()
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
      <th>City</th>
      <th>Total Number of Rides</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alvarezhaven</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alyssaberg</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anitamouth</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Antoniomouth</td>
      <td>22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aprilchester</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculate Total Number of Drivers Per City
total_drivers =  city_data.groupby("city").sum()

#Reset the index
total_drivers = total_drivers.reset_index()

#Rename the columns
total_drivers = total_drivers.rename(columns = {'city': 'City'})
total_drivers = total_drivers.rename(columns = {'type': 'City Type'})
total_drivers = total_drivers.rename(columns = {'driver_count': 'Total Number of Drivers'})

#Print top outcomes
total_drivers.head()
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
      <th>City</th>
      <th>Total Number of Drivers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alvarezhaven</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alyssaberg</td>
      <td>67</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anitamouth</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Antoniomouth</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aprilchester</td>
      <td>49</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Combine city type and total number of drivers
combined_df = city_data_df.rename(columns={'driver_count': 'Total Number of Drivers',
                                           'type':         'City Type',
                                           'city':         'City'})
combined_df.head()
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
      <th>City</th>
      <th>City Type</th>
      <th>Total Number of Drivers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alvarezhaven</td>
      <td>Urban</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alyssaberg</td>
      <td>Urban</td>
      <td>67</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anitamouth</td>
      <td>Suburban</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Antoniomouth</td>
      <td>Urban</td>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aprilchester</td>
      <td>Urban</td>
      <td>49</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Set up bubble plot data and create DataFrame with above information
bubble_plot = pd.merge(average_fare, pd.merge(total_rides, combined_df, on = 'City'), on = 'City')
bubble_plot['Total Fare per City'] = bubble_plot['Average Fare'] * bubble_plot['Total Number of Rides']

#Print top outcomes
bubble_plot.head()
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
      <th>City</th>
      <th>Average Fare</th>
      <th>Total Number of Rides</th>
      <th>City Type</th>
      <th>Total Number of Drivers</th>
      <th>Total Fare per City</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alvarezhaven</td>
      <td>23.93</td>
      <td>31</td>
      <td>Urban</td>
      <td>21</td>
      <td>741.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alyssaberg</td>
      <td>20.61</td>
      <td>26</td>
      <td>Urban</td>
      <td>67</td>
      <td>535.86</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Anitamouth</td>
      <td>37.32</td>
      <td>9</td>
      <td>Suburban</td>
      <td>16</td>
      <td>335.88</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Antoniomouth</td>
      <td>23.62</td>
      <td>22</td>
      <td>Urban</td>
      <td>21</td>
      <td>519.64</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aprilchester</td>
      <td>21.98</td>
      <td>19</td>
      <td>Urban</td>
      <td>49</td>
      <td>417.62</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Begin creating bubble plot - scatter plot testing with data and defining paramters of the bubble plot
bubble_plot.plot(kind='scatter',
                 x = 'Total Number of Rides',
                 y = 'Average Fare',
                 c = 'b',
                 s = bubble_plot['Total Number of Drivers']*5,
                 title = 'Pyber Ride Sharing Data (2016)', ylim = (18,50), alpha = 0.75)

plt.show()
```


![png](output_10_0.png)



```python
# Create bubble plot with above data
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")

# Create bubbleplot of dataframe with Seaborn library
sns.lmplot('Total Number of Rides',
           'Average Fare',
           data = bubble_plot,
           fit_reg = False,
           hue = "City Type",
           scatter_kws = {"marker": "D",
                          "s": bubble_plot['Total Number of Drivers'] *4 })


# Incorporate a text label regarding circle size
plt.text(40, 25, "Note:\nCircle size correlates with driver count per city.")

# Set title and axes labels
plt.title('Pyber Ride Sharing Data (2016)')
plt.xlabel('Total Number of Rides (per City)')
plt.ylabel('Average Fare ($)')
plt.grid(True)
plt.show()
```


![png](output_11_0.png)



```python
# Pie Chart % of Total Fares by City Type
total_fare = pyber_df.groupby("type")["fare"].sum()
plt.title("Total Fare by City Type")
plt.pie(total_fare, labels = ["Rural","Suburban","Urban"],
        colors = ["lightcoral", "lightblue", "gold"],
        explode = (0,0,.1), autopct="%.1f%%")
plt.axis("equal")
plt.show()
```


![png](output_12_0.png)



```python
# Pie Chart % of Total Rides by City Type
total_rides = pyber_df.groupby("type")["ride_id"].sum()
plt.title("Total Rides by City Type")
plt.pie(total_rides, labels = ["Rural","Suburban","Urban"],
        colors = ["lightcoral", "lightblue", "gold"],
        explode = (0,0,0.1), autopct="%.1f%%")
plt.axis("equal")
plt.show()
```


![png](output_13_0.png)



```python
# Pie Chart% of Total Drivers by City Type

total_drivers = pyber_df.groupby("type")["driver_count"].sum()
plt.title("Total Drivers by City Type")
plt.pie(total_drivers, labels = ["Rural","Suburban","Urban"],
        colors = ["lightcoral", "lightblue", "gold"],
        explode = (0.5,0.1,0.1), autopct="%.1f%%")
plt.axis("equal")
plt.show()
```


![png](output_14_0.png)
