from elasticsearch import Elasticsearch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# initializing Python client: Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# assigning the index for easier use
index_name = "car_requests"

# querying weekdays, grouping them by time intervals of 1h, and ensuring that we are taking intervals with at least one request
query = {
    "size": 0,
    "aggs": {
        "weekends": {
            "filter": {
                "term": {
                    "weekend": 0  # 0 represents weekdays
                }
            },
            "aggs": {
                "hours": {
                    "date_histogram": {
                        "field": "origin_datetime",
                        "calendar_interval": "1h",
                        "min_doc_count": 1
                    }
                }
            }
        }
    }
}

# Execute the query
response = es.search(index=index_name, body=query)

# Extract data from response
buckets = response['aggregations']['weekends']['hours']['buckets']

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame([(bucket['key_as_string'], bucket['doc_count']) for bucket in buckets],
                  columns=['DateTime', 'Demand'])

# Convert DateTime to proper datetime type(since above its a string) and set as index
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True) # create datetime as an index and do it in the dsame dataframe, without creating a new one
df['Hour'] = df.index.hour

# group by hour and calculate statistics
hourly_stats = df.groupby('Hour')['Demand'].describe()
print(hourly_stats)

# calculate overall mean demand and standard deviation
# insights into distribution and variability of the demand
overall_mean_demand = df['Demand'].mean()
overall_std_demand = df['Demand'].std()

# plotting
plt.figure(figsize=(14, 7))

# plotting hourly mean demand as bars
bars = plt.bar(hourly_stats.index, hourly_stats['mean'], color='cornflowerblue', label='Mean Hourly Demand')

# plotting standard deviation as error bars
plt.errorbar(hourly_stats.index, hourly_stats['mean'], yerr=hourly_stats['std'],
             fmt='^', color='red', ecolor='lightgreen', elinewidth=2, capsize=5, label='Standard Deviation')

# adding overall mean demand and standard deviation lines
plt.axhline(overall_mean_demand, color='green', linestyle='--', label='Overall Mean Demand')
plt.axhline(overall_mean_demand + overall_std_demand, color='red', linestyle='--', label='Overall +1 STD')
plt.axhline(overall_mean_demand - overall_std_demand, color='red', linestyle='--', label='Overall -1 STD')

# setting up the plot
plt.title('Statistical Measures of Hourly Customer Demand on Weekdays')
plt.xlabel('Hour of the Day')
plt.ylabel('Demand')
plt.xticks(hourly_stats.index)
plt.legend()
plt.grid(axis='y')

plt.show()

# 3b part:

# calculate the expected number of requests on a working day
expected_requests_weekday = hourly_stats['mean'].sum()

print("Expected demand with the provided data: ", expected_requests_weekday)

# Adjust for satisfied demand by increasing the expected number of requests by 30%
adjusted_expected_requests = expected_requests_weekday * 1.30

print("Expected number of requests on a working day (adjusted demand):", adjusted_expected_requests)

# 3c part:

# Sample size is the adjusted expected number of requests
sample_size = int(adjusted_expected_requests)
print(sample_size)

# Initialize the sample
sampled_demand = []


# Function to randomly sample demand for an hour
def sample_hourly_demand(mean, std, hour, total_sample_size, weight):
    # Determine the number of samples for the hour based on its weight
    num_samples = int(total_sample_size * weight)
    # Randomly sample demand using the normal distribution around the mean
    samples = np.random.normal(loc=mean, scale=std, size=num_samples)
    # Ensure demand is not negative
    samples = [max(0, sample) for sample in samples]
    # Add the hour information
    return [(hour, demand) for demand in samples]


# Total demand across all hours used for weights
total_mean_demand = hourly_stats['mean'].sum()

# Generate the sample
for hour, row in hourly_stats.iterrows():
    weight = row['mean'] / total_mean_demand  # Weight by the mean demand for the hour
    hourly_samples = sample_hourly_demand(row['mean'], row['std'], hour, sample_size, weight)
    sampled_demand.extend(hourly_samples)

# create a DataFrame from the sampled demand
sampled_demand_df = pd.DataFrame(sampled_demand, columns=['Hour', 'SampledDemand'])
sampled_demand_df['AdjustedSampledDemand'] = sampled_demand_df['SampledDemand'] * 1.30

# Aggregate this sample to get the mean and standard deviation for each hour
sampled_stats = sampled_demand_df.groupby('Hour')['AdjustedSampledDemand'].agg(['mean', 'std'])

# Calculate overall mean and standard deviation across all hours
sampled_overall_mean_demand = sampled_stats['mean'].mean()
sampled_overall_std_demand = sampled_stats['mean'].std()


# Plotting the data
plt.figure(figsize=(14, 7))

# Plot the bar chart for mean demand with error bars
bars = plt.bar(sampled_stats.index, sampled_stats['mean'], color='cornflowerblue', label="Sampled Mean Hourly Demand")
error_bars = plt.errorbar(sampled_stats.index, sampled_stats['mean'], yerr=sampled_stats['std'],  fmt='^', color='red',
                          ecolor='lightgreen', elinewidth=2, capsize=5, label='Standard Deviation')

# Plot overall mean demand line
plt.axhline(sampled_overall_std_demand, color='orange', linestyle='--', label='Overall Mean Demand')

# Plot overall standard deviation lines
plt.axhline(sampled_overall_std_demand + sampled_overall_std_demand, color='red', linestyle=':', label='Overall STD Deviation')
plt.axhline(sampled_overall_std_demand - sampled_overall_std_demand, color='red', linestyle=':')

# Configuring the plot:
plt.title('Simulated Sample of Hourly Customer Demand on Weekdays')
plt.xlabel('Hour of the Day')
plt.ylabel('Demand')
plt.xticks(sampled_stats.index)
plt.legend()
plt.grid(axis='y')

plt.show()


# ##### not requested in the case file, but I wanted to check the difference in demand weekday vs weekend

# querying weekends
weekend_query = {
    "size": 0,
    "aggs": {
        "weekends": {
            "filter": {
                "term": {
                    "weekend": 1
                }
            },
            "aggs": {
                "hours": {
                    "date_histogram": {
                        "field": "origin_datetime",
                        "calendar_interval": "1h",
                        "min_doc_count": 1
                    }
                }
            }
        }
    }
}

# execute the query for weekend data
weekend_response = es.search(index=index_name, body=weekend_query)

# extract weekend data from response
weekend_buckets = weekend_response['aggregations']['weekends']['hours']['buckets']

# Convert to pandas DataFrame
weekend_df = pd.DataFrame([(bucket['key_as_string'], bucket['doc_count']) for bucket in weekend_buckets],
                  columns=['DateTime', 'Demand'])

# Convert DateTime
weekend_df['DateTime'] = pd.to_datetime(weekend_df['DateTime'])
weekend_df.set_index('DateTime', inplace=True)
weekend_df['Hour'] = weekend_df.index.hour

# Group by hour and calculate statistics for weekend
weekend_hourly_stats = weekend_df.groupby('Hour')['Demand'].describe()

# Plotting hourly demand on weekends
plt.figure(figsize=(14, 7))
plt.bar(weekend_hourly_stats.index, weekend_hourly_stats['mean'], color='cornflowerblue', label='Mean Hourly Demand (Weekend)')
plt.errorbar(weekend_hourly_stats.index, weekend_hourly_stats['mean'], yerr=weekend_hourly_stats['std'],
             fmt='o', color='darkblue', ecolor='orange', elinewidth=3, capsize=5, label='Standard Deviation (Weekend)')

# Adding overall mean demand and standard deviation lines for weekends
plt.axhline(weekend_hourly_stats['mean'].mean(), color='blue', linestyle='--', label='Overall Mean Demand (Weekend)')
plt.axhline(weekend_hourly_stats['mean'].mean() + weekend_hourly_stats['std'].mean(), color='orange', linestyle='--', label='Overall +1 STD (Weekend)')
plt.axhline(weekend_hourly_stats['mean'].mean() - weekend_hourly_stats['std'].mean(), color='orange', linestyle='--', label='Overall -1 STD (Weekend)')

# Setting up the plot
plt.title('Hourly Customer Demand on Weekends')
plt.xlabel('Hour of the Day')
plt.ylabel('Demand')
plt.xticks(weekend_hourly_stats.index)
plt.legend()
plt.grid(axis='y')

plt.show()

# Calculate the overall demand on weekends
total_weekend_demand = weekend_hourly_stats['mean'].sum()

# Calculate the overall demand on weekdays for comparison
total_weekday_demand = hourly_stats['mean'].sum()

# Calculate the percent difference between weekend and weekday demand
comparison_percent = ((total_weekend_demand - total_weekday_demand) / total_weekday_demand) * 100

print(comparison_percent)