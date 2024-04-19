from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan, bulk
from loaders import ingest_json_file_into_elastic_index
from datetime import datetime
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import json

# initializing Python client: Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}]).options(ignore_status=[400, 405])

# assigning the index for easier use
index_name = "car_requests"

# assigning json file name or path for easier use
json_file_name = "request_data.json"


# creating a function to correct the lats and lons in the "request_data" file since they were switched
def fix_json_coordinates(file_path):
    fixed_data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                # swap longitude and latitude in origin_location and destination_location
                data['origin_location'] = [data['origin_location'][1], data['origin_location'][0]]
                data['destination_location'] = [data['destination_location'][1], data['destination_location'][0]]
                fixed_data.append(data)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from line: {line.strip()}")

    # write the fixed data back to the file
    with open(file_path, 'w') as file:
        for item in fixed_data:
            file.write(json.dumps(item) + '\n')


# calling the function to correct coordinates. should be run once and then commented out.
fix_json_coordinates(json_file_name)

# writing mappings for storing indices correctly
settings = {
    "settings": {
        "number_of_shards": 3
    },
    "mappings": {
        "properties": {
            "request_nr": {"type": "integer"},
            "origin_datetime": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"},
            "weekend": {"type": "integer"},
            "origin_location": {"type": "geo_point"},
            "destination_datetime": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"},
            "destination_location": {"type": "geo_point"}
        }
    }
}

# create indices and ingest json file to ES. should be run once and then commented out in order to not index again.
es.indices.create(index=index_name, body=settings)
ingest_json_file_into_elastic_index(json_file_name, es, index_name, buffer_size=5000)

# print the initial number of entries
print(f"Initial number of entries = {es.count(index=index_name)['count']}")

# determining the batch size. whenever the code reaches this size the code sends a bulk update request to Elasticsearch
batch_size = 10000

# list to store bulk update requests
batch = []

# selecting all with ES query
query = {
    "query": {
        "bool": {
            "must": {
                "match_all": {}
            }
        }
    }
}

# created new timeout setting due to some errors when running the code on my PC
es_different_timeout = es.options(request_timeout=60)

for hit in scan(client=es, index=index_name, query=query):
    origin_location = hit['_source']['origin_location']
    destination_location = hit['_source']['destination_location']
    origin_datetime = datetime.strptime(hit['_source']['origin_datetime'], "%Y-%m-%d %H:%M:%S")
    destination_datetime = datetime.strptime(hit['_source']['destination_datetime'], "%Y-%m-%d %H:%M:%S")

    # check whether location and datetime are different. If so, set delete = 0, otherwise set delete = 1
    if origin_location != destination_location and origin_datetime != destination_datetime:
        # calculate distance
        distance_km = geodesic(origin_location, destination_location).kilometers

        # Calculate time difference in hours
        time_delta = destination_datetime - origin_datetime
        time_hours = time_delta.total_seconds() / 3600

        # Calculate speed
        speed_kmh = distance_km / time_hours

        # Check whether speed is reasonable. If not, set delete = 1
        if speed_kmh <= 50 and speed_kmh >= 10:
            # Update the document with distance, time, and speed, and set delete = 0
            update_request = {
                "_op_type": "update",
                "_index": index_name,
                "_id": hit['_id'],
                "doc": {
                    "distance_km": distance_km,
                    "time_hours": time_hours,
                    "speed_kmh": speed_kmh,
                    "demand": 1,
                    "delete": 0
                }
            }
            # update request to batch
            batch.append(update_request)
        else:
            # Update the document with distance, time, and speed, and set delete = 1
            update_request = {
                "_op_type": "update",
                "_index": index_name,
                "_id": hit['_id'],
                "doc": {
                    "distance_km": distance_km,
                    "time_hours": time_hours,
                    "speed_kmh": speed_kmh,
                    "demand": 1,
                    "delete": 1
                }
            }
            # Add update request to batch
            batch.append(update_request)

    else:
        # Update the document with distance, time, and speed, and set delete = 1
        update_request = {
            "_op_type": "update",
            "_index": index_name,
            "_id": hit['_id'],
            "doc": {
                "distance_km": distance_km,
                "time_hours": time_hours,
                "speed_kmh": speed_kmh,
                "demand": 1,
                "delete": 1
            }
        }
    # update request to batch. Using a batch is just a faster way of updating
    batch.append(update_request)

    if len(batch) >= batch_size:
        bulk(client=es_different_timeout, actions=batch)
        batch = []
        print("sent to elasticsearch")

if batch:
    bulk(client=es_different_timeout, actions=batch)

print("Distance, time, speed calculations and updates completed.")

# delete all element which have delete = 1
query = {
    "query": {
        "term": {
            "delete": {
                "value": 1
            }
        }
    }
}

es.options(request_timeout=60).delete_by_query(index=index_name, body=query, wait_for_completion=True)

# Determine the number of entries after deleting the incorrect ones
print(f"Number of entries after data cleaning = {es.count(index=index_name)['count']}")

# aggregation query
body = {
    "size": 0,
    "aggs": {
        "requests_per_day": {
            "date_histogram": {
                "field": "origin_datetime",
                "calendar_interval": "day"
            }
        }
    }
}

# execute the query
response = es.search(index=index_name, body=body)

# extracting aggregation results
buckets = response['aggregations']['requests_per_day']['buckets']
dates = [datetime.strptime(bucket['key_as_string'], '%Y-%m-%d %H:%M:%S').date() for bucket in buckets]
counts = [bucket['doc_count'] for bucket in buckets]

# plotting the time series
plt.figure(figsize=(14, 7))
plt.plot(dates, counts, linestyle='-', color='cornflowerblue')
plt.title('Daily Demand for Carsharing Requests')
plt.xlabel('Date')
plt.ylabel('Number of Requests')
plt.grid(True)
plt.tight_layout()
plt.show()
