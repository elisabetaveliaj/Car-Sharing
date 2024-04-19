from elasticsearch import Elasticsearch, helpers
import uuid
import pandas as pd
import json


def convert_csv_file_to_bufferized_json_lines_list(csv_file_name, buffer_size):
    df = pd.read_csv(csv_file_name)
    json_list = json.loads(df.to_json(orient='records'))

    buffer_list = []
    total_nr_docs = len(json_list)
    for i in range(0, total_nr_docs, buffer_size):
        buffer_list.append(json_list[i: i + buffer_size])
    return buffer_list


def get_data_from_file(file_name):
    file = open(file_name, encoding="utf8", errors='ignore')
    data = [line.strip() for line in file]
    file.close()
    return data


def bufferize(file_name, buffer_size):
    json_list = get_data_from_file(file_name)
    buffer_list = []
    total_nr_docs = len(json_list)
    for i in range(0, total_nr_docs, buffer_size):
        buffer_list.append(json_list[i: i + buffer_size])
    return buffer_list


def bulk_json(json_buffer, _index):
    for doc in json_buffer:
        # use a `yield` generator so that the data
        # isn't loaded into memory
        if '{"index"' not in doc:
            yield {
                "_index": _index,
                "_id": uuid.uuid4(),
                "_source": doc
            }


def ingest_json_file_into_elastic_index(json_file_name, elastic_client: Elasticsearch, index_name, buffer_size=5000):
    chunks = bufferize(json_file_name, buffer_size)
    for i, buffer in zip(range(len(chunks)), chunks):
        try:
            # make the bulk call, and get a response
            response = helpers.bulk(elastic_client, bulk_json(json_buffer=buffer, _index=index_name))
            print("bulk_json() RESPONSE for chunk:", i, response)
        except Exception as e:
            print("\nERROR:", e)


def ingest_csv_file_into_elastic_index(csv_file_name, elastic_client: Elasticsearch, index_name, buffer_size=5000):
    chunks = convert_csv_file_to_bufferized_json_lines_list(csv_file_name, buffer_size=buffer_size)
    for i, buffer in zip(range(len(chunks)), chunks):
        try:
            # print("__TEST: first document from buffer: ", buffer[0])
            response = helpers.bulk(elastic_client, bulk_json(json_buffer=buffer, _index=index_name))
            print("bulk_json() RESPONSE for chunk:", i, response)
        except Exception as e:
            print("\nERROR:", e)

