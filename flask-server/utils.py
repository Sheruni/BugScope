
import csv
import codecs
from flask import Flask, jsonify

def read_and_process_csv(file_path):
    data = []

    with codecs.open(file_path, "r", encoding="utf-8", errors="replace") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            data.append({
            'file': row['File'],
            'start line': int(row['Start Line']),
            'end line': int(row['End Line']),
            'code': row['Code Block'],
            'similarity score': float(row['Similarity Score'])
            })

    data.sort(key=lambda x: x['similarity score'])

    return data[:10]

def convert_to_result_format(data):
    results = {}
    for index, item in enumerate(data, start=1):
        results[index] = {
            'file': item['file'],
            'start line': item['start line'],
            'end line': item['end line'],
            'code': item['code']
        }
    return results


