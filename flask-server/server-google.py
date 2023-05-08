from flask_cors import CORS    
import csv
import requests
from io import StringIO
from flask import Flask, jsonify
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

app = Flask(__name__)
CORS(app)

# Replace with the path to your downloaded JSON file containing the credentials
SERVICE_ACCOUNT_FILE = 'bugscope-4a067716a0a0.json'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_google_drive_service():
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    return service

def read_csv_from_drive(service, file_id):
    try:
        request = service.files().get_media(fileId=file_id)
        file = request.execute()
        csv_content = file.decode('utf-8')
        csv_file = StringIO(csv_content)
        return csv_file
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None

def read_and_process_csv(file_id):
    service = get_google_drive_service()
    csv_file = read_csv_from_drive(service, file_id)

    data = []
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

    return data

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

# Replace with the Google Drive file ID of your CSV file
file_id = '1pgLQP2SxLA6ZGxPkAWzHcyFFPHqB2Ye2'
data = read_and_process_csv(file_id)
results = convert_to_result_format(data)

@app.route('/api/code-snippets', methods=['GET'])
def get_code_snippets():
    return jsonify(results=results)

if __name__ == '__main__':
    app.run(debug=True)
