from flask_cors import CORS    
from flask import Flask, request, jsonify
from project_formatter_module import create_csv_from_project
from localizer_module import run_localizer
from utils import read_and_process_csv, convert_to_result_format
from image_text_extractor import extract_image_text
import os

app = Flask(__name__)
CORS(app)

@app.route('/process_data', methods=['POST'])
def process_data():

    # Get zip_path from the POST request
    data = request.get_json()
        
    description = data['description']
    zip_path = data['project_path']
    image_files = data['images']
    tags = data['tags']
    
    image_folder_path = "D:/bugscope2/public/images/"
    project_folder_path = "D:/bugscope2/public/projects/"

    # Extract the filename
    file_name = os.path.basename(zip_path)
    
    # Append the filename to the desired folder path
    project_path = os.path.join(project_folder_path, file_name)
    create_csv_from_project(project_path)


    combined_image_text = ""
    if image_files:
        for image_name in image_files:
            attachment_path = os.path.join(image_folder_path, image_name)
            image_text = extract_image_text(attachment_path, description)
            combined_image_text += " ".join(image_text) + " "  # Adding a space between texts for better readability

    result_csv = run_localizer(description, tags)
    data = read_and_process_csv(result_csv)
    results = convert_to_result_format(data)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)