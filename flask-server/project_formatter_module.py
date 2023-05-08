import csv
import zipfile
import os
import shutil
import re

def create_csv_from_project(project_path):
    
    max_length=512
    
    # Regular expressions for matching the start and end of code blocks
    start_pattern = re.compile(r'(function.*?\(.*?\)|const|let|if|for|{|while|def|class|var).*?;', re.DOTALL)
    end_pattern = re.compile(r"^\s*\}")  # Match the end of a code block

    # Define the allowed file extensions
    allowed_extensions = ['.txt', '.js', '.css', '.scss', '.html']

    # Path to the zip folder
    zip_path = project_path

    # Path to the output CSV file
    csv_codeblocks_path = r"D:\bugscope model data\data_blocks.csv"

    # Open the zip folder and extract all files
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("temp")

    # Open the CSV file for writing
    with open(csv_codeblocks_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)

        # Loop through all files in the extracted folder
        for dirpath, _, filenames in os.walk("temp"):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in allowed_extensions):
                    file_path = os.path.join(dirpath, filename)
                    
                    # Read the file and extract code blocks
                    with open(file_path, "r", encoding="utf-8") as file:
                        lines = file.readlines()
                        start_line = None
                        for line_number, line in enumerate(lines, start=1):
                            # Check if the current line starts a new code block
                            if start_pattern.match(line):
                                start_line = line_number
                            # Check if the current line ends the current code block
                            elif start_line is not None and end_pattern.match(line):
                                end_line = line_number
                                code_block = ''.join(lines[start_line-1:end_line])
                                truncated_code_block = ' '.join(code_block.split()[:max_length])
                                csv_writer.writerow([file_path, start_line, end_line, truncated_code_block])
                                start_line = None
                    
                    # Read the file line by line and write to CSV
                    # with open(file_path, "r", encoding="utf-8")as file:
                    #     for line_number, line_content in enumerate(file, start=1):
                    #         csv_writer.writerow([file_path, line_number, line_content])

    # Remove the temporary folder
    shutil.rmtree("temp")
