# We import the necessary packages
#import the needed packages
import cv2
import os
import pytesseract
from PIL import Image
import re
from sentence_transformers import SentenceTransformer
import numpy as np


def extract_image_text(image_path, description):
    #We then read the image with text
    images=cv2.imread(image_path)

    #convert to grayscale image
    gray=cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

    cv2.threshold(gray, 0,255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
        
    #adding image to memory
    filename = "{}.jpg".format(os.getpid())
    cv2.imwrite(filename, gray)
    extracted_text = pytesseract.image_to_string(Image.open(filename))
        
    # Regular expression pattern to match log errors
    # This pattern looks for a timestamp at the beginning of the line, followed by keywords like Error, Exception, or Warning
    log_error_pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} -(?: Error| Exception| Warning | Debug):"

    # Filter log errors from the extracted text
    log_errors = [text for text in extracted_text if re.match(log_error_pattern, text)]
    
        # Load a pre-trained sentence transformer model
    model = SentenceTransformer('paraphrase-distilroberta-base-v2')

    # Compute the embeddings for the given description and extracted text
    description_embedding = model.encode(description)

    sentences = extracted_text.split('\n')
    sentence_embeddings = model.encode(sentences)
    # Compute the similarity between the embeddings
    similarity_scores = np.inner(description_embedding, sentence_embeddings)

    # Set a threshold for filtering
    similarity_threshold = 0.6

    # Get the filtered text based on the threshold
    filtered_text = [text for idx, text in enumerate(sentences) if similarity_scores[idx] >= similarity_threshold]

    # print("Filtered text:", filtered_text)
    # print("Log errors:", log_errors)

    os.remove(filename)
    cv2.waitKey(0)
    return filtered_text





