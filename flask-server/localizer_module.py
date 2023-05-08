import torch
import csv
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from flask import Flask
from model import Model
from javascript_terms import related_terms
import codecs


def run_localizer(description, keywords):
    # Load the trained model and tokenizer
    model_path = r"D:\bugscope model data\model-final.pt"
    tokenizer_path = r"D:\bugscope model data\tokenizer-v4.pt"

    # model = torch.load(model_path, map_location=torch.device('cpu'))
    tokenizer = torch.load(tokenizer_path, map_location=torch.device('cpu'))
    MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    }

    pretrained_model = "microsoft/codebert-base"
    model_type = 'roberta'
    config_name = pretrained_model
    model_name_or_path = pretrained_model
    tokenizer_name = pretrained_model
    cache_dir=''
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(config_name if config_name else model_name_or_path,
                                        cache_dir=cache_dir if cache_dir else None)
    config.num_labels=1
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name,
                                                do_lower_case=True,
                                                cache_dir=cache_dir if cache_dir else None)

    if model_name_or_path:
        model = model_class.from_pretrained(model_name_or_path,
                                            from_tf=bool('.ckpt' in model_name_or_path),
                                            config=config,
                                            cache_dir=cache_dir if cache_dir else None)    
    else:
        model = model_class(config)



    # Instantiate the model and tokenizer
    model = Model(model, config, tokenizer)

    # Load the state dictionaries
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


    def calculate_bonus(keywords, code_block):
        bonus = 0
        for keyword in keywords:
            if keyword.lower() in related_terms:
                for term in related_terms[keyword.lower()]:
                    if term.lower() in code_block.lower():
                        bonus += 1
        return bonus

    # Inference function
    def inference(model, tokenizer, description, code, keywords):
        model.eval()
        device = next(model.parameters()).device  # Get the device of the model
        
        # Get the maximum length between code and description after tokenizing
        max_length = max(len(tokenizer.tokenize(code)), len(tokenizer.tokenize(description)))
        
        with torch.no_grad():
            code_input = tokenizer.encode(code, return_tensors="pt", padding='max_length', max_length=max_length, truncation=True).to(device)
            nl_input = tokenizer.encode(description, return_tensors="pt", padding='max_length', max_length=max_length, truncation=True).to(device)

            # Pass token_type_ids as None
            _, code_vec, nl_vec = model(code_input, nl_input, return_vec=False)
            bonus = calculate_bonus(keywords, code)
            similarity_score = torch.matmul(nl_vec, code_vec.t()).item()
            
            # Add the weighted bonus to the similarity score
            weight_factor = 0.1  # Adjust this value as needed
            similarity_score += weight_factor * bonus
            
        return similarity_score


    # Read the CSV file and calculate similarity scores
    csv_codeblocks_path = r"D:\bugscope model data\data_blocks.csv"
    csv_updated_path = r"D:\bugscope model data\data_blocks_updated.csv"

    with codecs.open(csv_codeblocks_path, "r", encoding="utf-8", errors="replace") as csv_file:
        csv_reader = csv.reader(csv_file)
        data = [row for row in csv_reader]

    with codecs.open(csv_updated_path, "w", encoding="utf-8", errors="replace") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header row
        csv_writer.writerow(["File", "Start Line", "End Line", "Code Block", "Similarity Score"])
        # Loop through all rows in the CSV file
        for row in data:
            file_path, start_line, end_line, code_block = row
            similarity_score = inference(model, tokenizer, description, code_block, keywords)
            csv_writer.writerow([file_path, start_line, end_line, code_block, similarity_score])
            # print(f"Similarity score between description and code block at {file_path}, lines {start_line}-{end_line}: {similarity_score}")

    return csv_updated_path


if __name__ == '__main__':
    # description = "All items are not displaying when page is loaded"
    # keywords = ["design", "performance"]
    print(run_localizer(description, keywords))
