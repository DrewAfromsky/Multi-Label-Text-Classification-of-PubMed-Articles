import argparse
import os
from typing import List
import requests


def query(payload, model_id, api_token):
	headers = {"Authorization": f"Bearer {api_token}"}
	API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


def read_input_file(file_path) -> List[str]:    
    """
    Read the input file and return a list of text samples
    :param file_path: path to the input file
    :return: list of all players numbers
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    if '\n' in lines:
        lines.remove('\n')

    return lines

def validate_user_input(pathname: str) -> bool:
    """
    Validate the user input
    :param pathname: path to a text file with 1 or more text inputs
    :return: True if the input is valid, False otherwise
    """
    if not isinstance(pathname, str) or not pathname: return False
    if not os.path.isfile(pathname): return False
    return True

if __name__ =="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input-file", type=str, help="Path to the txt file")
    args = args.parse_args()
    inputs = read_input_file(args.input_file)

    if validate_user_input(args.input_file):
        # Multi_Label_Classification_of_Pubmed_Articles(args.input_file, )
        model_id = "DAfromsky/Multi-Label-Classification-PubMed-Articles"
        api_token = "hf_XXXXX"
        for input_text in inputs:
            # payload = {"inputs": input_text}
            payload = input_text
            data = query(payload, model_id, api_token)
            print(data)
    else:
        print("Invalid input. Please provide a valid path to a txt file with 1 or more text input entries.")
