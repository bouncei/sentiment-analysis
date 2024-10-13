# SCRIPT FOR DOWNLOADING THE DATASET

import requests

def download_dataset(url, save_path):
    """Download the dataset from an external source"""
    
    response = requests.get(url)
    with open(save_path, "wb") as file:
        file.write(response.content)
    
    print(f"Dataset downloaded to {save_path}")
    
    
if __name__ == '__main__':
    # EXAMPLE REQUEST
    # url = "https://example.com/dataset.csv"
    # save_path = '../data/raw/sentiment_data.csv'
    # download_dataset(url, save_path)
    print("Add request url above")