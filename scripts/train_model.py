# SCRIPT FOR TRAINING THE MODEL

from src.train import train_model

if __name__ == '__main__':
    data_path = '../data/raw/IMDB_Dataset.csv'
    train_model(data_path)
    