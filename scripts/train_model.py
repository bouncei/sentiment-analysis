# SCRIPT FOR TRAINING THE MODEL

from src.train import train_model
from src.config import Config

if __name__ == '__main__':
    train_model(Config.DATA_PATH)
    