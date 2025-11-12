from src.train import train_model

if __name__ == "__main__":
    decoder, history = train_model(epochs=20, batch_size=8)
