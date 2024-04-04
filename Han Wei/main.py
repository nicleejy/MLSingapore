import data.make_dataset as dataset
import models.train_model as train_model

if __name__ == "__main__":
    train, val, test = dataset.data_pipe(0.8, 0.9)
    model = train_model.train(train, val, 1, 32)
    print(model.evaluate())