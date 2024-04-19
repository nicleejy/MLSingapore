import data.make_dataset as dataset
import models.train_model as train_model

if __name__ == "__main__":
    print("Start Data Preparation...")
    train, val, test = dataset.data_pipe(0.8, 0.9)
    print("Start Model Training...")
    model = train_model.train(train, val, 1, 32)
    X_test, y_test = test
    print(model.evaluate(X_test, y_test))