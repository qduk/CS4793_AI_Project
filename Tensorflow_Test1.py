import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "Data/sample_data.json"

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return  X, y


if __name__ == "__main__":

    # load data
    X, y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # build network topology
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # 1st dense layer
        keras.layers.Dense(512, activation='relu'),

        # 2nd dense layer
        keras.layers.Dense(256, activation='relu'),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu'),

        # output layer
        keras.layers.Dense(10, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50)

    # saves model
    model.save("Model_Test1")

    # this is a duplicate of our saved model, we can pick up training where we left off!!
    reconstructed_model = keras.models.load_model("Model_Test1")

    # let's check:
    np.testing.assert_allclose(
        model.predict(X_test), reconstructed_model.predict(X_test)
    )

    # list all data in history
    print(reconstructed_model.history.history.keys())

    # evaluates the saved model on the test data using evaluate()
    print("Evaluate saved data at epoch:")
    results = reconstructed_model.evaluate(X_test, y_test, batch_size=32)
    print("test loss, test acc:", results)

    # generates some kind of prediction on saved model data, not really sure what kind
    print("Generate predictions for all samples")
    predictions = reconstructed_model.predict(X_test)
    print("predictions shape:", predictions.shape)
