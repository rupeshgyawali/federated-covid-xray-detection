import sys

import flwr as fl
import tensorflow as tf

def get_compiled_model():
    num_classes = 4
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1./256, input_shape=(256, 256, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes),
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    return model


def get_dataset(dataset_path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path, seed=123, validation_split=0.2, subset="training")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path, seed=123, validation_split=0.2, subset="validation")

    return train_ds, test_ds


class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, train_ds, test_ds, log_dir='./logs/') -> None:
        self.model = model
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.log_dir = log_dir

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir)
        self.model.fit(
            self.train_ds, 
            validation_data=self.test_ds, 
            epochs=1, 
            callbacks=[tensorboard_callback])
        
        return self.model.get_weights(), len(self.train_ds), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.test_ds)
        return loss, len(self.test_ds), {"accuracy": acc}


def start_client(dataset, model, log_dir=None):
    train, test = dataset

    client = FederatedClient(model, train, test, log_dir)

    fl.client.start_numpy_client("0.0.0.0:5700", client=client)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        client_n = int(sys.argv[1])
        dataset = get_dataset('/content/Dataset/client' + str(client_n))
        model = get_compiled_model()
        start_client(
            dataset, 
            model, 
            log_dir='/content/drive/MyDrive/MajorProject/logs/federated/client' + str(client_n))
    else:
        sys.stderr.write('No argumets provided.\n')
