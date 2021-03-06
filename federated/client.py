import sys

import flwr as fl
import tensorflow as tf

def get_compiled_model():
    num_classes = 4

    data_augmentation = tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip("horizontal",input_shape=(256, 256, 3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    
    model = tf.keras.models.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1./256, input_shape=(256, 256, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


def get_dataset(dataset_path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path, seed=123, validation_split=0.2, subset="training")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path, seed=123, validation_split=0.2, subset="validation")

    # Configure dataset for performance
    # AUTOTUNE = tf.data.AUTOTUNE
    # train_ds = train_ds.cache().shuffle(4500).prefetch(buffer_size=AUTOTUNE)
    # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

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

        csv_logger = tf.keras.callbacks.CSVLogger(
            self.log_dir + 'training.log', 
            append=True)

        self.model.fit(
            self.train_ds, 
            validation_data=self.test_ds, 
            epochs=config.get('epochs', 1), 
            callbacks=[csv_logger])
        
        return self.model.get_weights(), len(self.train_ds), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.test_ds)

        # Log client's local evaluation loss and accuracy
        with open(self.log_dir + 'evaluation.log', 'a') as f:
            f.write(f'{config.get("round")},{loss},{acc}\n')
        
        return loss, len(self.test_ds), {"accuracy": acc}


def start_client(dataset, model, log_dir=None):
    train, test = dataset

    client = FederatedClient(model, train, test, log_dir)

    fl.client.start_numpy_client("0.0.0.0:5700", client=client)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        client_n = int(sys.argv[1])
        dataset = get_dataset('/content/Dataset_NIID/client' + str(client_n))
        model = get_compiled_model()
        start_client(
            dataset, 
            model, 
            log_dir='/content/drive/MyDrive/MajorProject/logs/experiment2/federated/client' + str(client_n))
    else:
        sys.stderr.write('No argumets provided.\n')
