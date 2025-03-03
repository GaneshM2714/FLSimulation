# client.py
import flwr as fl
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Dict, Optional
from flwr.common import Parameters, Scalar
from numpy import ndarray
from keras.models import load_model
import numpy as np

class ImageClassifierClient(fl.client.NumPyClient):
    """Flower client for image classification using Keras and ImageDataGenerator."""

    def __init__(self, model: keras.Model, train_generator, test_generator):
        self.model = model
        self.train_generator = train_generator
        self.test_generator = test_generator

    def get_parameters(self, config: Dict[str, Scalar]) -> Parameters:
        """Return the current model weights as Flower Parameters."""
        return fl.common.ndarrays_to_parameters(self.model.get_weights())

    def set_parameters(self, parameters: Parameters) -> None:
        """Set the model weights from Flower Parameters."""
        self.model.set_weights(fl.common.parameters_to_ndarrays(parameters))

    def fit(
        self,
        parameters: Parameters,
        config: Dict[str, Scalar],
    ) -> Tuple[Parameters, int, Dict[str, Scalar]]:
        """Train the model on local data using ImageDataGenerator."""
        self.set_parameters(parameters)
        epochs = int(config.get("epochs", 1)) # Get epochs from server config, default to 1
        batch_size = int(config.get("batch_size", 32)) # Get batch size from server config, default to 32

        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            steps_per_epoch=len(self.train_generator), # Train on the entire generator for each epoch
            verbose=0, # Set verbose to 0 for cleaner output during FL
        )
        # Return model parameters, number of examples in train_generator, and training metrics
        return (
            self.get_parameters({}),
            len(self.train_generator) * batch_size, # Total samples trained on (approximately)
            {"accuracy": history.history["accuracy"][-1]}, # Last epoch accuracy
        )

    def evaluate(
        self,
        parameters: Parameters,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on local test data using ImageDataGenerator."""
        self.set_parameters(parameters)
        batch_size = int(config.get("batch_size", 32)) # Get batch size from server config, default to 32

        loss, accuracy = self.model.evaluate(
            self.test_generator,
            steps=len(self.test_generator), # Evaluate on the entire generator
            verbose=0, # Set verbose to 0 for cleaner output
        )
        # Return loss, number of examples in test_generator, and evaluation metrics
        return (
            loss,
            len(self.test_generator) * batch_size, # Total samples evaluated on (approximately)
            {"accuracy": accuracy},
        )


def get_model(model_path):
    """Load a pre-trained model for image classification."""
    
    model = load_model(model_path)
    return model


def create_generators(train_data_dir, test_data_dir, image_size=(32, 32), batch_size=32):
    """Creates train and test ImageDataGenerators from directories."""
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical' # For multi-class classification
    )

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical' # For multi-class classification
    )
    return train_generator, test_generator


def client_fn(train_generator, test_generator, model_input):
    """Creates and returns a Flower client instance."""
    return ImageClassifierClient(model_input, train_generator, test_generator)


def main():
    """Main function to instantiate and run the Flower client."""
    # --- Data Preparation ---
   
    train_data_dir = 'path/to/your/train_data_directory'
    test_data_dir = 'path/to/your/test_data_directory'   
    image_size = (224, 224) 
    batch_size = 32

    train_generator, test_generator = create_generators(
        train_data_dir, test_data_dir, image_size, batch_size
    )

    # --- Model Creation ---
    model_path = '/home/ganesh/Paper_Onkar/VGG19/VGG19_1.keras'
    model = get_model(model_path)

    # --- Flower Client ---
    client = client_fn(train_generator, test_generator, model)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client, grpc_max_message_length = 1024 * 1024 * 1024)


if __name__ == "__main__":
    main()