# LEEME:
#
#
#
# Los scripts en esta carpeta fueron realizados en linea con el libro 'Neural networks from scratch with Python;.
# El objetivo es construir una red neuronal desde cero con el objetivo de entender el funcionamiento interno de
# las mismas. Esta carpeta contiene 2 scripts elementales:
#   - classes:  contiene las clases necesarias para crear, configurar, utilizar, guardar y cargar redes neuronales
#               funcionales. Incluye clases 'Layer', 'Activation Function', 'Loss', 'Accuracy' y 'Model'. 
#   - fashion_mnist_test:   implementa diversos testeos para corroborar el funcionamiento de las redes definidas 
#                           en el script de clases. La red para este caso particular es entrenada con los datos
#                           contenidos en el dataset 'FASHION MNIST', una serie de imagenes de ropa que la red 
#                           aprende a identificar. 
#
#
# Algunos pendientes:
#   - Incorporar mecanismo de testeo para imagenes por fuera de las imagenes empleadas para entrenar.
#   - Comentar mejor las clases.

# IMPORTS
import os
import urllib
import urllib.request
from zipfile import ZipFile
import cv2
import numpy as np
import nnfs 
from nnfs.datasets import spiral_data
from classes import *

# FILE MACROS 
URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip' 
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

# PARAMETER MACROS
EPOCHS      = 5
BATCH_SIZE  = 128

nnfs.init()

def download_files(url, file, folder):
    if not os.path.isfile(file):
        print(f' Downloading {url} and saving as {file}...') 
        urllib.request.urlretrieve(url, file)

    print("Unzipping file...")

    with ZipFile(file) as zip_images:
        zip_images.extractall(folder)

    print("Done!")

def load_dataset(dataset, path):

    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path,dataset))

    # Create lists for samples and labels
    x = []
    y = []

    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read
            image = cv2.imread(os.path.join(path, dataset, label, file),
                               cv2.IMREAD_UNCHANGED)
            
            # Append to list
            x.append(image)
            y.append(label)

    return np.array(x), np.array(y).astype('uint8')

def create_data_mnist(path):
    # Load sets separately
    x, y = load_dataset('train', path)
    x_test, y_test = load_dataset('test', path)

    # Return them
    return x, y , x_test, y_test

def create_and_train_model(x, y, x_test, y_test):

    # If not already, download files
    # download_files(URL, FILE, FOLDER)

    # Instantiate and configure model
    model = Model()

    model.add(Layer_Dense(x.shape[1], 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())

    # Set loss, optimizer and accuracy objects
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(decay=1e-3),
        accuracy=Accuracy_Categorical()
    )

    # Finalize
    model.finalize()

    # Train
    model.train(x, y, validation_data=(x_test, y_test), 
                epochs=10, batch_size=128, print_every=100)
        

    # Save model parameters
    model.save_parameters('fashion_mnist.params')

def use_loaded_model_parameters(x, y, x_test, y_test):

    # Instantiate and configure model
    model = Model()

    model.add(Layer_Dense(x.shape[1], 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())

    # Set loss and accuracy objects. No need for optimizer since it'll be loaded
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        accuracy=Accuracy_Categorical()
    )

    # Finalize
    model.finalize()

    # Load parameters
    model.load_parameters('fashion_mnist.params')

    # Evaluate
    model.evaluate(x_test, y_test)

    # Save model
    model.save('fashion_mnist.model')

def load_model(x_test, y_test):

    # Instantiate and configure model
    model = Model.load('fashion_mnist.model')

    # Evaluate
    model.evaluate(x_test, y_test)

def predict(x):

    model = Model.load('fashion_mnist.model')

    confidences = model.predict(x)
    predictions = model.output_layer_activation.predictions(confidences)
    print(predictions)

    # It'd be a good idea to test on other images. Keep in mind, these should be pre-processed to match
    # the input properties that the NN expects. 
    # TO DO 
    # TO DO
    # TO DO

def main():

    # Create dataset
    x, y, x_test, y_test = create_data_mnist('fashion_mnist_images')

    # Shuffle data
    keys = np.array(range(x.shape[0]))
    np.random.shuffle(keys)
    x = x[keys]
    y = y[keys]

    # Scale features
    x = (x.reshape(x.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    x_test = (x_test.reshape(x_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5


    #create_and_train_model(x, y, x_test, y_test)

    #print("\n END OF BLOCK \n")

    #use_loaded_model_parameters(x, y, x_test, y_test)

    #print("\n END OF BLOCK \n")

    load_model(x_test, y_test)
    load_model(x, y)

    #predict(x)

if __name__ == "__main__":
    main()






    
    


