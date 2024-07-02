MNIST CNN Model
This project involves building, training, and evaluating a Convolutional Neural Network (CNN) on the MNIST dataset using TensorFlow and Keras. The model aims to classify handwritten digits with high accuracy.

Project Structure
mnist_cnn_model.h5: Saved model file.
README.md: Project documentation.
requirements.txt: List of dependencies required for the project.
Dataset
The MNIST dataset is used in this project. It consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is 28x28 pixels.

Preprocessing
The images are normalised by dividing by 255.0. The dataset is reshaped to include a single colour channel.

Model Architecture
The CNN model consists of the following layers:

Input layer with shape (28, 28, 1)
Two Conv2D layers with 32 filters, followed by Batch Normalization, and a MaxPooling layer
Two Conv2D layers with 64 filters, followed by Batch Normalization, and a MaxPooling layer
Two Conv2D layers with 128 filters, followed by Batch Normalization, and a MaxPooling layer
Flatten layer
Dense layer with 256 units and ReLU activation
Output layer with 10 units and softmax activation
Training
The model is compiled with the Adam optimiser, sparse categorical crossentropy loss function, and accuracy as the evaluation metric. The model is trained for 20 epochs with a batch size of 32.

Evaluation
The model’s performance is evaluated on the test dataset. The test accuracy is printed at the end of the training.

Dependencies
To install the required dependencies, run:

pip install -r requirements.txt
