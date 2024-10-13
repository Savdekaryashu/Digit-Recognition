# Handwritten Digit Recognition

This project implements a simple handwritten digit recognition system using TensorFlow and Matplotlib.
The model is trained on the MNIST dataset, which contains 70,000 grayscale images of handwritten digits (0-9).

## Features
- Draw digits on a 28x28 pixel grid.
- Predict the drawn digit using a trained neural network model.
- Clear the canvas to start a new drawing.

## Installation
To run this project, you need to have Python installed on your machine. 
You can then install the required packages by following these steps:

1. Clone the repository:
   git clone https://github.com/Savdekaryashu/Digit-Recognition.git
   cd Digit-Recognition
   
2. Installing requirements:
   pip install -r requirements.txt

## Usage
Run the Predictor.py script
A window will open where you can draw a digit using your mouse.
Press the "Predict" button to see the model's prediction.
Press the "Clear" button to reset the drawing area.

## How It Works
The model is trained on the MNIST dataset using a simple neural network architecture.
When you draw a digit, it gets transformed into a 28x28 pixel grayscale image.
The model then predicts the digit based on the drawn image.

## Acknowledgments
This project uses the MNIST dataset, which is a benchmark dataset for handwritten digit recognition.
Thanks to TensorFlow and Matplotlib for providing powerful libraries for deep learning and data visualization.
    
