# CNN Shape Classifier

This project is a TensorFlow-based Convolutional Neural Network (CNN) for classifying simple geometric shapes (circle, rectangle, square, ellipse, triangle) from grayscale PNG images.

> 🔁 Looking for the **PyTorch version**? Check it out here: [CNN Shape Classifier (PyTorch)](https://github.com/JacemHaggui/cnn-shape-classifier)

## Project Structure

```
├── dataset.py         # Custom TensorFlow data loader
├── labels.py          # Shape label mapping
├── model.py           # CNN model definition
├── train.py           # Model training script
├── test.py            # Model evaluation script
├── predict.py         # Single image prediction script
├── train/             # Training dataset folder (64x64 PNG images)
├── test/              # Testing dataset folder (64x64 PNG images)
├── .gitignore         # Git ignore file
```

## Requirements

- Python 3.8+
- [TensorFlow](https://www.tensorflow.org/)
- [Pillow](https://python-pillow.org/)

Install dependencies:

```sh
pip install tensorflow pillow
```

## Dataset

- The `train/` and `test/` directories contain the training and testing images.
- Images should be 64×64 grayscale PNGs, named with the shape name and a number (e.g., `circle60.png`).

## Training

Train the model with:

```sh
python train.py
```

This will:

- Load images from `train/`
- Train the CNN (default: 300 epochs — adjust in `train.py`)
- Save the trained model as `CNN_model.keras`

## Testing

Evaluate the model’s accuracy on a folder of images:

```sh
python test.py
```

This will:

- Load images from `train/` (you can change the folder path in `test.py`)
- Print predictions and accuracy statistics

## Predicting a Single Image

Predict the shape in a single image:

```sh
python predict.py
```

Edit the `img_path` variable in `predict.py` to point to your image.

**Example Output**

```
Predicted: ellipse (class 3)
```

## Label Mapping

See `labels.py` for the mapping of shape names to integer labels.

## Model Architecture

See `model.py` for the full CNN definition.
