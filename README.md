# ResNet-18 for CIFAR-10

Train a ResNet-18 image classifier on CIFAR-10 and use the saved model to classify
an image selected from a file dialog.

## Requirements

- Python 3.8 or newer
- PyTorch
- torchvision
- Pillow
- matplotlib
- tkinter (usually included with Python)

Install the Python packages:

```bash
python -m pip install torch torchvision Pillow matplotlib
```

## Train

Run the training script from this project directory:

```bash
python "ResNet for CIFAR train.py"
```

The script downloads the CIFAR-10 dataset into `data/` if needed, trains for 15
epochs, reports test accuracy, and displays accuracy and loss plots. It writes
the model checkpoint to `resnet18_cifar10.pth` beside the script. Training uses
CUDA when available and otherwise runs on the CPU.

## Predict an image

The included `resnet18_cifar10.pth` checkpoint can be used directly. Run:

```bash
python "ResNet for CIFAR predict.py"
```

Choose a JPG, JPEG, PNG, or BMP image in the dialog. The script displays the
predicted CIFAR-10 class and the model's confidence. The model checkpoint must
be in the same directory as the prediction script.

## Model details

The model is a ResNet-18 adapted for CIFAR-10: its first convolution uses a
3-by-3 kernel with stride 1, max pooling is removed, and the final layer
classifies the ten CIFAR-10 classes:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`,
`truck`.

The training and prediction scripts use the CIFAR-10 channel normalization
values. The checkpoint contains the model state dictionary.
