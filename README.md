# Intel Scene Classification Using CNNs

This project classifies natural scene images into six categories using
TensorFlow and Keras. It compares a baseline convolutional neural network
(CNN) with a CNN using data augmentation and dropout.

## Classes

- buildings
- forest
- glacier
- mountain
- sea
- street

## Project Objective

The objective is to train and evaluate two CNNs on a small scene
classification dataset, examine their training and validation curves,
and compare their performance on unseen test images.

## Dataset

The project uses a small subset supplied for a deep learning assignment.
The original dataset is available here:

[Intel Image Classification on Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/)

[Assignment dataset](https://drive.google.com/drive/folders/1bevGzEJ7pXbUOWYmRFs6pImmEd6zarcH?usp=sharing)

Dataset ownership remains with the respective owners. This repository
uses the dataset for educational purposes; see the source for applicable
dataset terms.

### Dataset Split

| Split | Number of images | Source |
|---|---:|---|
| Training | 100 | Training folder |
| Validation | 44 | Separate images from the training folder |
| Testing | 100 | Test folder |

Training and validation images do not overlap. Both models use the same
split for comparison. The test set is reserved for final evaluation.

Images are loaded using `image_dataset_from_directory` with
`label_mode="categorical"` and resized to **180 × 180 pixels**.

## Model Architecture

Both models use the following architecture:

| Component | Configuration |
|---|---|
| Input | 180 × 180 × 3 RGB image |
| Rescaling | Pixel values divided by 255 |
| Convolution block 1 | 32 filters + max pooling |
| Convolution block 2 | 64 filters + max pooling |
| Convolution block 3 | 128 filters + max pooling |
| Convolution block 4 | 256 filters + max pooling |
| Convolution block 5 | 512 filters + max pooling |
| Convolution block 6 | 1024 filters + max pooling |
| Flatten | Converts feature maps into a vector |
| Output | 6 neurons with softmax activation |

Each convolution uses a **3 × 3 kernel**, **ReLU activation**, and
`padding="same"`. Each max-pooling layer uses a **2 × 2 pool size**.

Softmax produces a probability distribution over the six classes.

### Baseline Model

The baseline uses the architecture above without augmentation or dropout.

### Augmented Model

The second model adds these transformations before pixel rescaling:

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3)
])
```

A `Dropout(0.5)` layer is added after flattening.

Augmentation and dropout are active during training and inactive during
normal evaluation and prediction.

## Training Configuration

- Optimizer: RMSprop
- Loss: categorical cross-entropy
- Metric: accuracy
- Batch size: 16
- Initial training: 20 epochs per model
- Checkpoint selection: lowest validation loss

The loss function matches the one-hot categorical labels.

Separate `ModelCheckpoint` callbacks save the best baseline and augmented
models. Saved checkpoints are loaded explicitly before evaluation,
because saving a checkpoint does not automatically restore its weights
in the in-memory model.

## Overfitting Analysis

Training and validation accuracy and loss are plotted for both models.

Overfitting is assessed by checking whether training performance improves
while validation performance consistently deteriorates. If reduced-epoch
retraining is needed, the model is rebuilt with fresh weights and trained
for an epoch count selected from the validation curves.

**Final observations:** Replace this paragraph with findings from the
latest plots, including any reduced-epoch retraining performed.

## Results

Update this table with the final measured results:

| Model | Test accuracy | Test loss |
|---|---|---|
| Baseline CNN | TO BE FILLED | TO BE FILLED |
| CNN with augmentation and dropout | TO BE FILLED | TO BE FILLED |

**Comparison:** State which model performed better and explain the
difference using the training curves and test results.

Augmentation and dropout may reduce memorization, but they do not
guarantee higher accuracy. Strong transformations may create unrealistic
scenes, and regularization can make training harder on a small dataset.
Because the second model changes both augmentation and dropout, this
experiment cannot isolate their individual effects.

Results may vary between runs due to random initialization, image
ordering, augmentation, and dropout.

## Predicting an Image

The notebook includes a `predict_scene` function that:

1. Accepts an image path.
2. Loads and resizes the image to 180 × 180.
3. Converts it into a batch.
4. Runs the selected model.
5. Returns the class with the highest predicted probability.

Example using the loaded checkpoints:

```python
from pathlib import Path

image_path = "data/test/sea/20167.jpg"

print("Actual class:", Path(image_path).parent.name)

baseline_prediction = predict_scene(image_path, best_baseline)
augmented_prediction = predict_scene(image_path, best_augmented)

print("Without augmentation:", baseline_prediction)
print("With augmentation:", augmented_prediction)
```

The class-name order must match the order used when loading the datasets.
Images are not divided by 255 again during prediction because rescaling
is already included in the models.

## Running the Notebook

1. Download or clone this repository.
2. Open `scene_classification.ipynb` in Google Colab or Jupyter.
3. Make the dataset available in your environment.
4. Update the dataset path in the notebook.
5. Run the cells in order.
6. Inspect the plots and evaluate the saved checkpoints.
7. Try predictions on individual images.

For local execution, install dependencies:

```bash
pip install -r requirements.txt
```

For Google Colab, mount Google Drive if the dataset is stored there:

```python
from google.colab import drive
drive.mount("/content/drive")
```

The original notebook uses Google Drive paths. Replace them with your own
paths; for local execution, the dataset root can be `Path("data")`.

## Limitations

- Only 100 images are used for training.
- Validation and test results are sensitive to the small sample sizes.
- The CNN has many parameters relative to the training-set size.
- Vertical flips can produce unnatural scene images.
- Results from a single run do not establish that one approach will
  consistently outperform the other.

## Acknowledgments

- The original Intel Image Classification dataset and its contributors.
- The instructor-provided reduced dataset.
- The course reference notebook:
  `chapter08_intro_to_dl_for_computer_vision.ipynb`,
  a companion notebook to *Deep Learning with Python, Second Edition*
  by François Chollet.
