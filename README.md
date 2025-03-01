# Gravitational Lens Binary Classification with ResNet18

This repository contains a Jupyter Notebook implementing a PyTorch-based **ResNet18** model for **binary classification** of astronomical images. The model distinguishes between images containing gravitational lenses and those that do not. The project addresses a significant class imbalance in the provided dataset using weighted random sampling and other best practices.  The notebook uses a pre-trained ResNet18 model, *fine-tuning all layers* for optimal performance on this specific task.

## Task Description

**Specific Test II. Lens Finding**

**Task:** Build a PyTorch model to identify gravitational lenses in astronomical images. This is a *binary classification* problem: determining whether a lens is present or not.

**Dataset:** The dataset comprises observational data of strong lenses and non-lensed galaxies. Images are provided in three different filters for each object, resulting in an array shape of (3, 64, 64) for each object.

*   **Training Data:** Located in the `train_lenses` and `train_nonlenses` directories.
*   **Evaluation Data:** Located in the `test_lenses` and `test_nonlenses` directories.
*   **Class Imbalance:**  The number of non-lensed galaxies is *significantly larger* than the number of lensed galaxies. The code addresses this imbalance using weighted random sampling during training.

**Evaluation Metrics:**

*   ROC curve (Receiver Operating Characteristic curve)
*   AUC score (Area Under the ROC Curve)

**Dataset Link:**

[https://drive.google.com/file/d/1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5/view?usp=drive_link](https://drive.google.com/file/d/1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5/view?usp=drive_link)

## File Structure
```
gravitational_lens/
├── data/
│ └── Specific_test_2/ <- Unzip the dataset HERE
│ ├── train_lenses/ <- Training images WITH lenses (.npy files)
│ ├── train_nonlenses/ <- Training images WITHOUT lenses (.npy files)
│ ├── test_lenses/ <- Test images WITH lenses (.npy files)
│ └── test_nonlenses/ <- Test images WITHOUT lenses (.npy files)
├── notebooks/
│ └── classification_notebooks.ipynb <- Main notebook (using ResNet18)
├── models/ <- Directory for saved models (created automatically)
│ └── lens_finder_model.pth <- Trained model weights (created during training)
├── README.md <- This file
└── requirements.txt <- Python dependencies
```
**VERY IMPORTANT - Dataset Setup:**

1.  **Download:** Download the dataset (`.zip` file) from the Google Drive link above. *Make sure to use a direct download link.*
2.  **Unzip *Directly*:** Extract the contents of the downloaded `.zip` file *directly* into the `data/Specific_test_2/` directory.  *Do not create any extra nested `train` or `test` folders*.  The correct final structure *must* be:
    *   `data/Specific_test_2/train_lenses`
    *   `data/Specific_test_2/train_nonlenses`
    *   `data/Specific_test_2/test_lenses`
    *   `data/Specific_test_2/test_nonlenses`
    *Incorrect* structures like `data/Specific_test_2/train/train_lenses` will cause errors.

## Dependencies

This project requires the following Python packages:

*   **torch:**  The core PyTorch library.
*   **torchvision:**  Provides datasets, models, and transforms for computer vision.
*   **numpy:**  For numerical operations and array handling.
*   **matplotlib:**  For plotting (ROC curves, training history, etc.).
*   **pillow (PIL):**  For image manipulation (used with torchvision transforms).
*   **scikit-learn (sklearn):**  For calculating evaluation metrics (ROC AUC, confusion matrix).
*   **seaborn:**  For creating more visually appealing plots (confusion matrix).
*   **tqdm:**  For displaying progress bars during training.

Install them using pip:

```bash
pip install -r requirements.txt
```
# Gravitational Lens Classification

## Setup and Installation

**Highly Recommended**: Use a virtual environment (e.g., conda or venv) to isolate your project's dependencies and avoid conflicts with other Python projects.

### How to Run - Step-by-Step Instructions

1. **Clone the Repository (Optional)**: If you haven't already, clone this repository to your local machine:
   ```bash
   git clone <your_repository_url>  # Replace with the actual URL
   cd multi-image-classification-task # Or your project directory name
   ```
   If you have not cloned, create a local project with the file structure shown above.

2. **Download and Extract the Dataset**: Download the dataset from the provided Google Drive link and extract its content directly into the `data/Specific_test_2` folder. Double-check that the directory structure matches the "File Structure" section above.

3. **Install Dependencies**: Make sure you have the required Python packages installed (see the "Dependencies" section).
   ```bash
   pip install -r requirements.txt
   ```

4. **Open the Notebook**: Open the `notebooks/classification_notebooks.ipynb` file in Jupyter Notebook or JupyterLab.

5. **Run All Cells**: Execute all cells in the notebook sequentially, from top to bottom. There should be no need to modify any code. The notebook is designed to be self-contained and runnable.

## Notebook Workflow and What to Expect

The notebook performs the following steps:

### 1. Imports
Imports necessary libraries.

### 2. Constants and Directory Creation
- Defines constants like `DATA_DIR`, `MODEL_SAVE_PATH`, `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, and `DEVICE` (which automatically detects if a GPU is available or if the CPU is being used).
- Creates the models directory (where the best model will be saved) if it doesn't already exist.

### 3. Data Directory Verification
The notebook includes a code block that explicitly checks for the existence of the `DATA_DIR` and its required subdirectories (`train_lenses`, `train_nonlenses`, `test_lenses`, `test_nonlenses`). If any of these are missing, the notebook will print an error message and raise a `FileNotFoundError`, preventing the code from running with incorrect paths.

### 4. Device Check
Prints whether a CUDA-enabled GPU is being used (if available) or if the CPU is being used.

### 5. Transformations
Defines the data augmentation and normalization transformations that will be applied to the images:
- **Training Transforms**: Include resizing, random horizontal flips, random rotations, and random affine transformations (small shifts and scaling). These augmentations help the model generalize better. ImageNet normalization is also applied.
- **Validation and Test Transforms**: Include only resizing and ImageNet normalization (no random augmentations).

### 6. Dataset and Data Loaders

#### LensDataset Class
A custom PyTorch Dataset class that handles:
- Loading the `.npy` image files from the specified class directories.
- Applying the appropriate transformations (defined in step 4).
- Returning image-label pairs.
- Includes error handling to gracefully skip any corrupted or unreadable files.
- Has a `report_errors` method to print a summary of any loading errors.

#### create_dataloaders Function
This function is the heart of the data loading process:
- Creates `LensDataset` instances for each of the four class directories: `train_lenses`, `train_nonlenses`, `test_lenses`, and `test_nonlenses`. This ensures that all data is loaded correctly.
- Combines the `train_lenses` and `train_nonlenses` datasets into a single `train_dataset` using `torch.utils.data.ConcatDataset`. This is done before splitting into training and validation, ensuring correct label handling. The same is done for the test sets.
- Splits the combined `train_dataset` into training and validation sets using `torch.utils.data.random_split` (80% train, 20% validation).
- **Handles Class Imbalance**: Calculates the number of samples in each class (lens and non-lens) within the training set. It then creates a `WeightedRandomSampler` that will be used by the training data loader. This sampler ensures that, during training, the model sees a balanced number of samples from each class, even though the original dataset is imbalanced.
- Creates `DataLoader` instances for the training, validation, and test sets:
  - `train_loader`: Uses the `WeightedRandomSampler` for balanced sampling.
  - `val_loader`: Does not use a sampler (shuffles the data, but doesn't weight it).
  - `test_loader`: Does not use a sampler and does not shuffle (for consistent evaluation).
- Returns the three data loaders.
- Added Error Handling: Explicitly raises a `ValueError` if all Datasets are empty, and if all train datasets are empty.

### 7. Model, Loss, Optimizer, Training Loop

#### Model Definition
Defines the ResNet18 model, pre-trained on ImageNet. The final fully connected layer is replaced with a single output neuron and a sigmoid activation function, making it suitable for binary classification.

#### Loss Function
Uses `nn.BCEWithLogitsLoss()`. This combines a sigmoid activation with binary cross-entropy loss, which is the standard loss function for binary classification problems in PyTorch.

#### Optimizer
Uses the Adam optimizer (`optim.Adam`), a popular and generally effective optimization algorithm.

#### Learning Rate Scheduler
Uses `torch.optim.lr_scheduler.ReduceLROnPlateau`. This scheduler dynamically adjusts the learning rate during training. If the validation loss stops improving (plateaus), the learning rate is reduced. This helps the model converge more effectively and avoid getting stuck in local minima.

#### train_model Function
This function contains the core training logic:
- **Epoch Loop**: Iterates over the specified number of training epochs (`NUM_EPOCHS`).
- **Training Phase**:
  - Sets the model to training mode (`model.train()`).
  - Iterates through batches of training data using the `train_loader`. The `tqdm` library provides a progress bar.
  - Moves the input images and labels to the appropriate device (CPU or GPU).
  - **Forward Pass**: Calculates the model's output (predictions) for the current batch.
  - **Loss Calculation**: Calculates the loss between the predictions and the true labels using `criterion`.
  - **Backpropagation**: Calculates the gradients of the loss with respect to the model's parameters.
  - **Optimizer Step**: Updates the model's weights using the calculated gradients and the Adam optimizer.
  - Tracks the running loss and predictions for calculating the training AUC.
  - Calculates and prints the training loss and training AUC for each epoch.
- **Validation Phase**:
  - Sets the model to evaluation mode (`model.eval()`). This disables dropout and batch normalization layers (which behave differently during training and evaluation).
  - Iterates through the validation data (using the `val_loader`) without updating the model's weights (using `torch.no_grad()`).
  - Calculates the validation loss and predictions.
  - Calculates and prints the validation loss and validation AUC.
- **Learning Rate Scheduling**: Calls `scheduler.step(epoch_loss)` to update the learning rate based on the validation loss.
- **Model Saving**: If the current validation AUC is better than the best seen so far, the model's weights are saved to `MODEL_SAVE_PATH`. This ensures that you always have the best-performing model.
- Returns lists of training losses, validation losses, training AUC scores, and validation AUC scores for later plotting.

#### Training Initiation
Calls the `train_model` function.

### 8. Plot Training History
Defines a function `plot_training_history` and plots the training/validation loss and AUC.

### 9. Plotting Utility Functions (ROC Curve and Confusion Matrix)
Defines functions to create and display the ROC curve and confusion matrix.

### 10. Plotting
Calls the plotting utility functions with the results of training.

### 11. Load Best Model and Evaluate
- Loads the best model weights (saved during training).
- Sets the model to evaluation mode.
- Defines an `evaluate_model` function to calculate predictions and labels on the test set.
- Calls `evaluate_model` on the test data.
- Calculates and prints the test AUC.
- Calls the functions from Step 8 to display the ROC curve and confusion matrix for the test set.

## Expected Output

When the notebook is run it should output the data directories and files found, and the device that is being used, and after splitting the data into train, validation and test sets, it will also display the number of examples on each set, and the first labels.

During training, it should also display:
- A progress bar for each epoch, showing training progress.
- The training loss and AUC for each epoch.
- The validation loss and AUC for each epoch.
- Messages indicating when the best model is saved.

After training, it will display:
- Plots of the training and validation loss over epochs.
- Plots of the training and validation AUC over epochs.
- The final test AUC score.
- An ROC curve for the test set.
- A confusion matrix for the test set.

## Important Considerations (Addressed in the Code)

The code is designed to be robust and addresses several important considerations:

1. **Data Loading and Paths**: The `LensDataset` class and `create_dataloaders` function are now guaranteed to correctly load the data, provided the directory structure is set up as described above. Extensive checks and error handling are included.

2. **Class Imbalance**: The `WeightedRandomSampler` is used in the training data loader to ensure that the model sees a balanced representation of lenses and non-lenses during training. This is critical for good performance on imbalanced datasets.

3. **Normalization**: The images are correctly normalized per channel before being converted to tensors. This is a standard practice in image classification and is essential for good model performance. The standard ImageNet normalization is then applied.

4. **Train/Validation/Test Split**: The data is split into three independent sets (training, validation, and test) to ensure unbiased evaluation. The validation set is used to tune hyperparameters (learning rate) and save the best model, while the test set is used only for the final evaluation.

5. **Model Saving**: The best model (based on validation AUC) is saved during training. This allows you to load the best-performing model even if the training process is interrupted.

6. **Error Handling**: In case of errors during data loading, the notebook will print and error and return a black image.

7. **Reproducibility**: Although not explicitly set with `random.seed()`, the random splitting and shuffling operations are handled by PyTorch's data loading utilities, which generally provide good reproducibility if you are using the same version of PyTorch and the same underlying libraries. For absolute reproducibility, you could set the random seed using `torch.manual_seed()` and potentially `numpy.random.seed()`. However, for most practical purposes, the current setup should be sufficient.

## Potential Further Improvements

1. **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, optimizer parameters (e.g., weight decay), and scheduler settings. Consider using techniques like grid search, random search, or Bayesian optimization for more systematic hyperparameter tuning.

2. **More Complex Models**: If ResNet18's performance plateaus, try larger ResNet architectures (ResNet34, ResNet50, ResNet101, ResNeXt) or other pre-trained models (EfficientNet, DenseNet, etc.).

3. **Different Augmentations**: Experiment with other data augmentation techniques (e.g., color jittering, random erasing, Cutout, Mixup). Be careful not to introduce unrealistic augmentations that could mislead the model.

4. **Cross-Validation**: Implement k-fold cross-validation for a more robust estimate of the model's generalization performance.

5. **Error Analysis**: After training, carefully examine the images that the model misclassifies (look at the confusion matrix and potentially visualize some examples). This can give you insights into the model's weaknesses and suggest ways to improve (e.g., collecting more data for specific types of lenses, adjusting the model architecture).

6. **Focal Loss**: Consider using Focal Loss instead of BCEWithLogitsLoss. Focal Loss is specifically designed for imbalanced classification problems and can sometimes improve performance.

7. **Ensemble Methods**: Explore ensemble methods (e.g., bagging, boosting) to combine predictions from multiple models.

8. **Regularization**: Consider adding L1 or L2 regularization in addition to Dropout, to avoid overfitting.
