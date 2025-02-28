# Multi-Class Classification of Gravitational Lensing Images

This repository provides a deep learning solution for classifying gravitational lensing images into three categories:

*   **no:** No substructure
*   **sphere:** Strong lensing with no substructure
*   **vort:** Vortex substructure

The implementation uses a PyTorch-based ResNet18 model with transfer learning and fine-tuning.

## Dataset

**Important:** The dataset is *not* included in this repository due to its size.

**Download and Extraction:**

1.  **Download:** Download the `dataset.zip` file from the following Google Drive link:

    [https://drive.google.com/file/d/1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ/view?usp=sharing](https://drive.google.com/file/d/1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ/view?usp=sharing)

    **Crucial:** Use a *direct* download link.  A direct download link will *immediately* start downloading the `dataset.zip` file. If you are redirected to a Google Drive preview page, the link is *incorrect*.

2.  **Extract:**  Run the provided `download_data.py` script:

    ```bash
    python download_data.py
    ```

    This script uses the `gdown` and `zipfile` libraries to automatically download and extract the dataset.

**Expected Directory Structure:**

After downloading and extracting, your project directory *must* have the following structure:

multi-image-classification-task/  <-- Your project root
    ├── data/
    │   └── dataset/
    │       ├── train/
    │       │   ├── no/
    │       │   ├── sphere/
    │       │   └── vort/
    │       ├── val/
    │       │   ├── no/
    │       │   ├── sphere/
    │       │   └── vort/
    │       └── test/
    │           ├── no/
    │           ├── sphere/
    │           └── vort/
    ├── notebooks/
    │   └── classification_notebook.ipynb
    ├── models/
    │   └── best_model.pth  <-- This file will be created during training
    ├── download_data.py
    ├── README.md
    └── requirements.txt


The `models/` directory will be created automatically when you run the notebook and save the best model.

## Dependencies

This project requires the following Python packages:

*   torch
*   torchvision
*   numpy
*   matplotlib
*   pillow
*   scikit-learn
*   seaborn
*   tqdm
*   gdown

Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```
Highly Recommended: Use a virtual environment (e.g., conda or venv) to manage your dependencies and avoid conflicts.

# How to Run, Model Details, Results, and Important Considerations

This section provides detailed instructions on running the project, describes the model and its key features, summarizes the results, and outlines important considerations addressed in the code.

## How to Run

1.  **Clone the Repository:**

    ```bash
    git clone <your_repository_url>  # Replace <your_repository_url> with the actual URL
    cd multi-image-classification-task
    ```

2.  **Download and Prepare Data:** Follow the instructions in the "Dataset" section of the main README.  This involves downloading `dataset.zip` from the provided Google Drive link and running `python download_data.py`.

3.  **Open the Notebook:** Open `notebooks/classification_notebook.ipynb` in Jupyter Notebook or JupyterLab.

4.  **Run All Cells:** Execute all cells in the notebook sequentially.

## Important Notes & Workflow

*   **First Run (Grid Search):**  On the *very first run*, make sure `perform_grid_search = True` in **Cell 6** of the notebook. This starts the hyperparameter grid search to find the best learning rates.  This is a computationally intensive process and can take several hours, depending on your hardware.

*   **Resuming Grid Search:** If the grid search is interrupted (e.g., due to a power outage or you manually stop it), you can *resume* it.  Simply re-run the notebook. The code is designed to automatically load the previous checkpoint and skip any learning rate combinations that have already been tested.

*   **Loading the Best Model:** After the grid search is complete (or if you want to *skip* the grid search entirely), set `perform_grid_search = False` in **Cell 6**. This will:
    *   Load the best model found during the grid search (if it ran).
    *   Load a default model if the grid search was skipped.
    After this, the notebook will proceed to evaluate the loaded model or allow for further training.

*   **Continuing Training:** To train the loaded model (either the best from the grid search or the default) for additional epochs, set `additional_epochs` to the desired number of epochs in **Cell 10**.

## Model and Approach

The core of this project is a ResNet18 convolutional neural network (CNN), pre-trained on the ImageNet dataset. We utilize transfer learning and fine-tuning to adapt the model to the gravitational lensing image classification task.

**Key Features:**

*   **Transfer Learning:** We leverage the powerful feature representations learned by ResNet18 on the massive ImageNet dataset.  This provides a strong starting point for our model, even with a relatively smaller dataset of lensing images.

*   **Fine-tuning:**  Instead of just using the pre-trained weights as fixed feature extractors, we *unfreeze* all layers of the ResNet18 model. This allows the backpropagation algorithm to adjust *all* the weights during training, fine-tuning the model to the specific characteristics of our gravitational lensing data.

*   **Data Augmentation:** To improve the model's generalization ability and reduce overfitting, we apply the following data augmentation techniques to the *training* set:
    *   **Random Horizontal Flip:** Randomly flips images horizontally.
    *   **Random Rotation:** Rotates images by a random angle up to 10 degrees.
    *   **Random Affine Transformations:** Applies small random translations (shifts) and scaling to the images.

*   **Optimizer:** We use the Adam optimizer, a popular and efficient stochastic gradient descent algorithm.

*   **Learning Rate Scheduler:** A `ReduceLROnPlateau` learning rate scheduler is employed.  This dynamically adjusts the learning rate during training.  If the validation loss plateaus (stops improving), the learning rate is reduced.  This helps the model converge to a better solution and avoid getting stuck in local minima.

*   **Hyperparameter Tuning (Grid Search):** A crucial part of the approach is a grid search over a range of learning rates.  We explore different learning rates for different groups of layers within the ResNet18 architecture:
    *   Early layers
    *   Middle layers
    *   Late layers
    *   Fully connected layers
    This systematic search helps identify the optimal learning rate configuration for our specific dataset and model.

*   **Three-Way Data Split:** The dataset is divided into three distinct sets:
    *   **Training Set:** Used to train the model's weights.
    *   **Validation Set:** Used during the grid search to evaluate the performance of different hyperparameter combinations (learning rates) and select the best ones.
    *   **Test Set:**  Used *only* at the very end to provide a final, unbiased evaluation of the trained model's performance. This ensures that our reported results are not overly optimistic.

*   **Checkpointing:**  The code includes a robust checkpointing mechanism.  During the grid search and training:
    *   The best-performing model (based on validation AUC) is saved to `models/best_model.pth`.
    *   The corresponding hyperparameters (learning rates) are also saved.
    This allows us to resume training from where it left off if interrupted and to easily load the best model later.

## Results

The model achieves a Test AUC (Area Under the Receiver Operating Characteristic Curve) of approximately **0.989**. This high AUC score indicates excellent performance in classifying the gravitational lensing images.  More detailed results, including training/validation loss and AUC curves over epochs, are presented within the notebook.

**(Optional: Include Images Here)**

If you have generated images during the notebook execution (e.g., using `matplotlib`), you can include them here for a more comprehensive report.

*   **Training History:**  [Insert image of training history plot here - e.g., `images/training_history.png`]
*   **ROC Curve:** [Insert image of ROC curve here - e.g., `images/roc_curve.png`]
*   **Confusion Matrix:** [Insert image of confusion matrix here - e.g., `images/confusion_matrix.png`]

## File Descriptions

*   **`classification_notebook.ipynb`:** The main Jupyter Notebook. This contains all the code for:
    *   Data loading and preprocessing
    *   Model definition (ResNet18)
    *   Training loop
    *   Hyperparameter tuning (grid search)
    *   Model evaluation (AUC, ROC curve, confusion matrix)
    *   Visualization of results

*   **`download_data.py`:** A Python script to automatically download and extract the `dataset.zip` file from the provided Google Drive link.

*   **`requirements.txt`:**  Lists all the required Python packages needed to run the project.  Use `pip install -r requirements.txt` to install them.

*   **`README.md`:** This file (the one you are reading), which provides a comprehensive overview of the project.

*   **`models/best_model.pth`:**  *(Created during training)*  This file stores the weights of the best-performing model (the one with the highest validation AUC) found during the training process.

*   **`data/`:** *(Not included in the repository; download required)* This directory will contain the dataset after you run `download_data.py`. It should have the structure: `data/dataset/train/`, `data/dataset/val/`, and `data/dataset/test/`, each with subdirectories for `no`, `sphere`, and `vort`.

## Important Considerations (Addressed in the Code)

The following important considerations have been addressed in the code to ensure robustness, reliability, and good performance:

*   **Data Loading Errors:** The `LensingDataset` class (in the notebook) includes error handling. If there's a problem loading a specific `.npy` file (e.g., it's corrupted), the code will gracefully skip that file and continue, rather than crashing.  This is important for dealing with potential data issues.

*   **Dataset Balance:** The notebook includes code to check and confirm that the dataset is balanced across the three classes (`no`, `sphere`, `vort`).  A balanced dataset is important for preventing the model from being biased towards the majority class.

*   **Test Set Creation:** The test set is created by splitting off a portion of the *original* training data. This ensures that the test set is independent and representative of the data distribution.

*   **Overfitting Mitigation:** Several techniques are used to prevent overfitting:
    *   **`ReduceLROnPlateau` Scheduler:** Dynamically adjusts the learning rate to prevent the model from getting stuck in local minima and to fine-tune convergence.
    *   **Validation Set:**  A separate validation set is used to monitor performance during training and select the best hyperparameters.  This helps prevent the model from simply memorizing the training data.
    *   **Data Augmentation:**  Increases the effective size of the training set and makes the model more robust to variations in the input images.

*   **Reproducibility:**  `random.seed()` is used to ensure that the results are reproducible.  Setting a random seed makes the random number generation consistent, so you should get the same results each time you run the notebook (assuming the same data and hyperparameters).

*   **Empty Test Set Handling:**  A check is included to prevent the code from crashing if the test set is accidentally empty.  Instead of crashing, an informative error message will be raised.

*   **Resuming Training:** The checkpointing mechanism (saving `models/best_model.pth`) allows you to resume training from the last saved checkpoint. This is crucial for long training runs or if the training process is interrupted.
