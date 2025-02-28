# Multi-Class Classification of Gravitational Lensing Images

This repository contains a Jupyter Notebook (`classification_notebook.ipynb`) that implements a deep learning model for classifying gravitational lensing images into three classes:

* **no:** No substructure
* **sphere:** Strong lensing with no substructure
* **vort:** Vortex substructure

The model is built using PyTorch and uses a ResNet18 architecture with transfer learning and fine-tuning.

## Dataset

**Important:** The dataset is *not* included in this repository due to its size.

**Download Instructions:**

1. Download the `dataset.zip` file from the following Google Drive link:

   [https://drive.google.com/file/d/1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ/view?usp=sharing](https://drive.google.com/file/d/1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ/view?usp=sharing)

   *(Make sure you use a direct download link. If you see a Google Drive preview page, the link is incorrect. A direct download link will immediately start downloading the `dataset.zip` file.)*

2. Run the provided `download_data.py` script to automatically download and extract the dataset:

   ```bash
   python download_data.py
This script uses the gdown library to download the file and zipfile to extract it.
Expected Directory Structure:
After downloading and extracting, your project directory should have the following structure:
Copymulti-image-classification-task/ <-- Your project root
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
│   └── best_model.pth <-- This file will be created during training
├── download_data.py
├── README.md
└── requirements.txt
The models/ directory will be created automatically when you run the notebook and save the best model.
Dependencies
This project requires the following Python packages:

torch
torchvision
numpy
matplotlib
pillow
scikit-learn
seaborn
tqdm
gdown

Install the required packages using pip:
bashCopypip install -r requirements.txt
It's highly recommended to use a virtual environment (e.g., conda or venv) to manage your dependencies.
How to Run

Clone the repository:
bashCopygit clone <your_repository_url>
cd multi-image-classification-task

Download and prepare data using the method explained above.
Open the notebook: Open notebooks/classification_notebook.ipynb in Jupyter Notebook or JupyterLab.
Run all cells: Execute all cells in the notebook sequentially.

Important Notes:

First Run (Grid Search): The first time you run the notebook, make sure perform_grid_search = True in Cell 6. This will perform a hyperparameter grid search to find optimal learning rates. This process can take some time (up to a few hours, depending on your hardware).
Resuming Grid Search: If the grid search is interrupted, you can resume it by simply re-running the notebook. The code will automatically load the previous checkpoint and skip already-tested learning rate combinations.
Loading the Best Model: After the grid search is complete (or if you want to skip the grid search), set perform_grid_search = False in Cell 6. This will load the best model found during the grid search and proceed directly to evaluation or further training.
Continuing Training: To train the best model for additional epochs, set additional_epochs to the desired number of epochs in Cell 10.

Model and Approach
The model is a ResNet18 convolutional neural network, pre-trained on ImageNet. We use transfer learning and fine-tune all layers of the model on the gravitational lensing dataset.
Key Features of the Approach:

Transfer Learning: Leverages the features learned by ResNet18 on the large ImageNet dataset.
Fine-tuning: All layers of the ResNet18 model are unfrozen and fine-tuned on the lensing dataset, allowing the model to adapt to the specific characteristics of the images.
Data Augmentation: The following data augmentation techniques are applied to the training set:

Random Horizontal Flip
Random Rotation (up to 10 degrees)
Random Affine Transformations (small translations and scaling)


Optimizer: Adam optimizer with a learning rate scheduler (ReduceLROnPlateau).
Hyperparameter Tuning: A grid search is performed over a range of learning rates for different layer groups (early, middle, late, and fully connected layers). This helps to find the optimal learning rate configuration for the specific dataset and model architecture.
Three-Way Data Split: The dataset is split into training, validation, and test sets. The validation set is used during the grid search to select the best hyperparameters, and the test set is used for the final, unbiased evaluation.
Checkpointing: The script contains a checkpointing mechanism which saves the best performing model, along with the best parameters, and can resume training from this checkpoint.

Results
The model achieves a Test AUC of approximately 0.989. This is a very high score. Below you'll see a more complete overview, including the loss and AUC curves.
[Add an image of your training history plot here]
The ROC curve and confusion matrix are shown below:
[Add image of ROC curve here]
[Add image of confusion matrix here]
File Descriptions

classification_notebook.ipynb: The main Jupyter Notebook containing the code for data loading, preprocessing, model definition, training, hyperparameter tuning, and evaluation.
download_data.py: A Python script to download and extract the dataset from Google Drive.
requirements.txt: A list of the required Python packages.
README.md: This file, providing an overview of the project.
models/best_model.pth: (Created during training) This file stores the weights of the best-performing model (based on validation AUC).
data/: Not included in repo. Directory where dataset will be located.

Important Considerations (Addressed in the Code)

Data Loading Errors: The LensingDataset class includes error handling to gracefully handle any issues with loading individual .npy files.
Dataset Balance: The notebook includes code to check and confirm that the dataset is balanced across the three classes.
Test Set Creation: The test set is created by splitting off a portion from training set.
Overfitting: The ReduceLROnPlateau scheduler and the use of a separate validation set help to mitigate overfitting. The training/validation loss and AUC curves should be monitored.
Reproducibility: The use of random.seed() is important for reproducibility.
Empty Test Set: Added a check for empty test and gives error, instead of crashing.
Resuming: The grid search will save checkpoints, and it will pick up training from the last saved checkpoint.

