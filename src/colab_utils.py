def setup_colab_environment(dataset_identifier):
    """
    Sets up Kaggle integration and downloads a specified dataset in Google Colab.

    Parameters:
    - dataset_identifier (str): The Kaggle dataset identifier (e.g., "username/dataset-name").

    Usage:
    setup_colab_environment("unclesamulus/blood-cells-image-dataset")
    """
    import os
    from google.colab import files
    !pip install -q kaggle

    # Define Kaggle directory and JSON path
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

    # Ensure Kaggle directory exists and has proper permissions
    os.makedirs(kaggle_dir, exist_ok=True)
    if os.path.exists(kaggle_json_path):
        print("kaggle.json found. Ensuring proper configuration.")
        !chmod 600 ~/.kaggle/kaggle.json
    else:
        print("kaggle.json not found. Please upload the Kaggle API key (kaggle.json):")
        uploaded = files.upload()

        if 'kaggle.json' not in uploaded:
            raise ValueError("Kaggle API key file (kaggle.json) is required.")

        # Move uploaded file to the Kaggle directory
        !mv kaggle.json ~/.kaggle/
        !chmod 600 ~/.kaggle/kaggle.json

    # Download the specified dataset
    print(f"Downloading dataset: {dataset_identifier}")
    !kaggle datasets download -d {dataset_identifier}

    # Unzip the dataset
    zip_file = dataset_identifier.split('/')[-1] + ".zip"
    !unzip -o {zip_file}
    print("Dataset setup complete!")

def setup_colab_environment_V0(dataset_identifier):
    """
    Sets up Kaggle integration and downloads a specified dataset in Google Colab.

    Parameters:
    - dataset_identifier (str): The Kaggle dataset identifier (e.g., "username/dataset-name").

    Usage:
    setup_colab_environment("unclesamulus/blood-cells-image-dataset")
    """
    from google.colab import files
    import os
    !pip install -q kaggle

    # Upload the Kaggle API key
    print("Please upload the Kaggle API key (kaggle.json):")
    uploaded = files.upload()

    if 'kaggle.json' not in uploaded:
        raise ValueError("Kaggle API key file (kaggle.json) is required.")

    # Step 3: Configure Kaggle
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json

    # Step 4: Download the specified dataset
    print(f"Downloading dataset: {dataset_identifier}")
    !kaggle datasets download -d {dataset_identifier}

    # Step 5: Unzip the dataset
    zip_file = dataset_identifier.split('/')[-1] + ".zip"
    !unzip -o {zip_file}
    print("Dataset setup complete!")


# Usage:
dataset_path = "unclesamulus/blood-cells-image-dataset"
setup_colab_environment(dataset_path)
