# Save DataFrame
def save_dataframe(dataframe, file_name):
    try:
        dataframe.to_pickle(f"{file_name}.pkl")
        print(f"DataFrame successfully saved as '{file_name}.pkl'.")
    except Exception as e:
        print(f"An error occurred while saving the DataFrame: {e}")


# Load DataFrame
def load_dataframe(file_name):
    try:
        dataframe = pd.read_pickle(f"{file_name}.pkl")
        print(f"DataFrame successfully loaded from '{file_name}.pkl'.")
        return dataframe
    except Exception as e:
        print(f"An error occurred while loading the DataFrame: {e}")


# Save DataLoader
def save_dataloaders(train_loader, val_loader, test_loader, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Save the datasets (train, val, test)
    torch.save(train_loader.dataset, os.path.join(save_dir, 'train_dataset.pth'))
    torch.save(val_loader.dataset, os.path.join(save_dir, 'val_dataset.pth'))
    torch.save(test_loader.dataset, os.path.join(save_dir, 'test_dataset.pth'))

    # Save the DataLoader configurations
    config = {
        'batch_size': train_loader.batch_size,
        'shuffle': train_loader.shuffle
    }

    torch.save(config, os.path.join(save_dir, 'dataloader_config.pth'))
    print(f"DataLoaders and configurations saved to {save_dir}")


# Load DataLoader
def load_dataloaders(save_dir):
    # Load datasets
    train_dataset = torch.load(os.path.join(save_dir, 'train_dataset.pth'))
    val_dataset = torch.load(os.path.join(save_dir, 'val_dataset.pth'))
    test_dataset = torch.load(os.path.join(save_dir, 'test_dataset.pth'))

    # Load configurations
    config = torch.load(os.path.join(save_dir, 'dataloader_config.pth'))

    # Recreate DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"DataLoaders loaded from {save_dir}")
    return train_loader, val_loader, test_loader


# Processes a directory of folders to create a Pandas DataFrame
def create_image_dataframe(data_folder, output_file):
    """
    Processes a directory of folders to create a Pandas DataFrame.
    Each folder's name is treated as the label for the images inside it.

    Args:
        data_folder (str): Path to the main folder containing subfolders of images.
        output_file (str): Path to save the resulting dataframe as a CSV file.

    Returns:
        pd.DataFrame: A dataframe with columns 'imgPath' and 'label'.
    """
    img_paths = []
    labels = []

    # Traverse each folder in the main data folder
    for label in os.listdir(data_folder):
        label_folder = os.path.join(data_folder, label)
        if os.path.isdir(label_folder):  # Ensure it is a folder
            for img_file in os.listdir(label_folder):
                img_path = os.path.join(label_folder, img_file)
                if os.path.isfile(img_path):  # Ensure it is a file
                    img_paths.append(img_path)
                    labels.append(label)

    df = pd.DataFrame({'imgPath': img_paths, 'label': labels})

    if output_file:
      save_dataframe(df, output_file)

    return df


# Create a Pandas DataFrame from a dataset
def create_filtered_image_dataframe(data_folder, output_file=None, file_extension=".jpg"):
    img_paths = []
    labels = []

    for label in os.listdir(data_folder):
        label_folder = os.path.join(data_folder, label)
        if not os.path.isdir(label_folder):
            continue
        for img_file in os.listdir(label_folder):
            if not img_file.lower().endswith(file_extension):  # Filter by extension
                continue
            img_path = os.path.join(label_folder, img_file)
            img_paths.append(img_path)
            labels.append(label)

    img_paths_series = pd.Series(img_paths, name='imgPath')
    labels_series = pd.Series(labels, name='label')
    df = pd.concat([img_paths_series, labels_series], axis=1)

    if output_file:
        df.to_csv(output_file, index=False)

    return df


# Create a Pandas DataFrame from a dataset
def create_image_label_dataframe(dataset_path):
    # Prepare containers to store image file paths and associated class labels
    image_paths = []
    class_labels = []

    # Iterate over each subdirectory in the dataset directory
    class_directories = os.listdir(dataset_path)
    for class_dir in class_directories:
        class_dir_path = os.path.join(dataset_path, class_dir)  # Full path to the class subdirectory
        if not os.path.isdir(class_dir_path):  # Skip entries that are not directories
            continue
        images = os.listdir(class_dir_path)
        for image_name in images:
            if not image_name.lower().endswith('.jpg'):  # Ensure only JPG files are processed
                continue
            image_path = os.path.join(class_dir_path, image_name)
            image_paths.append(image_path)
            class_labels.append(class_dir)

    # Create a Pandas DataFrame to consolidate image paths and labels
    image_paths_series = pd.Series(image_paths, name='imgPath')
    class_labels_series = pd.Series(class_labels, name='label')
    df = pd.concat([image_paths_series, class_labels_series], axis=1)

    return df


# Prints the memory usage of the current process in MB
def print_memory_usage():
    import psutil

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"RSS (Resident Set Size): {memory_info.rss / 1024**2:.2f} MB")
    print(f"VMS (Virtual Memory Size): {memory_info.vms / 1024**2:.2f} MB")
    print(f"Shared Memory Size: {memory_info.shared / 1024**2:.2f} MB")


def push_notebook_to_github_V1(
    notebook_name,
    commit_message=None,
    branch='colab_branch'
):
    """
    Automatically push the current notebook to GitHub using Colab Secrets

    Args:
    - notebook_name (str): Name of the notebook file to push
    - commit_message (str, optional): Custom commit message
    - branch (str, optional): Git branch to push to (default: 'main')
    """
    from google.colab import userdata, files

    try:
        # Retrieve GitHub credentials from Colab Secrets
        try:
            github_token = userdata.get('GITHUB_TOKEN')
            repo_path = userdata.get('GITHUB_REPO')
            notebook_name = userdata.get('NOTEBOOK_NAME')
        except Exception as secret_error:
            print("❌ Error retrieving GitHub secrets:")
            print(f"   {secret_error}")
            print("Please ensure you've set up both 'GITHUB_TOKEN' and 'GITHUB_REPO' secrets in Colab")
            return

        # Validate credentials
        if not github_token or not repo_path:
            print("❌ Missing GitHub token or repository path.")
            print("Please add 'GITHUB_TOKEN' and 'GITHUB_REPO' in Colab Secrets")
            return

        # Configure Git user information
        !git config --global user.email "$(git config user.email || echo 'colab@example.com')"
        !git config --global user.name "$(git config user.name || echo 'Colab User')"

        # Set up repository URL with token
        repo_url = f"https://{github_token}@github.com/{repo_path}.git"

        # Clone or update the repository
        !git clone {repo_url} colab_repo || (cd colab_repo && git pull)

        # Change to repository directory
        %cd colab_repo

        # Copy the notebook to the repository
        !cp "../{notebook_name}" .

        # Stage the notebook
        !git add "{notebook_name}"

        # Create commit message if not provided
        if commit_message is None:
            from datetime import datetime
            commit_message = f"Update notebook from Colab - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        # Commit changes
        !git commit -m "{commit_message}"

        # Push to specified branch
        !git push origin {branch}

        print(f"✅ Notebook {notebook_name} successfully pushed to {repo_path} on {branch}")

    except Exception as e:
        print(f"❌ Error pushing notebook to GitHub: {e}")


def push_notebook_to_github_V2(
    notebook_name,
    commit_message=None,
    branch='main'
):
    from google.colab import userdata, files

    try:
        github_token = userdata.get('GITHUB_TOKEN')
        repo_path = userdata.get('GITHUB_REPO')

        if not github_token or not repo_path:
            raise ValueError("Missing GITHUB_TOKEN or GITHUB_REPO in Colab Secrets.")

        repo_url = f"https://{github_token}@github.com/{repo_path}.git"

        # Clone repository only if it doesn't exist
        if not os.path.exists('colab_repo'):
            !git clone {repo_url} colab_repo
        else:
            !cd colab_repo && git pull

        os.chdir('colab_repo')  # Navigate to the repo directory

        # Copy the notebook and commit
        !cp "../{notebook_name}" .
        !git add "{notebook_name}"

        if not commit_message:
            from datetime import datetime
            commit_message = f"Update {notebook_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        !git commit -m "{commit_message}"
        !git push origin {branch}

        print(f"✅ Pushed {notebook_name} to GitHub ({repo_path}, branch: {branch}).")

    except Exception as e:
        print(f"❌ Error: {e}")


def push_notebook_to_github_V3(
    notebook_name,
    commit_message=None,
    branch='main'
):
    import os
    from google.colab import userdata, files

    try:
        github_token = userdata.get('GITHUB_TOKEN')
        repo_path = userdata.get('GITHUB_REPO')

        if not github_token or not repo_path:
            raise ValueError("Missing GITHUB_TOKEN or GITHUB_REPO in Colab Secrets.")

        repo_url = f"https://{github_token}@github.com/{repo_path}.git"

        # Clone repository only if it doesn't exist
        if not os.path.exists('colab_repo'):
            !git clone {repo_url} colab_repo
        else:
            !cd colab_repo && git pull

        os.chdir('colab_repo')  # Navigate to the repo directory

        # Copy the notebook and commit
        notebook_path = f"/content/{notebook_name}"
        if not os.path.exists(notebook_path):
            raise FileNotFoundError(f"Notebook {notebook_name} not found in /content.")

        !cp "{notebook_path}" .
        !git add "{notebook_name}"

        if not commit_message:
            from datetime import datetime
            commit_message = f"Update {notebook_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        !git commit -m "{commit_message}"
        !git push origin {branch}

        print(f"✅ Pushed {notebook_name} to GitHub ({repo_path}, branch: {branch}).")

    except Exception as e:
        print(f"❌ Error: {e}")


def push_notebook_to_github_V5(
    notebook_name,
    commit_message=None,
    branch='colab_branch',
    user_mail = 'generic_mail@mail.com',
    user_name = 'generic_name'
):
    """
    Automatically push the current notebook to GitHub using Colab Secrets

    Args:
    - notebook_name (str): Name of the notebook file to push
    - commit_message (str, optional): Custom commit message
    - branch (str, optional): Git branch to push to (default: 'main')
    """
    from google.colab import userdata, files

    try:
        # Retrieve GitHub credentials from Colab Secrets
        try:
            github_token = userdata.get('GITHUB_TOKEN')
            repo_path = userdata.get('GITHUB_REPO')
            repo_name = userdata.get('REPO_NAME')
        except Exception as secret_error:
            print("❌ Error retrieving GitHub secrets:")
            print(f"   {secret_error}")
            print("Please ensure you've set up both 'GITHUB_TOKEN' and 'GITHUB_REPO' secrets in Colab")
            return

        # Validate credentials
        if not github_token or not repo_path:
            print("❌ Missing GitHub token or repository path.")
            print("Please add 'GITHUB_TOKEN' and 'GITHUB_REPO' in Colab Secrets")
            return

        # Configure Git user identity
        !git config user.email "{user_mail}"
        !git config user.name "{user_name}"

        # Set up repository URL with token
        repo_url = f"https://{github_token}@github.com/{repo_path}.git"

        # Clone or update the repository
        !git clone {repo_url} colab_repo #|| (cd colab_repo && git pull)

        # Change to repository directory
        %cd {repo_name}

        # Copy the notebook to the repository
        !cp "/content/{notebook_name}" .

        # Stage the notebook
        !git add "{notebook_name}"

        # Create commit message if not provided
        if commit_message is None:
            from datetime import datetime
            commit_message = f"Update notebook from Colab - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        # Commit changes
        !git commit -m "{commit_message}"

        # Push to specified branch
        !git push origin {branch}

        print(f"✅ Notebook {notebook_name} successfully pushed to {repo_path} on {branch}")

    except Exception as e:
        print(f"❌ Error pushing notebook to GitHub: {e}")


def load_and_normalize_image(image_path):
    try:
        image = Image.open(image_path)

        # Convert to NumPy array with type conversion
        image_array = np.array(image, dtype=np.float32)

        # Normalize pixel values between 0 and 1, handling different image types
        normalized_image = image_array / 255.0

        return normalized_image

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def print_memory_usage_msg(message):
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"{message} - Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")


# Utility for model debbuging
def check_model_numerics(model, input_tensor):
    # Check for any infinite or nan values during forward pass
    model.eval()
    with torch.no_grad():
        try:
            # Check input tensor
            print("Input tensor stats:")
            print(f"Input min: {input_tensor.min()}")
            print(f"Input max: {input_tensor.max()}")
            print(f"Input mean: {input_tensor.mean()}")
            print(f"Input std: {input_tensor.std()}")
            print(f"Any nan in input: {torch.isnan(input_tensor).any()}")
            print(f"Any inf in input: {torch.isinf(input_tensor).any()}")

            # Intermediate checks
            x = input_tensor
            for i, layer in enumerate(model.model_architecture):
                x = layer(x)
                print(f"\nLayer {i} ({layer.__class__.__name__}):")
                print(f"Output shape: {x.shape}")
                print(f"Any nan: {torch.isnan(x).any()}")
                print(f"Any inf: {torch.isinf(x).any()}")
                print(f"Min: {x.min()}")
                print(f"Max: {x.max()}")
                print(f"Mean: {x.mean()}")
                print(f"Std: {x.std()}")

                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"NUMERICAL ISSUE DETECTED IN LAYER {i}")
                    break

        except Exception as e:
            print(f"Error during forward pass: {e}")
