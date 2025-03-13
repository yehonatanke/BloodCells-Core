# Functions for loading and preprocessing blood cell data


# Specify the directory containing the dataset
dataset_path = 'bloodcells_dataset'

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

def check_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()  # Check for file corruption
        return True
    except Exception as e:
        print(f"Corrupted image: {file_path} - {e}")
        return False


df['valid_image'] = df['imgPath'].apply(check_image)
# Display corrupted image details
print(df[df['valid_image'] == False])

def check_dimensions(file_path):
    try:
        img = Image.open(file_path)
        return img.size  # (width, height)
    except:
        return None


df['dimensions'] = df['imgPath'].apply(check_dimensions)
print(df['dimensions'].value_counts())  # Count unique dimensions

# Identify irregular dimensions
print(df[df['dimensions'] != (360, 363)])


# Generates a dataset verification report
def dataset_verification_report(df, label_col='label'):
    print("Starting dataset verification...\n")
    print("--- Dataset Verification Report ---\n")

    # Preview of the dataset
    print("Dataset Preview (First 5 Rows):")
    print(df.head())
    print("-" * 50)

    # Dataset summary
    print("Basic Dataset Info:")
    print(df.info())
    print("-" * 50)

    # Check for missing values
    print("Missing Values in Each Column:")
    print(df.isnull().sum())
    print("-" * 50)

    # Unique values per column
    print("Unique Values per Column:")
    print(df.nunique())
    print("-" * 50)

    # Label distribution
    # Expected values
    expected_counts = {
        'neutrophil': 3329,
        'eosinophil': 3117,
        'ig': 2895,
        'platelet': 2348,
        'erythroblast': 1551,
        'monocyte': 1420,
        'basophil': 1218,
        'lymphocyte': 1214
    }

    # Label distribution
    print("Label Distribution:")
    observed_counts = df[label_col].value_counts()
    print(observed_counts)

    # Check for mismatches
    for label, expected_count in expected_counts.items():
        observed_count = observed_counts.get(label, 0)  # Default to 0 if label is missing
        if observed_count != expected_count:
            print(f"Warning: {label} count is {observed_count}, expected {expected_count}.")
    print("-" * 50)

    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    print(f"Number of Duplicate Rows: {duplicate_count}")
    print("-" * 50)

    # Check the dataframe shape
    df_shape = df.shape
    print(f"The dataframe shape: {df_shape}")
    print("-" * 50)

    print("\n --- End of data verification --- \n")


# dataset_verification_report(df, 'label')

# Labels distribution with percentages
def plot_label_distribution(df, label_column):
    # Validate input
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in DataFrame.")
    if df.empty:
        raise ValueError("DataFrame is empty.")

    # Compute label distribution and percentages
    total_images = len(df)
    label_counts = df[label_column].value_counts().reset_index()
    label_counts.columns = [label_column, 'count']
    label_counts['percentage'] = (label_counts['count'] / total_images * 100).round(2)
    # custom_colors = ['#2D3047','#1E3E62', '#5D0E41' ,'#A0153E', '#27374D']
    # custom_colors = ['#060047']
    # custom_colors = ['#134B70']

    fig = px.bar(
        label_counts,
        x=label_column,
        y='count',
        title="Proportional Distribution of Labels in the Dataset",
        labels={label_column: 'Label', 'count': 'Count'},
        text='percentage',
        text_auto='.2s',
        width=10,
        # opacity=0.9,
        # color_discrete_sequence=custom_colors,
        # color=label_column,
    )

    fig.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside',
        hovertemplate=(
            '<b>Label:</b> %{x}<br>'
            '<b>Count:</b> %{y}<br>'
            '<b>Percentage of total:</b> %{text}%'
            '<extra></extra>'
        )
    )

    fig.update_layout(
        #font_size=16,
        title_font_size=18,
        xaxis_title='Label',
        yaxis_title='Count',
        showlegend=False,
        width=1100,
        height=520
    )

    fig.show()


# plot_label_distribution(df, 'label')

class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.classes = sorted(self.dataframe['label'].unique())
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.iloc[idx]['imgPath']
        label_str = self.dataframe.iloc[idx]['label']
        label = self.class_to_idx[label_str]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# Data transformations (resize and normalization)
transform = transforms.Compose([
    transforms.Resize((360, 363)),
    transforms.ToTensor()
])


full_dataset = ImageDataset(df, transform=transform)

# Compares images from two datasets
def compare_datasets_images(dataframe1, dataset2, start_idx=0, num_images=8):
    plt.figure(figsize=(num_images * 2, 4))

    for i in range(num_images):
        idx = start_idx + i

        img_path1 = dataframe1.iloc[idx]['imgPath']
        image1 = Image.open(img_path1)

        image2, label2 = dataset2[idx]
        image2 = image2.permute(1, 2, 0).numpy()
        ax1 = plt.subplot(2, num_images, i + 1)
        ax1.imshow(image1)
        ax1.axis('off')
        ax1.text(0.98, 0.98, f"Range: [0-255] , (Index: {idx})", fontsize=6, color='blue', ha='right', va='top', transform=ax1.transAxes)

        ax2 = plt.subplot(2, num_images, num_images + i + 1)
        ax2.imshow(image2)
        ax2.axis('off')
        ax2.text(0.98, 0.98, f"Range: [0-1] , (Index: {idx})", fontsize=6, color='blue', ha='right', va='top', transform=ax2.transAxes)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


# compare_datasets_images(df, full_dataset, start_idx=0, num_images=8)

def process_image_channels(full_dataset):
    """
    Args:
      full_dataset (ImageDataset): PyTorch dataset containing images

    Returns:
      pd.DataFrame: DataFrame with image channel statistics and labels
    """
    red_sigmas = []
    green_sigmas = []
    blue_sigmas = []
    avg_noises = []
    labels = []

    # Iterate through the dataset
    for idx in range(len(full_dataset)):
        # Get the image and label from the dataset
        img, label = full_dataset[idx]

        # Convert image to numpy array
        img_np = img.numpy() if torch.is_tensor(img) else np.array(img)

        # Ensure the image is in the correct shape (channels, height, width)
        if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
            # If channel-first, transpose to channel-last
            img_np = np.transpose(img_np, (1, 2, 0))

        if img_np.ndim == 0:
            raise ValueError(f"Image at index {idx} is empty or not in the correct format.")

        # Separate color channels
        red_channel = img_np[:,:,0]
        green_channel = img_np[:,:,1]
        blue_channel = img_np[:,:,2]

        # Calculate estimate_sigma for each channel
        red_sigma = estimate_sigma(red_channel, average_sigmas=False)
        green_sigma = estimate_sigma(green_channel, average_sigmas=False)
        blue_sigma = estimate_sigma(blue_channel, average_sigmas=False)

        # Calculate average noise estimate
        avg_noise = estimate_sigma(img_np, channel_axis=-1, average_sigmas=True)

        # Store values
        red_sigmas.append(red_sigma)
        green_sigmas.append(green_sigma)
        blue_sigmas.append(blue_sigma)
        avg_noises.append(avg_noise)
        labels.append(full_dataset.classes[label])

    # Create a new DataFrame with the results
    noise_df = pd.DataFrame({
        'red_channel': red_sigmas,
        'green_channel': green_sigmas,
        'blue_channel': blue_sigmas,
        'img_avg_noise': avg_noises,
        'label': labels
    })

    return noise_df


# noise_df = process_image_channels(full_dataset)

def inspect_data(df):
    print("--- Dataset Inspection ---\n")

    print("Unique values in 'label' column:")
    print(df['label'].unique())
    print('-' * 50)

    print("Shape of the DataFrame:")
    print(df.shape)
    print('-' * 50)

    print("First few rows of the DataFrame:")
    print(df.head())
    print('-' * 50)

    print("Unique Values per Column:")
    print(df.nunique())
    print('-' * 50)

    null_values = df.isnull().sum()
    print("Null Values per Column:")
    print(null_values)
    print('-' * 50)

    print("Total Number of Values for Each Channel:")
    channels = df.columns.tolist()
    for channel in channels:
        if channel in df.columns:
            print(f"{channel}: {df[channel].count()}")
        else:
            print(f"Column '{channel}' not found in DataFrame.")
    print('-' * 50)

    print("Minimum Values for Each Channel:")
    for channel in channels:
        if channel in df.columns:
            if pd.api.types.is_numeric_dtype(df[channel]):
                print(f"{channel}: {df[channel].min()}")
        else:
            print(f"Column '{channel}' not found in DataFrame.")
    print('-' * 50)

    print("Maximum Values for Each Channel:")
    for channel in channels:
        if channel in df.columns:
            if pd.api.types.is_numeric_dtype(df[channel]):
                print(f"{channel}: {df[channel].max()}")
        else:
            print(f"Column '{channel}' not found in DataFrame.")
    print('-' * 50)

    print("--- End of Inspection ---\n")


# inspect_data(noise_df)

