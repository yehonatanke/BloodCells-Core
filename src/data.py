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
