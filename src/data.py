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
