# Visualize the distribution of the images dimensions
def visualize_dimensions(df, dimensions_column):
    # Prepare data
    dimension_counts = df[dimensions_column].value_counts().reset_index()
    dimension_counts.columns = ['Dimensions', 'Count']
    dimension_counts['Dimensions'] = dimension_counts['Dimensions'].astype(str)  # Convert dimensions to string for better labeling
    total_count = dimension_counts['Count'].sum()

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        specs=[[{"type": "xy"}, {"type": "domain"}]]
    )

    fig.add_trace(
        go.Bar(
            x=dimension_counts['Dimensions'],
            y=dimension_counts['Count'],
            text=dimension_counts['Count'],
            textposition='outside',
            hovertemplate=(
                "Image dimensions: %{x}<br>"
                "Image count: %{y}<br>"
                "Percentage from total: %{customdata:.2f}%<extra></extra>"
            ),
            customdata=(dimension_counts['Count'] / total_count * 100),
            showlegend=False
        ),
        row=1, col=1
    )

    # Pie Chart
    fig.add_trace(
        go.Pie(
            labels=dimension_counts['Dimensions'],
            values=dimension_counts['Count'],
            hovertemplate=(
                "Image dimensions: %{label}<br>"
                "Image count: %{value}<br>"
                "Percentage from total: %{percent}<extra></extra>"
            ),
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text="Distribution of Image Dimensions",
        height=600,
        width=1000
    )

    fig.show()


# visualize_dimensions(df, 'dimensions')

# Polar bar chart label distribution
def plot_label_distribution_polar(df, label_column):
    # Validate input
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in DataFrame.")
    if df.empty:
        raise ValueError("DataFrame is empty.")

    # Calculate the distribution of labels and percentages
    label_counts = df[label_column].value_counts().reset_index()
    label_counts.columns = ['label', 'count']
    total_images = len(df)
    label_counts['percentage'] = (label_counts['count'] / total_images * 100).round(2)

    # Bin the percentages into discrete intervals
    num_bins = 30
    label_counts['percentage_bin'] = pd.cut(
        label_counts['percentage'],
        bins=np.linspace(0, label_counts['percentage'].max(), num_bins + 1),
        labels=[f"{int(i.left)}-{int(i.right)}%" for i in pd.interval_range(0, label_counts['percentage'].max(), num_bins)],
        include_lowest=True
    )

    fig = px.bar_polar(
        label_counts,
        r='percentage',
        theta='label',
        color='percentage_bin',
        labels={'label': 'Label',
                'count': 'Count',
                'percentage': 'Percentage of total',
                'percentage_bin' : 'Percentage'},
        title="Polar Visualization of Label Distributions with Proportional Metrics",
        color_discrete_sequence= px.colors.sequential.Plasma_r,
    )

    fig.update_traces(
        hovertemplate=(
            '<b>Label:</b> %{theta}<br>' +
            '<b>Count:</b> %{customdata}<br>' +
            '<b>Percentage of total:</b> %{r:.2f}%<br>'
        ),
        customdata=label_counts[['count']].to_numpy(),
        #customdata=label_counts[['count']].values,
        #customdata=label_counts[['label', 'count']].values,
    )

    fig.update_layout(
        polar_radialaxis=dict(showticklabels=False),
        font_size=16,
        legend_font_size=16,
        polar_radialaxis_ticksuffix='%',
        polar_angularaxis_rotation=90,
        width=1100,
        height=520
    )

    fig.show()


# plot_label_distribution_polar(df, 'label')

def plot_first_images(df, paths='imgPath', label_col='label', num_images_per_label=4):
    # Get unique labels
    unique_labels = df[label_col].unique()

    num_labels = len(unique_labels)
    fig, axes = plt.subplots(num_labels, num_images_per_label,
                              figsize=(2*num_images_per_label, 2*num_labels))

    if num_labels == 1:
        axes = axes.reshape(1, -1)

    for i, label in enumerate(unique_labels):
        label_images = df[df[label_col] == label]

        for j in range(min(num_images_per_label, len(label_images))):
            img_path = label_images.iloc[j][paths]
            img = Image.open(img_path)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')

            if j == 0:
                axes[i, j].set_title(label, loc='left', color='blue', fontsize= 12)

    plt.tight_layout()
    plt.show()


# Plot histograms for the red, green, and blue channels across the entire dataset
def plot_noise_distribution_by_color_channel(df):
    channels = ['red_channel', 'green_channel', 'blue_channel']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Noise Distribution by Color Channel', fontsize=16, fontweight='bold')
    sns.set_style("whitegrid")
    color_map = {
        'red_channel': '#E63946',
        'green_channel': '#2A9D8F',
        'blue_channel': '#457B9D'
    }

    for  i, (channel, color) in enumerate(zip(channels, color_map)):
        sns.histplot(data=df,
                     stat='count',
                     x=channel,
                     ax=axes[i],
                     color=color_map.get(channel, 'gray'),
                     alpha=0.6,
                     kde=True)
        axes[i].set_title(f'{channel.replace("_", " ").title()} Noise Distribution')
        axes[i].set_xlabel('Noise Value (Sigma)')
        axes[i].set_ylabel('Count')

    plt.tight_layout()
    plt.gcf()
    plt.show()


# Plot histograms for the color channels grouped by label
def plot_noise_distribution_by_label(df):
    channels = ['red_channel', 'green_channel', 'blue_channel']
    # Get unique labels
    labels = df['label'].unique()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Noise Distribution by Label', fontsize=16, fontweight='bold')
    sns.set_style("whitegrid")

    channel_palettes = [
        sns.color_palette("Reds", n_colors=len(labels)),
        sns.color_palette("Greens", n_colors=len(labels)),
        sns.color_palette("Blues", n_colors=len(labels)),
    ]

    for i, (channel, palette) in enumerate(zip(channels, channel_palettes)):
        for j, label in enumerate(labels):
            label_data = df[df['label'] == label]
            sns.histplot(data=label_data,
                         stat='count',
                         x=channel,
                         ax=axes[i],
                         label=label,
                         alpha=0.4,
                         kde=True,
                         color=palette[j])

        axes[i].set_title(f'{channel.replace("_", " ").title()} Noise Distribution by Label')
        axes[i].set_xlabel('Noise Value (Sigma)')
        axes[i].set_ylabel('Count')
        axes[i].legend()

    plt.tight_layout()
    plt.gcf()
    plt.show()


def plot_color_channel_histogram(data, figsize=(10, 6), show_extreme_values=False):
    channels=['red_channel', 'green_channel', 'blue_channel']
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")

    color_map = {
        'red_channel': '#E63946',
        'green_channel': '#2A9D8F',
        'blue_channel': '#457B9D'
    }

    for channel in channels:

        # Calculate min_val and max_val for the current channel
        min_val = data[channel].min()
        max_val = data[channel].max()

        if show_extreme_values:
            label = f"{channel.replace('_', ' ').title()} (Min: {min_val:.03g}, Max: {max_val:.03g})"
        else:
            label = channel.replace('_', ' ').title()

        sns.histplot(
            data=data[channel],
            stat='count',
            color=color_map.get(channel, 'gray'),
            alpha=0.6,
            label=label, kde= True,
        )

        sns.kdeplot(label='KDE (Frequency)')

    plt.title('Noise Distribution Across Color Channels', fontsize=16, fontweight='bold')
    plt.xlabel('Noise Value (Sigma)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Channels', loc='upper right')
    plt.tight_layout()

    plt.gcf()
    plt.show()


def plot_color_channel_avg(data, column='img_avg_noise'):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=data[column],
        stat='count',
        alpha=0.7,
        color=sns.color_palette('ch:s=-.2,r=.6', 5)[2],
        # color='#6B7280',
        label="Full Spectrum",
        kde= True,
    )

    sns.kdeplot(
        label='KDE (Frequency)'
    )

    plt.title('Average Noise Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Noise Value (Sigma)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Channel', loc='upper right')
    plt.tight_layout()

    plt.gcf()
    plt.show()


def plot_violin(dataframe, value_vars, category_col):

    # Melt the dataframe for long-format
    melted_df = pd.melt(dataframe,
                        id_vars=[category_col],
                        value_vars=value_vars,
                        var_name="Channel",
                        value_name="Intensity")

    # Color mapping
    unique_categories = melted_df[category_col].unique()
    color_map = {cat: px.colors.sequential.thermal[i % len(px.colors.sequential.thermal)]
                 for i, cat in enumerate(unique_categories)}

    fig = px.violin(
        melted_df,
        x="Channel",
        y="Intensity",
        color=category_col,
        box=True,
        points="all",
        violinmode='overlay',
        color_discrete_sequence= px.colors.sequential.Agsunset_r,
    )

    fig.update_traces(
        meanline_visible=True,
        scalegroup='Channel',
        marker=dict(outliercolor='red',size=4)
    )

    fig.update_layout(
        title='Noise Distribution by Color Channel',
        yaxis_title="Noise Value (Sigma)",
        legend_title='  Label',
        title_font_size=20,
        legend=dict(font=dict(size=12)),
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=12))
    )

    fig.update_xaxes(
        ticktext=['Red Channel', 'Green Channel', 'Blue Channel', 'Full Spectrum'],
        tickvals=["red_channel", "green_channel", "blue_channel", 'img_avg_noise']
    )

    fig.update_layout(
        title_font_size=18,
        width=1100,
        height=1000
    )

    fig.show()

# plot_violin(noise_df, value_vars=["red_channel", "green_channel", "blue_channel", 'img_avg_noise'], category_col="label")

def plot_parallel_categories(df, map_colors=False):
    if map_colors:
        label_color_map = {'basophil' : 0, 'neutrophil' : 1, 'ig' : 2, 'monocyte' : 3,
                       'eosinophil' : 4, 'erythroblast' : 5, 'lymphocyte' : 6, 'platelet' : 7}
        noise_df['label_mapping'] = noise_df['label'].map(label_color_map)

    colorscale = ['lightgray', '#00868B', 'red']

    fig = px.parallel_categories(
        df,
        dimensions=["red_channel_categorized", "green_channel_categorized",
                    "blue_channel_categorized", "img_avg_noise_categorized", "label"],
        color="label_mapping",
        color_continuous_scale=colorscale,
        labels={
            'red_channel_categorized' : 'Red Channel',
            'green_channel_categorized' : 'Green Channel',
            'blue_channel_categorized' : 'Blue Channel',
            'img_avg_noise_categorized' : 'Full Spectrum',
            'label' : 'Label',
            'label_mapping' : 'Mapping'
        }
    )

    fig.update_layout(
        title='Parallel Categories of Noise',
        coloraxis_colorbar=dict(tickvals=list(range(8)),
                                ticktext=['Basophil', 'Neutrophil', 'IG', 'Monocyte',
                                          'Eosinophil', 'Erythroblast', 'Lymphocyte', 'Platelet'],
                                tickmode='array',
                                tickfont=dict(size=10),
                                x=1.05,
                                # y=0.5,
                                xanchor="left",
                               ),
        width=1100,
        height=700,
    )

    fig.update_traces(patch={"line": {'shape':'hspline'}})
    fig.show()


def visualize_dataset_splits(train_df, val_df, test_df, label_column='label'):
    train_df['split'] = 'Train'
    val_df['split'] = 'Validation'
    test_df['split'] = 'Test'

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    label_counts = combined_df.groupby(['split', label_column]).size().reset_index(name='count')
    label_props = label_counts.groupby('split')['count'].transform(lambda x: x / x.sum())
    label_counts['proportion'] = label_props

    color_map = ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
    color_map2 = {
        'neutrophil': '#E63946',
        'eosinophil': '#2A9D8F',
        'ig': '#457B9D',
        'platelet': '#6A4C93',
        'erythroblast': '#F4A261',
        'monocyte': '#E9C46A',
        'basophil': '#264653',
        'lymphocyte': '#F77F00'
    }

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Label Counts Across Splits', 'Label Proportions Across Splits'))

    count_fig = px.bar(label_counts,
                       x='split',
                       y='count',
                       color=label_column,
                       barmode='group',
                       labels={'count': 'Label Count', 'split': 'Dataset Split'},
                       opacity=0.9,
                       color_discrete_sequence=color_map
                      )
    for trace in count_fig.data:
        fig.add_trace(trace, row=1, col=1)

    prop_fig = px.bar(label_counts,
                      x='split',
                      y='proportion',
                      color=label_column,
                      barmode='group',
                      labels={'proportion': 'Label Proportion', 'split': 'Dataset Split'},
                      opacity=0.9,
                      color_discrete_sequence=color_map
                     )
    prop_fig.update_yaxes(tickformat='.0%')

    for trace in prop_fig.data:
        trace.update(showlegend=False)

    for trace in prop_fig.data:
        fig.add_trace(trace, row=1, col=2)

    # Update layout
    fig.update_layout(
        title_text='Label Counts and Proportions Across Splits',
        title_font_size=20,
        barmode='group',
        height=600,
        width=1100,
        template='plotly_white'
    )

    fig.update_yaxes(
        title_text='Count',
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text='Proportion',
        tickformat='.0%',
        row=1,
        col=2
    )

    fig.show()


def plot_one_sample(train, val, test):
    datasets = [("Train", train), ("Validation", val), ("Test", test)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    for i, (title, dataset) in enumerate(datasets):
        image, label = dataset[0]
        image = image.permute(1, 2, 0).numpy()
        class_name = dataset.idx_to_class[label]
        axes[i].imshow(image)
        axes[i].set_title(f"{title} Dataset\nClass = '{class_name}'")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

# plot_one_sample(train_dataset, val_dataset, test_dataset)


def plot_multiple_samples(train, val, test, num_samples=8):
    datasets = [("Train", train), ("Validation", val), ("Test", test)]
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))

    for row, (title, dataset) in enumerate(datasets):
        for col in range(num_samples):
            image, label = dataset[col]
            image = image.permute(1, 2, 0).numpy()
            axes[row, col].imshow(image)
            axes[row, col].set_title(dataset.idx_to_class[label])
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(title, fontsize=12, labelpad=10)

    plt.tight_layout()
    plt.show()


def plot_model_performance(model, class_names=None, model_details=None):
    import matplotlib.patches as patches

    color_palette = {
        'train': '#4B0082',
        'val': '#FF6347',
        'precision': '#F72585',
        'recall': '#4361EE',
        'f1': '#3A0CA3'
    }
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold', color='#333333')
    fig.patch.set_facecolor('white')

    axs[0, 0].plot(model.history['train_loss'], label='Train Loss', color=color_palette['train'], linewidth=2, marker=".")
    axs[0, 0].plot(model.history['val_loss'], label='Validation Loss', color=color_palette['val'], linewidth=2, marker=".")
    axs[0, 0].set_title('Loss', fontweight='bold')
    axs[0, 0].set_xlabel('Epochs', color='#555555')
    axs[0, 0].set_ylabel('Loss', color='#555555')
    axs[0, 0].legend()

    axs[0, 1].plot(model.history['train_acc'], label='Train Accuracy', color=color_palette['train'], linewidth=2, marker=".")
    axs[0, 1].plot(model.history['val_acc'], label='Validation Accuracy', color=color_palette['val'], linewidth=2, marker=".")
    axs[0, 1].set_title('Accuracy', fontweight='bold')
    axs[0, 1].set_xlabel('Epochs', color='#555555')
    axs[0, 1].set_ylabel('Accuracy', color='#555555')
    axs[0, 1].legend()

    val_metrics = [
        ('val_precision', 'Validation Precision', color_palette['precision']),
        ('val_recall', 'Validation Recall', color_palette['recall']),
        ('val_f1_score', 'Validation F1 Score', color_palette['f1'])
    ]

    for metric, label, color in val_metrics:
        axs[1, 0].plot(model.history[metric], label=label, color=color, linewidth=2, marker=".")

    axs[1, 0].set_title('Validation Metrics', fontweight='bold')
    axs[1, 0].set_xlabel('Epochs', color='#555555')
    axs[1, 0].set_ylabel('Score', color='#555555')
    axs[1, 0].legend()

    conf_matrix = np.array(model.history['confusion_matrix'][-1])

    # Generate class names if not provided
    if class_names is None:
        class_names = [f'Class {i}' for i in range(conf_matrix.shape[0])]

    sns.heatmap(conf_matrix,
                cmap='RdPu',
                # vmin=1.56,
                # vmax=4.15,
                square=True,
                linewidth=0.3,
                # cbar_kws={'shrink': .72},
                annot_kws={'size': 12},
                annot=True,
                fmt='d',
                ax=axs[1, 1],
                xticklabels=class_names,
                yticklabels=class_names
                # cbar=False)
                )
    axs[1, 1].set_title('Confusion Matrix', fontweight='bold')
    axs[1, 1].set_xlabel('Predicted Labels', color='#555555')
    axs[1, 1].set_ylabel('True Labels', color='#555555')

    for ax in axs.flat:
        ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.tick_params(width=0.5)
        ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7)
        ax.tick_params(colors='#555555')

        # Add model details at the top
    if model_details:
        detail_text = (
            r"$\bf{Model\ Name:}$" + f" {model_details.get('model_name', 'N/A')}\n"
            r"$\bf{Loss\ Function:}$" + f" {model_details.get('loss_function', 'N/A')}\n"
            r"$\bf{Optimizer:}$" + f" {model_details.get('optimizer', 'N/A')}\n"
            r"$\bf{Accuracy\ Metric:}$" + f" {model_details.get('accuracy_metric', 'N/A')}\n"
            r"$\bf{Learning\ Rate:}$" + f" {model_details.get('learning_rate', 'N/A')}\n"
            r"$\bf{Epochs:}$" + f" {model_details.get('epochs', 'N/A')}"
        )
        # fig.text(0.5, 0.96, detail_text, ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round'))
        fig.text(
            0.02, 0.98, detail_text,
            ha='left', va='top', fontsize=8,
            color='#333333',
            # color='#555555',
            bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.2', alpha=0.9),
            usetex=False
        )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    #plt.tight_layout()
    plt.show()

    return fig


def train_model(epochs, train_loader, val_loader, model, loss_function,
                optimizer, accuracy_metric, device, num_classes, debug=False):
    """Train and validate the model, tracking metrics for both sets."""
    precision_metric = MulticlassPrecision(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    f1_score_metric = MulticlassF1Score(num_classes=num_classes, average="weighted", zero_division=0).to(device)

    if not hasattr(model, 'history'):
        model.history = {
            "train_loss": [],
            "train_acc": [],
            "train_precision": [],
            "val_loss": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1_score": []
        }

    for epoch in trange(epochs, desc="Overall Progress: Epochs", leave=True,
                        position=0, bar_format="{l_bar}{bar} | Batch {n_fmt}/{total_fmt}"):
        train_loss, train_acc, train_precision = 0, 0, 0
        accuracy_metric.reset()
        precision_metric.reset()

        model.train()
        for batch, (X, y) in enumerate(tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}: Training Phase",
                                            leave=False, position=1, bar_format="{l_bar}{bar} | Batch {n_fmt}/{total_fmt}")):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc += accuracy_metric(y_pred, y).item()
            train_precision += precision_metric(y_pred, y).item()

            if debug:
                print("Check model numeric (debug):")
                print("X shape:", X.shape)
                print("y shape:", y.shape)
                print("y unique values:", torch.unique(y))
                print(f"[batch={batch}]")
                check_model_numerics(model, X)
                print(f"Looked at {batch * len(X)}/{len(train_loader.dataset)} samples")
                print("-" * 50)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_precision /= len(train_loader)

        model.history["train_loss"].append(train_loss)
        model.history["train_acc"].append(train_acc)
        model.history["train_precision"].append(train_precision)

        model.eval()
        val_loss, val_acc, val_precision, val_recall, val_f1 = 0, 0, 0, 0, 0
        precision_metric.reset()
        recall_metric.reset()
        f1_score_metric.reset()
        accuracy_metric.reset()

        with torch.inference_mode():
            for X, y in tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch + 1}: Validation Phase",
                             leave=False, position=2, bar_format="{l_bar}{bar} | Batch {n_fmt}/{total_fmt}"):
                X, y = X.to(device), y.to(device)
                val_pred = model(X)
                val_loss += loss_function(val_pred, y).item()

                val_acc += accuracy_metric(val_pred, y).item()
                val_precision += precision_metric(val_pred, y).item()
                val_recall += recall_metric(val_pred, y).item()
                val_f1 += f1_score_metric(val_pred, y).item()

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            val_precision /= len(val_loader)
            val_recall /= len(val_loader)
            val_f1 /= len(val_loader)

            model.history["val_loss"].append(val_loss)
            model.history["val_acc"].append(val_acc)
            model.history["val_precision"].append(val_precision)
            model.history["val_recall"].append(val_recall)
            model.history["val_f1_score"].append(val_f1)

        print(f"\nEpoch {epoch + 1}/{epochs} Performance Report:")
        print(f"└─ [Train] Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f} | Precision: {train_precision:.2f}")
        print(f"└─ [Validation] Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f} | Precision: {val_precision:.2f} | Recall: {val_recall:.2f} | F1-Score: {val_f1:.2f}")
    print("Finished training and validation.")
    return model.history


def test_model(test_loader, model, loss_function, accuracy_metric, device, num_classes):
    """Evaluate the model on the test set."""
    precision_metric = MulticlassPrecision(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    f1_score_metric = MulticlassF1Score(num_classes=num_classes, average="weighted", zero_division=0).to(device)
    confusion_matrix_metric = ConfusionMatrix(num_classes=num_classes, task="multiclass").to(device)

    if not hasattr(model, 'history'):
      model.history = {
          "test_loss": [],
          "test_acc": [],
          "test_precision": [],
          "test_recall": [],
          "test_f1_score": [],
          "confusion_matrix": []
          }

    test_loss, test_acc, test_precision, test_recall, test_f1 = 0, 0, 0, 0, 0
    y_true, y_pred = [], []

    model.eval()
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_score_metric.reset()
    confusion_matrix_metric.reset()

    with torch.inference_mode():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_function(test_pred, y).item()

            test_acc += accuracy_metric(test_pred, y).item()
            test_precision += precision_metric(test_pred, y).item()
            test_recall += recall_metric(test_pred, y).item()
            test_f1 += f1_score_metric(test_pred, y).item()
            confusion_matrix_metric.update(test_pred, y)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(test_pred.argmax(dim=1).cpu().numpy())

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        test_precision /= len(test_loader)
        test_recall /= len(test_loader)
        test_f1 /= len(test_loader)

    model.history["test_loss"].append(test_loss)
    model.history["test_acc"].append(test_acc)
    model.history["test_precision"].append(test_precision)
    model.history["test_recall"].append(test_recall)
    model.history["test_f1_score"].append(test_f1)
    model.history["confusion_matrix"].append(confusion_matrix_metric.compute().cpu().numpy())
    confusion_matrix_metric.reset()
    print(f"\n[Test] Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f} | Precision: {test_precision:.2f} | Recall: {test_recall:.2f} | F1-Score: {test_f1:.2f}")
    print("Finished test evaluation.")


def plot_confusion_matrix(y_true, y_pred, class_names=None, model_details=None, figsize=(12, 8)):
    """
    Plots a confusion matrix with model details.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_names (list, optional): List of class names. Defaults to None, which will generate class names as 'Class 0', 'Class 1', etc.
        model_details (dict, optional): Dictionary with model details. Defaults to None.
        figsize (tuple, optional): Size of the figure. Defaults to (12, 8).
    """
    conf_matrix = np.array(confusion_matrix(y_true, y_pred))
    if class_names is None:
        class_names = [f'Class {i}' for i in range(conf_matrix.shape[0])]

    fig, axs = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(conf_matrix,
                cmap='RdPu',
                square=True,
                linewidth=0.3,
                annot_kws={'size': 12},
                annot=True,
                fmt='d',
                ax=axs,
                xticklabels=class_names,
                yticklabels=class_names)

    axs.set_title('Confusion Matrix', fontweight='bold')
    axs.set_xlabel('Model Prediction', color='#555555')
    axs.set_ylabel('True Labels', color='#555555')

    # Add model details
    if model_details is not None:
        detail_text = (
            r"$\bf{Model\ Name:}$" + f" {model_details.get('model_name', 'N/A')}\n"
            r"$\bf{Loss\ Function:}$" + f" {model_details.get('loss_function', 'N/A')}\n"
            r"$\bf{Optimizer:}$" + f" {model_details.get('optimizer', 'N/A')}\n"
            r"$\bf{Accuracy\ Metric:}$" + f" {model_details.get('accuracy_metric', 'N/A')}\n"
            r"$\bf{Learning\ Rate:}$" + f" {model_details.get('learning_rate', 'N/A')}\n"
            r"$\bf{Epochs:}$" + f" {model_details.get('epochs', 'N/A')}"
        )

        fig.text(
            0.02, 0.98, detail_text,
            ha='left', va='top', fontsize=8,
            color='#333333',
            bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.2', alpha=0.9)
        )

    plt.tight_layout()
    plt.show()



