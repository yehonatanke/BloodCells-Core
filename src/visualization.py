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
