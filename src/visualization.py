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

