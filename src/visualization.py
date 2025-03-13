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


visualize_dimensions(df, 'dimensions')
