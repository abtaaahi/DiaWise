import plotly.graph_objects as go

# Define workflow steps and stage colors
nodes = [
    ("Dataset Preparation", "Data"),
    ("Embedding", "Model"),
    ("Model Training", "Model"),
    ("Evaluation", "Evaluation"),
    ("Explainability", "Evaluation"),
    ("Streamlit Deployment", "Deployment")
]

stage_colors = {
    "Data": "lightblue",
    "Model": "lightgreen",
    "Evaluation": "orange",
    "Deployment": "violet"
}

# Assign manual positions (left-to-right)
node_positions = {node[0]: (i*2, 0) for i, node in enumerate(nodes)}

# Define edges
edges = [(nodes[i][0], nodes[i+1][0]) for i in range(len(nodes)-1)]

# Edge coordinates
edge_x = []
edge_y = []
for start, end in edges:
    x0, y0 = node_positions[start]
    x1, y1 = node_positions[end]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=2, color='gray'),
    mode='lines+markers',
    marker=dict(size=5, color='gray'),
    hoverinfo='none'
)

# Node coordinates and colors
node_x = []
node_y = []
node_colors = []
node_text = []

for node_name, stage in nodes:
    x, y = node_positions[node_name]
    node_x.append(x)
    node_y.append(y)
    node_colors.append(stage_colors[stage])
    node_text.append(f"{node_name}")

node_trace = go.Scatter(
    x=node_x, y=node_y,
    text=node_text,
    textposition="bottom center",
    mode='markers+text',
    hoverinfo='text',
    marker=dict(
        color=node_colors,
        size=50,
        line=dict(color='black', width=2)
    )
)

# Create figure with arrow annotations
fig = go.Figure(data=[edge_trace, node_trace])

for start, end in edges:
    x0, y0 = node_positions[start]
    x1, y1 = node_positions[end]
    fig.add_annotation(
        x=x1, y=y1,
        ax=x0, ay=y0,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor='gray'
    )

# Layout
fig.update_layout(
    title='',
    showlegend=False,
    hovermode='closest',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    margin=dict(l=20, r=20, t=50, b=20)
)

fig.show()
