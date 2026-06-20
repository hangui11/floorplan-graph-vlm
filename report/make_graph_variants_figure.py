"""
Generate the 'two graph variants' figure for the Background chapter, at the
FLOOR level used by this thesis: nodes are apartments plus the shared vertical
circulation (staircase, elevator). The same building floor is encoded as a
(dense) adjacency graph and the (sparse) access graph the pipeline extracts.

Run from the report/ directory:
    python make_graph_variants_figure.py
Produces: figures/graph_variants.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# A building floor: four perimeter apartments around a central core
# (staircase + elevator). Fixed positions so the two panels are comparable.
pos = {
    "Apt A":     (-1.3,  1.0),
    "Apt B":     (1.3,  1.0),
    "Apt C":     (-1.3, -1.0),
    "Apt D":     (1.3, -1.0),
    "Stair":     (-0.35, 0.0),
    "Elevator":  (0.35,  0.0),
}

COLORS = {
    "apartment": "#4A90D9",
    "Stair":     "#E74C3C",
    "Elevator":  "#F39C12",
}


def node_color(n):
    return COLORS.get(n, COLORS["apartment"])


def node_shape(n):
    return {"Stair": "s", "Elevator": "D"}.get(n, "o")


# Adjacency graph: an edge wherever two units/elements SHARE A WALL (dense) --
# neighbouring apartments touch, and each apartment abuts the core.
adjacency_edges = [
    ("Apt A", "Apt B"), ("Apt C", "Apt D"),
    ("Apt A", "Apt C"), ("Apt B", "Apt D"),
    ("Apt A", "Stair"), ("Apt B", "Elevator"),
    ("Apt C", "Stair"), ("Apt D", "Elevator"),
    ("Stair", "Elevator"),
]

# Access graph: edges only where you can actually move through a door/landing
# (sparse) -- hub-and-spoke: every apartment reaches the shared core, the core
# elements connect to each other. Apartments do NOT connect directly.
access_edges = [
    ("Apt A", "Stair"), ("Apt B", "Stair"),
    ("Apt C", "Stair"), ("Apt D", "Stair"),
    ("Stair", "Elevator"),
]


def draw(ax, edges, title, edge_color):
    G = nx.Graph()
    G.add_nodes_from(pos.keys())
    G.add_edges_from(edges)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=2.0, alpha=0.8)
    # Draw nodes per-type so apartments/stair/elevator get distinct shapes.
    for n in G.nodes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=[n],
            node_color=node_color(n), node_shape=node_shape(n),
            node_size=2200, edgecolors="black", linewidths=1.2,
        )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="white",
                            font_weight="bold")
    ax.set_title(f"{title}\n({G.number_of_edges()} edges)", fontsize=12)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.8, 1.8)
    ax.axis("off")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
draw(ax1, adjacency_edges, "Adjacency graph (dense)", "#9B9B9B")
draw(ax2, access_edges, "Access graph (sparse)", "#E74C3C")

# Shared legend for node types.
from matplotlib.lines import Line2D
legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["apartment"],
           markeredgecolor="black", markersize=11, label="Apartment"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["Stair"],
           markeredgecolor="black", markersize=11, label="Staircase"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["Elevator"],
           markeredgecolor="black", markersize=11, label="Elevator"),
]
fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.04, 1, 1))

out = "figures/graph_variants.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
