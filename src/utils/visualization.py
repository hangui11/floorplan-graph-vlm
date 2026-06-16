"""
Visualization utilities for aggregation graphs.

Renders aggregation graphs showing apartments, staircases, and elevators
as color-coded nodes, with side-by-side comparison to the original plan.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx

from src.config import NODE_COLORS


def build_networkx_graph(graph_data: dict) -> nx.Graph:
    """Convert an aggregation graph dict to a NetworkX Graph."""
    G = nx.Graph()

    for node in graph_data["nodes"]:
        label_parts = [node["type"].upper()]
        if node.get("label"):
            label_parts.insert(0, str(node["label"]))
        display_label = "\n".join(label_parts)

        G.add_node(
            node["node_id"],
            node_type=node["type"],
            label=node.get("label", ""),
            position=node.get("position"),
            display_label=display_label,
        )

    for edge in graph_data["edges"]:
        G.add_edge(edge["from_node_id"], edge["to_node_id"])

    return G


def draw_graph(
    graph_data: dict,
    save_path: str | Path | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
):
    """Draw an aggregation graph with color-coded node types."""
    G = build_networkx_graph(graph_data)

    if len(G.nodes) == 0:
        return

    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Use node positions from VLM if available, otherwise spring layout
    pos = {}
    has_positions = all(
        G.nodes[n].get("position") is not None for n in G.nodes
    )
    if has_positions:
        for n in G.nodes:
            p = G.nodes[n]["position"]
            # VLM gives [x%, y%] — flip y so (0,0) is top-left matching image
            pos[n] = (p[0], 100 - p[1])
    else:
        pos = nx.spring_layout(G, seed=42, k=2.5)

    # Node colors and sizes by type
    node_colors = []
    node_sizes = []
    for n in G.nodes:
        ntype = G.nodes[n].get("node_type", "apartment")
        node_colors.append(NODE_COLORS.get(ntype, "#BDC3C7"))
        if ntype == "apartment":
            node_sizes.append(2000)
        else:
            node_sizes.append(1400)

    labels = {n: G.nodes[n].get("display_label", str(n)) for n in G.nodes}

    # Node shapes: apartments are circles, stairs/elevators are squares
    # NetworkX doesn't support per-node shapes, so draw them in groups
    apt_nodes = [n for n in G.nodes if G.nodes[n].get("node_type") == "apartment"]
    stair_nodes = [n for n in G.nodes if G.nodes[n].get("node_type") == "staircase"]
    elev_nodes = [n for n in G.nodes if G.nodes[n].get("node_type") == "elevator"]

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6, width=2)

    # Draw apartment nodes (circles)
    if apt_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=apt_nodes,
            node_color=[NODE_COLORS["apartment"]] * len(apt_nodes),
            node_size=[2000] * len(apt_nodes),
            node_shape="o", ax=ax, alpha=0.9, edgecolors="black", linewidths=1.5,
        )

    # Draw staircase nodes (squares)
    if stair_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=stair_nodes,
            node_color=[NODE_COLORS["staircase"]] * len(stair_nodes),
            node_size=[1400] * len(stair_nodes),
            node_shape="s", ax=ax, alpha=0.9, edgecolors="black", linewidths=1.5,
        )

    # Draw elevator nodes (diamonds)
    if elev_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=elev_nodes,
            node_color=[NODE_COLORS["elevator"]] * len(elev_nodes),
            node_size=[1400] * len(elev_nodes),
            node_shape="D", ax=ax, alpha=0.9, edgecolors="black", linewidths=1.5,
        )

    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold", ax=ax)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLORS["apartment"],
               markersize=12, label='Apartment'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=NODE_COLORS["staircase"],
               markersize=12, label='Staircase'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=NODE_COLORS["elevator"],
               markersize=12, label='Elevator'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    if title:
        ax.set_title(title, fontsize=12)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.close()
    elif show:
        plt.show()


def draw_side_by_side(
    image_path: str | Path,
    graph_data: dict,
    save_path: str | Path | None = None,
    title: str | None = None,
):
    """Draw original floor plan image and extracted aggregation graph side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Left: original image
    img = mpimg.imread(str(image_path))
    ax1.imshow(img)
    ax1.set_title("Original Floor Plan", fontsize=11)
    ax1.axis("off")

    # Right: extracted graph
    draw_graph(graph_data, ax=ax2, title="Aggregation Graph")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def draw_3d_graph(
    building: dict,
    save_path: str | Path | None = None,
    title: str | None = None,
    floor_spacing: float = 100.0,
):
    """
    Draw a 3D building graph with each floor stacked vertically and vertical
    edges connecting staircase/elevator nodes across floors.

    Each floor is rendered as a translucent plane at its own height so the
    levels read as clearly separated storeys. The box aspect is set so floors
    are evenly spaced instead of being squished by matplotlib's auto-scaling.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 11))
    ax = fig.add_subplot(111, projection="3d")

    nodes_by_id = {n["node_id"]: n for n in building["nodes"]}
    n_floors = building.get("n_floors", 1)

    # 3D coordinates per node. z = floor index * spacing.
    coords = {}
    for n in building["nodes"]:
        floor = n.get("floor", 0)
        pos = n.get("position") or [50, 50]
        try:
            x, y = float(pos[0]), float(pos[1])
        except (TypeError, ValueError, IndexError):
            x, y = 50.0, 50.0
        # Flip y so (0,0) reads as top-left, matching the source image.
        coords[n["node_id"]] = (x, 100 - y, floor * floor_spacing)

    # Draw a translucent slab + outline for each floor so storeys are distinct.
    for f in range(n_floors):
        z = f * floor_spacing
        corners = [(0, 0, z), (100, 0, z), (100, 100, z), (0, 100, z)]
        slab = Poly3DCollection([corners], alpha=0.06, facecolor="steelblue",
                                edgecolor="gray", linewidths=0.6)
        ax.add_collection3d(slab)
        # Floor label anchored at a back corner of its own slab.
        ax.text(2, 100, z + floor_spacing * 0.04,
                f"Floor {f}", fontsize=10, fontweight="bold",
                color="darkslategray")

    # Draw edges
    for e in building["edges"]:
        a = coords.get(e["from_node_id"])
        b = coords.get(e["to_node_id"])
        if a is None or b is None:
            continue
        if e["edge_type"] == "vertical":
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    color="black", linewidth=2.0, alpha=0.9)
        else:
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    color="gray", linewidth=1.2, alpha=0.5)

    # Draw nodes grouped by type for legend
    for ntype, marker, size in (
        ("apartment", "o", 90),
        ("staircase", "s", 80),
        ("elevator",  "D", 80),
    ):
        xs, ys, zs = [], [], []
        for nid, n in nodes_by_id.items():
            if n["type"] == ntype:
                c = coords[nid]
                xs.append(c[0]); ys.append(c[1]); zs.append(c[2])
        if xs:
            ax.scatter(
                xs, ys, zs,
                c=NODE_COLORS.get(ntype, "#BDC3C7"),
                marker=marker, s=size,
                edgecolors="black", linewidths=0.8,
                label=ntype.capitalize(),
                depthshade=False,
            )

    ax.set_xlabel("X (% image width)")
    ax.set_ylabel("Y (% image height)")
    ax.set_zlabel("Floor height")
    # Hide numeric z ticks (floor labels carry the meaning) and place ticks at storeys.
    ax.set_zticks([f * floor_spacing for f in range(n_floors)])
    ax.set_zticklabels([str(f) for f in range(n_floors)])

    # Even, readable proportions: keep x/y square and give z room per floor so
    # storeys do not bunch up. A viewing elevation that shows the stack clearly.
    ax.set_box_aspect((1, 1, max(0.6, 0.5 * n_floors)))
    ax.view_init(elev=18, azim=-60)

    ax.legend(loc="upper left", fontsize=9)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
