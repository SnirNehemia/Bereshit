import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from creature import Creature


def plot_lineage_tree(creatures: dict[int, Creature]):
    # build graph
    G = nx.DiGraph()

    for creature_id, creature in creatures.items():
        G.add_node(creature_id, gen=creature.gen)
        if creature.parent_id is not None:
            G.add_edge(creature.parent_id, creature_id, birth_frame=creature.birth_frame)  # Store birth frame

    # Get generation information for all nodes
    generations = nx.get_node_attributes(G, "gen")

    # Create a manual positioning for nodes: vertical positioning based on generation
    pos = {}
    generation_positions = {}  # Dictionary to keep track of node positions at each generation

    # Generate positions based on generations
    for node, gen in generations.items():
        if gen not in generation_positions:
            generation_positions[gen] = []
        generation_positions[gen].append(node)

    # Set positions vertically (Y = generation level) and horizontally (X = position within the generation)
    for gen, nodes in generation_positions.items():
        # Evenly space nodes horizontally by generating a range of x values
        x_positions = np.linspace(-len(nodes) // 2, len(nodes) // 2, len(nodes))  # Spread nodes horizontally
        for i, node in enumerate(nodes):
            pos[node] = (x_positions[i], -gen)  # X is the horizontal position, Y is the generation level

    # Color nodes by generation
    unique_gens = sorted(set(generations.values()))
    colormap = cm.Set2(np.linspace(0, 1, len(unique_gens)))  # Gradient colors
    gen_color_map = {gen: colormap[i] for i, gen in enumerate(unique_gens)}
    node_colors = [gen_color_map[generations[node]] for node in G.nodes]

    # Extract birth frame labels for edges
    edge_labels = {(u, v): f"Frame {d['birth_frame']}" for u, v, d in G.edges(data=True)}

    # Plot the graph
    plt.figure(figsize=(8, 6))

    # Draw the graph with horizontal alignment of nodes
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color=node_colors,
            edge_color="black", font_size=10, font_weight="bold", arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

    plt.title("Creature Lineage Tree (Color-Coded, Frame Labels)")
    plt.show()


if __name__ == '__main__':
    class Creature:
        def __init__(self, gen, parent_id, birth_frame):
            self.gen = gen
            self.parent_id = parent_id
            self.birth_frame = birth_frame  # Frame when born


    # Example data
    creatures = {
        "0": Creature(gen=0, parent_id=None, birth_frame=0),
        "1": Creature(gen=0, parent_id=None, birth_frame=0),
        "2": Creature(gen=1, parent_id="0", birth_frame=10),
        "3": Creature(gen=1, parent_id="0", birth_frame=10),
        "4": Creature(gen=1, parent_id="0", birth_frame=10),
        "5": Creature(gen=1, parent_id="1", birth_frame=12),
        "6": Creature(gen=2, parent_id="1", birth_frame=20),
        "7": Creature(gen=2, parent_id="2", birth_frame=22),
        "8": Creature(gen=3, parent_id="2", birth_frame=30),
        "9": Creature(gen=3, parent_id="2", birth_frame=30),
        "10": Creature(gen=3, parent_id="2", birth_frame=30),
        "11": Creature(gen=3, parent_id="2", birth_frame=30),
        "12": Creature(gen=3, parent_id="2", birth_frame=30),
    }

    # Build and plot the lineage tree
    # all_creatures = {**creatures, **dead_creatures}
    plot_lineage_tree(creatures)
