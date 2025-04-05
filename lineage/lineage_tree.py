import random

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class Creature:
    def __init__(self, id, gen, parent_id, birth_step, color,
                 mass, height, strength, speed, energy):
        self.id = id
        self.gen = gen
        self.parent_id = parent_id
        self.birth_step = birth_step  # Renamed from birth_frame
        self.color = color

        # New attributes
        self.mass = mass
        self.height = height
        self.strength = strength
        self.speed = speed
        self.energy = energy

    def __repr__(self):
        return (f"Creature(id={self.id}, gen={self.gen}, parent={self.parent_id}, "
                f"step={self.birth_step}, mass={self.mass:.2f}, height={self.height:.2f}, "
                f"str={self.strength:.2f}, spd={self.speed:.2f}, eng={self.energy:.2f})")


def plot_lineage_tree(creatures: dict[str, Creature], start_from_step: int | None = None):
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np

    # Build full graph
    full_graph = nx.DiGraph()
    for creature_id, creature in creatures.items():
        full_graph.add_node(creature_id, gen=creature.gen)
        if creature.parent_id is not None:
            full_graph.add_edge(creature.parent_id, creature_id, birth_step=creature.birth_step)

    if start_from_step is not None:
        # Get all creatures born at or after the step
        starting_nodes = [cid for cid, c in creatures.items() if c.birth_step >= start_from_step]
        # Get all their descendants
        nodes_to_include = set()
        for node in starting_nodes:
            nodes_to_include |= nx.descendants(full_graph, node)
            nodes_to_include.add(node)

        G = full_graph.subgraph(nodes_to_include).copy()
    else:
        G = full_graph

    if len(G.nodes) == 0:
        print(f"No creatures born from step {start_from_step}.")
        return

    generations = nx.get_node_attributes(G, "gen")

    # Manual layout
    pos = {}
    generation_positions = {}
    for node, gen in generations.items():
        generation_positions.setdefault(gen, []).append(node)

    for gen, nodes in generation_positions.items():
        x_positions = np.linspace(-len(nodes) // 2, len(nodes) // 2, len(nodes))
        for i, node in enumerate(nodes):
            pos[node] = (x_positions[i], -gen)

    node_colors = [creatures[node].color for node in G.nodes]
    edge_labels = {(u, v): f"Step {d['birth_step']}" for u, v, d in G.edges(data=True)}

    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color=node_colors,
            edge_color="black", font_size=10, font_weight="bold", arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
    plt.title(f"Creature Lineage Tree (from step {start_from_step})")
    plt.show()


def generate_creature_tree(num_roots=2, max_children=3, max_generations=4, mutation_strength=0.1):
    creatures = {}
    next_id = 0

    def get_id():
        nonlocal next_id
        cid = str(next_id)
        next_id += 1
        return cid

    def mutate_value(value, strength=0.1, min_val=0.1, max_val=10):
        return float(np.clip(value + np.random.normal(scale=strength), min_val, max_val))

    def mutate_color(color, mutation_strength=0.1):
        # Mutate each RGB component with a Gaussian distribution
        mutated = np.clip(np.array(color) + np.random.normal(scale=mutation_strength, size=3), 0, 1)
        return tuple(mutated)

    # Start with root creatures
    current_generation = []
    for _ in range(num_roots):
        cid = get_id()
        color = tuple(np.random.rand(3))
        creatures[cid] = Creature(
            id=cid,
            gen=0,
            parent_id=None,
            birth_step=0,
            color=color,
            mass=np.random.uniform(1, 5),
            height=np.random.uniform(0.5, 2),
            strength=np.random.uniform(1, 10),
            speed=np.random.uniform(1, 10),
            energy=np.random.uniform(10, 100)
        )
        current_generation.append(cid)

    for gen in range(1, max_generations):
        next_generation = []
        for parent_id in current_generation:
            num_children = random.randint(0, max_children)
            parent = creatures[parent_id]
            for _ in range(num_children):
                cid = get_id()
                color = mutate_color(parent.color, mutation_strength)
                birth_step = parent.birth_step + random.randint(5, 20)

                creatures[cid] = Creature(
                    id=cid,
                    gen=gen,
                    parent_id=parent_id,
                    birth_step=birth_step,
                    color=color,
                    mass=mutate_value(parent.mass),
                    height=mutate_value(parent.height),
                    strength=mutate_value(parent.strength),
                    speed=mutate_value(parent.speed),
                    energy=mutate_value(parent.energy, strength=5, min_val=0, max_val=200)
                )
                next_generation.append(cid)
        current_generation = next_generation
        if not current_generation:
            break

    return creatures


if __name__ == '__main__':
    creatures = generate_creature_tree(num_roots=20, max_children=3, max_generations=10, mutation_strength=0.1)

    max_birth_step = max([creature.birth_step for creature in creatures.values()])
    print(f'{max_birth_step=}')

    start_from_step = max_birth_step - 20

    count = 0
    for creature in creatures.values():
        if creature.birth_step >= start_from_step:
            print(creature)
            count += 1
    print(f'{count=}')

    plot_lineage_tree(creatures=creatures, start_from_step=start_from_step)
