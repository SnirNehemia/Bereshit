"""
Temporal ecology: run simulation with drought and seasons, then analyze results.
Run from code/ directory: python -m directions.main_temporal_ecology
"""
import sys
from pathlib import Path

# Ensure code/ (parent of directions/) is in path for imports
_code_root = Path(__file__).resolve().parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless analysis
import matplotlib.pyplot as plt
import numpy as np

from b_basic.sim_config import sim_config
from c_models.physical_model_factory import PhysicalModelFactory
from directions.simulation import Simulation
from e_logs.statistics_logs import StatisticsLogs


def run_simulation():
    """Load config, run simulation with temporal ecology (drought, seasons)."""
    directions_folder = Path(__file__).resolve().parent
    import matplotlib
    matplotlib.use('Agg')  # headless-friendly; must be before load_config triggers TkAgg
    sim_config.load_config(
        config_name="config_temporal_ecology.yaml",
        folder_full_path=directions_folder,
    )
    PhysicalModelFactory.discover_models()

    sim = Simulation()
    total_time = sim.run_and_visualize()
    print(f"\nSimulation completed in {total_time:.2f} seconds")
    return sim_config.config.timestamp, sim_config.config.STATISTICS_LOGS_JSON_FILEPATH


def run_analyses(timestamp: str, statistics_json_path: Path, output_dir: Path):
    """
    Load statistics and plot population, grass abundance, and herbivore/carnivore vs step.
    Overlay seasonal phase for temporal ecology context.
    """
    statistics_logs = StatisticsLogs.from_json(filepath=statistics_json_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = np.arange(len(statistics_logs.num_creatures_per_step))
    season_length = getattr(sim_config.config, 'SEASON_LENGTH', 5000)
    season_phase = (steps % season_length) / season_length if season_length > 0 else np.zeros_like(steps)

    # Figure 1: Population and grass dynamics
    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes1[0].plot(steps, statistics_logs.num_creatures_per_step, 'b-', label='alive')
    axes1[0].plot(steps, statistics_logs.num_new_creatures_per_step, 'g-', alpha=0.7, label='new')
    axes1[0].plot(steps, statistics_logs.num_dead_creatures_per_step, 'r-', alpha=0.7, label='dead')
    axes1[0].set_ylabel('Creatures')
    axes1[0].set_title(f'Temporal Ecology: {timestamp}\nPopulation dynamics')
    axes1[0].legend(loc='upper right', fontsize=8)
    axes1[0].grid(True, alpha=0.3)

    axes1[1].plot(steps, statistics_logs.num_grass_history, 'g-', label='grass')
    axes1[1].plot(steps, statistics_logs.num_leaves_history, 'darkgreen', alpha=0.8, label='leaves')
    axes1[1].set_ylabel('Food points')
    axes1[1].set_title('Grass and leaves abundance')
    axes1[1].legend(loc='upper right', fontsize=8)
    axes1[1].grid(True, alpha=0.3)

    axes1[2].plot(steps, statistics_logs.num_herbivores_per_step, 'g-', label='herbivores')
    axes1[2].plot(steps, statistics_logs.num_carnivores_per_step, 'r-', label='carnivores')
    axes1[2].set_xlabel('Step')
    axes1[2].set_ylabel('Count')
    axes1[2].set_title('Herbivores vs carnivores')
    axes1[2].legend(loc='upper right', fontsize=8)
    axes1[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig1.savefig(output_dir / f'{timestamp}_temporal_ecology_population_grass.png', dpi=150)
    plt.close(fig1)
    print(f"Saved: {output_dir / f'{timestamp}_temporal_ecology_population_grass.png'}")

    # Figure 2: Seasonal phase overlay
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.fill_between(steps, 0, 1, where=(season_phase < 0.5), alpha=0.15, color='orange', label='Type A active')
    ax2.fill_between(steps, 0, 1, where=(season_phase >= 0.5), alpha=0.15, color='blue', label='Type B active')
    ax2.plot(steps, statistics_logs.num_grass_history, 'g-', linewidth=1, label='grass')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Grass count')
    ax2.set_title(f'Temporal Ecology: Grass abundance with seasonal phase overlay\n{timestamp}')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig2.savefig(output_dir / f'{timestamp}_temporal_ecology_season_overlay.png', dpi=150)
    plt.close(fig2)
    print(f"Saved: {output_dir / f'{timestamp}_temporal_ecology_season_overlay.png'}")


def main():
    print("=== Temporal Ecology: Simulation and Analysis ===\n")

    timestamp, stats_path = run_simulation()

    project_folder = sim_config.config.project_folder
    if project_folder and Path(project_folder).exists():
        analysis_output_dir = Path(project_folder) / "outputs" / "temporal_ecology" / timestamp
    else:
        analysis_output_dir = Path(stats_path).parent

    print(f"\nRunning analyses (output: {analysis_output_dir})...")
    run_analyses(
        timestamp=timestamp,
        statistics_json_path=stats_path,
        output_dir=analysis_output_dir,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
