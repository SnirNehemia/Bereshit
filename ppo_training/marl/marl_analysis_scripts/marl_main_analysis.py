from ppo_training.marl.marl_analysis_scripts.marl_analyze_entropy import analyze_exploration_decay
from ppo_training.marl.marl_analysis_scripts.marl_analyze_weights import analyze_synaptic_attention
from ppo_training.marl.marl_analysis_scripts.marl_evaluate_actor_video import render_ecosystem
from ppo_training.marl.marl_analysis_scripts.marl_value_map_video import create_evolution_movie
from ppo_training.marl.marl_analysis_scripts.probe_decision_making import run_probes

if __name__ == '__main__':
    species_list = ['herb', 'carn']
    results_folder = 'updates500_ent005_ex_energy_hp_reward_2'
    update_milestone = 500

    # Training process
    for species in species_list:
        analyze_exploration_decay(results_folder=results_folder, species=species)  # analyze entropy
        analyze_synaptic_attention(results_folder=results_folder, species=species)  # analyze weights

    # Value map video
    value_types = ['Food', 'Creature']
    a_types = [-1.0, 1.0]  # -1.0 (herbivore), 1.0 (carnivore)
    a_masses = [-1.0, 1.0]  # -1.0 (tiny), 1.0 (huge)

    for value_type in value_types:
        if value_type == 'Food':
            create_evolution_movie(results_folder=results_folder, value_type=value_type)
        else:
            for a_type in a_types:
                for a_mass in a_masses:
                    create_evolution_movie(results_folder=results_folder,
                                           value_type=value_type, a_type=a_type, a_mass=a_mass)

    # Actor evaluation
    render_ecosystem(results_folder=results_folder, update_milestone=update_milestone)
    run_probes(results_folder=results_folder, update_milestone=update_milestone)
