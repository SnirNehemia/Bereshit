from ppo_training.marl.marl_analysis_scripts.marl_analyze_entropy import analyze_exploration_decay
from ppo_training.marl.marl_analysis_scripts.marl_analyze_value_map import plot_critic_value_map_food, \
    plot_critic_value_map_creature
from ppo_training.marl.marl_analysis_scripts.marl_analyze_weights import analyze_synaptic_attention
from ppo_training.marl.marl_analysis_scripts.marl_evaluation import render_ecosystem
from ppo_training.marl.marl_analysis_scripts.probe_decision_making import run_probes

if __name__ == '__main__':
    species_list = ['herb', 'carn']
    results_folder = 'marl_results_500_ent005_ex_eat_dist_hp_reward'
    update_milestone = 500

    # Analyze
    for species in species_list:
        analyze_exploration_decay(results_folder=results_folder, species=species)  # analyze entropy
        plot_critic_value_map_food(results_folder=results_folder, species=species)  # analyze value map (see food)
        plot_critic_value_map_creature(results_folder=results_folder, species=species)  # analyze value map (see creature)
        analyze_synaptic_attention(results_folder=results_folder, species=species)  # analyze weights

    # Evaluate
    render_ecosystem(results_folder=results_folder, update_milestone=update_milestone)
    run_probes(results_folder=results_folder, update_milestone=update_milestone)
