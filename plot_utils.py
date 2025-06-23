## it has functions used by simulation to plot the dashboard

import numpy as np

from input.codes.config import config
from input.codes.physical_model import physical_model


def plot_rebalance(ax, agent, debug=False, mode='energy', add_title=False, add_x_label=False, ax_secondary=None):
    # TODO: rename | assign better colors | remove title and x label for most | show angle friction in RHS
    if debug:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(1, 1)
    ax.clear()
    try:
        if add_title:
            if len(agent.log.record['eat']) > 0 and len(
                    agent.log.record[
                        'energy_consumption']) > 3:  # TODO: make sure energy consumption exclude eating events
                title = (f"P out = {np.mean(agent.log.record['energy_consumption']) / config.DT:.1f} J/sec | "
                         f"Meals dt = {np.mean(np.diff([agent.birth_step] + agent.log.record['eat'])) * config.DT:.0f} sec | "
                         f"Meal E = {np.mean(agent.log.record['energy_excess']):.0f} J | "
                         f"[m, h] = {np.mean(agent.mass):.1f} Kg, {np.mean(agent.height):.0f} m")
            else:
                title = (f"P out = {np.mean(agent.log.record['energy_consumption']) / config.DT:.1f} J/sec | "
                         f"No eating events | "
                         f"[m, h] = {agent.mass:.1f} Kg, {agent.height:.1f} m")
        else:
            title = ''
    except:
        title = f"No energy logs | No eating events | [m, h] = {agent.mass:.1f} Kg, {agent.height:.1f} m"
    ax.set_title(title)
    if mode == 'speed':
        ax.plot(range(int(agent.age / config.DT)), agent.log.record['speed'], color='teal', alpha=0.25, label='Speed')
        ax.set_ylim(0, max(agent.log.record['speed']) * 1.1)
        ax.tick_params(axis='y', colors='teal')
        ax.spines['left'].set_color('maroon')
        ax.spines['right'].set_color('teal')
        ax.legend(loc='upper right')
        ax.yaxis.set_label_position("right")
        ax.set_ylabel('Speed [m/sec]')
    elif mode == 'energy':
        ax.plot(range(int(agent.age / config.DT + 1)), agent.log.record['energy'], color='maroon', alpha=0.5,
                label='Energy')
        ax.set_ylim(0, agent.max_energy)
        ax.tick_params(axis='y', colors='maroon')
        ax.spines['left'].set_color('maroon')
        ax.spines['right'].set_color('teal')
        ax.legend(loc='upper left')
        ax.set_ylabel('Energy [J]')
    elif mode == 'energy_use':
        if 'energy_consumption' in agent.log.record and len(agent.log.record['energy_consumption']) > 0:
            ax.plot(range(int(agent.age / config.DT) - len(agent.log.record['energy_propulsion']),
                          int(agent.age / config.DT)),
                    agent.log.record['energy_propulsion'], color='maroon', alpha=0.5, label='Prop. E')
            ax.plot(
                range(int(agent.age / config.DT) - len(agent.log.record['energy_inner']), int(agent.age / config.DT)),
                agent.log.record['energy_inner'], color='maroon', alpha=0.5, label='Inner E', linestyle='dashed')
            ax.tick_params(axis='y', colors='maroon')
            ax.spines['left'].set_color('maroon')
            ax.spines['right'].set_color('teal')
            ax.legend(loc='upper left')
            ax.set_ylabel('Energy consumption [J]')
    elif mode == 'force':
        if 'reaction_friction_force_mag' in agent.log.record and len(
                agent.log.record['reaction_friction_force_mag']) > 0:
            max_force = max(agent.log.record['reaction_friction_force_mag'])
            ax.plot(range(int(agent.age / config.DT) - len(agent.log.record['reaction_friction_force_mag']),
                          int(agent.age / config.DT)),
                    agent.log.record['reaction_friction_force_mag'], color='maroon', alpha=0.2, label='Friction mag')
            ax.plot(range(int(agent.age / config.DT) - len(agent.log.record['linear_drag_force']),
                          int(agent.age / config.DT)),
                    agent.log.record['linear_drag_force'], color='red', alpha=0.2, label='Lin. drag',
                    linestyle='dashed')
            ax.plot(range(int(agent.age / config.DT) - len(agent.log.record['quadratic_drag_force']),
                          int(agent.age / config.DT)),
                    agent.log.record['quadratic_drag_force'], color='olive', alpha=0.2, label='Quad. drag',
                    linestyle='dashed')
            ax.tick_params(axis='y', colors='maroon')
            ax.spines['left'].set_color('maroon')
            # ax.spines['right'].set_color('violet')
            ax.legend(loc='upper left')
            ax.set_ylabel('Force [N]')
            if ax_secondary is not None:
                ax_secondary.clear()
                ax_secondary.plot(
                    range(int(agent.age / config.DT) - len(agent.log.record['reaction_friction_force_angle']),
                          int(agent.age / config.DT)),
                    [angle * 180 / (np.pi * 2) for angle in agent.log.record['reaction_friction_force_angle']],
                    color='violet', alpha=0.5, label='Friction angle')
                ax_secondary.set_ylim(max(agent.log.record['reaction_friction_force_angle']) * 180 / (np.pi * 2))
                ax_secondary.set_ylabel('friction [degrees]')
            # else:
            #     ax.plot(range(int(agent.age / config.DT) - len(agent.log.record['reaction_friction_force_angle']),
            #               int(agent.age / config.DT)),
            #         [angle * max_force / (np.pi * 2) for angle in agent.log.record['reaction_friction_force_angle']],
            #         color='blueviolet', alpha=0.5, label='Friction angle')
    elif mode == 'friction angle':
        if 'reaction_friction_force_angle' in agent.log.record:
            ax.plot(range(int(agent.age / config.DT) - len(agent.log.record['reaction_friction_force_angle']),
                          int(agent.age / config.DT)),
                    [angle / (np.pi) for angle in agent.log.record['reaction_friction_force_angle']],
                    color='blueviolet', alpha=0.5, label='Friction angle')
            ax.tick_params(axis='y', colors='blueviolet')
            ax.spines['left'].set_color('blueviolet')
            # ax.spines['right'].set_color('violet')
            ax.set_ylabel('angle [rad/pi]')
    elif mode == 'power':
        if 'energy_consumption' in agent.log.record and len(
                agent.log.record['energy_consumption']) > 0:
            if 'energy_excess' in agent.log.record and len(agent.log.record['energy_excess']) > 0:
                ax.plot(range(int(agent.age / config.DT) - len(agent.log.record['energy_excess']),
                              int(agent.age / config.DT)),
                        np.sum(agent.log.record['energy_excess']) / (agent.age / config.DT), color='coral', alpha=0.5,
                        label='Power gain')
            else:
                ax.plot(range(int(agent.age / config.DT) - len(agent.log.record['energy_excess']),
                              int(agent.age / config.DT)),
                        np.sum(agent.log.record['energy_excess']) / (agent.age / config.DT), color='coral',
                        alpha=0.5,
                        label='Power gain')
            ax.plot(range(int(agent.age / config.DT) - len(agent.log.record['energy_consumption']),
                          int(agent.age / config.DT)),
                    np.sum(agent.log.record['energy_consumption']) / (agent.age / config.DT), color='lawngreen',
                    alpha=0.5,
                    label='Power use')
            ax.tick_params(axis='y', colors='brown')
            ax.spines['left'].set_color('brown')
            ax.legend(loc='upper left')
            ax.set_ylabel('Power [Watt]')

    if add_x_label:
        ax.set_xlabel('Age [step]')
    else:
        ax.set_xticks([])


def plot_live_status(ax, agent, debug=False, plot_horizontal=True):
    """
    Plots the agent's status (energy, hunger, thirst) on the given axes.
    """
    if debug:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(1, 1)
    # Define attributes dynamically
    ls = ['energy', 'age']  # , 'hunger', 'thirst'
    colors = ['green', 'grey']  # , 'red', 'blue'
    values = [getattr(agent, attr) for attr in ls]  # Dynamically get values
    ax.clear()
    if plot_horizontal:
        ax.barh(ls, values, color=colors, height=0.2)
        if 'energy' in ls:
            ax.scatter([config.REPRODUCTION_ENERGY + config.MIN_LIFE_ENERGY], ['energy'], color='black', s=20)
        if 'age' in ls:
            ax.scatter([agent.max_age], ['age'], color='black', s=20)
            ax.scatter([agent.adolescence], ['age'], color='pink', s=20)
            # ax.barh(['Energy', 'Hunger', 'Thirst'], [agent.energy, agent.hunger, agent.thirst], color=['green', 'red', 'blue'])
            ax.set_xlim(0, max(agent.max_energy, agent.max_age))
            # ax.set_xticks([0,agent.max_energy/2, agent.max_energy])
            ax.set_yticks(ls)
    else:
        ax.bar(ls, values, color=colors, width=0.5)
        if 'energy' in ls:
            ax.scatter(['energy'], [config.REPRODUCTION_ENERGY + config.MIN_LIFE_ENERGY], color='black', s=20)
        if 'age' in ls:
            ax.scatter(['age'], [agent.max_age], color='black', s=20)
            ax.scatter(['age'], [agent.adolescence], color='pink', s=20)
            ax.set_ylim(0, max(agent.max_energy, agent.max_age))
            ax.set_xticks(ls)
            ax.set_xticklabels(ls, rotation=90, ha='right')
            ax.set_yticks([])
            ax.yaxis.set_tick_params(labelleft=False)


def plot_live_status_power(ax, agent, debug=False, plot_horizontal=True):
    ax.clear()
    if 'energy_consumption' in agent.log.record and len(
            agent.log.record['energy_consumption']) > 0:
        P_used = sum(agent.log.record['energy_consumption']) / (len(agent.log.record['energy_consumption']) * config.DT)
        if 'energy_excess' in agent.log.record and len(agent.log.record['energy_excess']) > 0:
            P_gain = sum(agent.log.record['energy_excess']) / (len(agent.log.record['energy_consumption']) * config.DT)
        else:
            P_gain = 0
        colors = ['lawngreen', 'coral']
        if plot_horizontal:
            ax.barh(['Power gain', 'Power use'], [P_gain, P_used], color=colors, height=0.2)
        else:
            ax.bar(['Power gain', 'Power use'], [P_gain, P_used], color=colors, width=0.5)


def plot_acc_status(ax, agent, debug=False, plot_type=1, curr_step=-1):
    """
    Plots the agent's accumulated status (logs) on the given axes.
    """
    if debug:
        print('debug_mode')
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(1, 1)
    # Define attributes dynamically
    ls = ['eat', 'reproduce']
    colors = ['green', 'pink']
    ax.clear()
    if max(agent.color) > 1 or min(agent.color) < 0:
        raise ('color exceed [0, 1] range')
    ax.set_facecolor(list(agent.color) + [0.3])
    ax.set_title(f'C# {agent.creature_id} | Anc. = {len(agent.ancestors)}')
    if plot_type == 0:
        # option 1
        values = [len(agent.log.record[attr]) for attr in ls]  # Dynamically get values
        ax.set_title(f'Agent # {agent.id}')
        ax.bar(ls, values, color=colors, width=0.2)
        ax.set_ylim(0, 10)
        ax.set_yticks([0, 5, 10, 100])
        ax.set_xticks(ls)
    if plot_type == 1:
        # option 2
        if curr_step == -1: curr_step = agent.max_age / config.DT + agent.birth_step
        # values = [getattr(agent, attr) for attr in ls]  # Dynamically get values
        eating_frames = agent.log.record['eat']
        reproducing_frames = agent.log.record['reproduce']
        ax.scatter(eating_frames, [1] * len(eating_frames), color='green', marker='o', s=100, label='Eating')
        ax.scatter(reproducing_frames, [2] * len(reproducing_frames), color='red', marker='D', s=100,
                   label='Reproducing')
        ax.set_yticks([1, 2])
        # ax.set_yticklabels(['Eating', 'Reproducing'])
        # Label x-axis and add a title
        ax.set_xlabel('Frame Number')
        # ax.set_title('Event Timeline')
        ax.set_xlim([agent.birth_step - 1, curr_step + 1])
        ax.set_ylim([0.5, 2.5])
        ax.legend()
