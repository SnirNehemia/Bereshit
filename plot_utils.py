
## it has functions used by simulation to plot the dashboard

import numpy as np

from input.codes.config import load_config
from input.codes.physical_model import load_physical_model

# Load config
config_yaml_relative_path = r"input\yamls\2025_04_18_config.yaml"
config = load_config(yaml_relative_path=config_yaml_relative_path)

# Load physical model
physical_model_yaml_relative_path = r"input\yamls\2025_04_18_physical_model.yaml"
physical_model = load_physical_model(yaml_relative_path=physical_model_yaml_relative_path)

def plot_rebalance(ax, agent, debug=False, mode='energy', add_title=False, add_x_label=False, ax_secondary=None):
    # TODO: rename | assiggn better colors | remove title and x label for most | show angle friction in RHS
    if debug:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(1, 1)
    ax.clear()
    try:
        if add_title:
            if len(agent.log.record['eat']) > 0 and len(
                    agent.log.record['energy_consumption']) > 3:  # TODO: make sure energy consumption exclude eating events
                title = (f"P out = {np.mean(agent.log.record['energy_consumption']) / config.DT:.1f} J/sec | "
                         f"Meals dt = {np.mean(np.diff([agent.birth_step] + agent.log.record['eat'])) *
                                       config.DT:.0f} sec | "
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
        ax.plot(range(int(agent.age / config.DT + 1)), agent.log.record['energy'], color='maroon', alpha=0.5, label='Energy')
        ax.set_ylim(0, agent.max_energy)
        ax.tick_params(axis='y', colors='maroon')
        ax.spines['left'].set_color('maroon')
        ax.spines['right'].set_color('teal')
        ax.legend(loc='upper left')
        ax.set_ylabel('Energy [J]')
    elif mode == 'energy_use':
        if 'energy_consumption' in agent.log.record and len(agent.log.record['energy_consumption']) > 0:
            ax.plot(range(int(agent.age / config.DT) - len(agent.log.record['energy_propulsion']), int(agent.age / config.DT)),
                    agent.log.record['energy_propulsion'], color='maroon', alpha=0.5, label='Prop. E')
            ax.plot(range(int(agent.age / config.DT) - len(agent.log.record['energy_inner']), int(agent.age / config.DT)),
                    agent.log.record['energy_inner'], color='maroon', alpha=0.5, label='Inner E', linestyle='dashed')
            ax.tick_params(axis='y', colors='maroon')
            ax.spines['left'].set_color('maroon')
            ax.spines['right'].set_color('teal')
            ax.legend(loc='upper left')
            ax.set_ylabel('Energy consumption [J]')
    elif mode == 'force':
        if 'reaction_friction_force_mag' in agent.log.record and len(agent.log.record['reaction_friction_force_mag']) > 0:
            max_force = max(agent.log.record['reaction_friction_force_mag'])
            ax.plot(range(int(agent.age / config.DT) - len(agent.log.record['reaction_friction_force_mag']),
                          int(agent.age / config.DT)),
                    agent.log.record['reaction_friction_force_mag'], color='maroon', alpha=0.5, label='Friction mag')
            ax.plot(range(int(agent.age / config.DT) - len(agent.log.record['linear_drag_force']),
                          int(agent.age / config.DT)),
                    agent.log.record['linear_drag_force'], color='red', alpha=0.5, label='Lin. drag',
                    linestyle='dashed')
            ax.plot(range(int(agent.age / config.DT) - len(agent.log.record['quadratic_drag_force']),
                          int(agent.age / config.DT)),
                    agent.log.record['quadratic_drag_force'], color='olive', alpha=0.5, label='Quad. drag',
                    linestyle='dashed')
            ax.tick_params(axis='y', colors='maroon')
            ax.spines['left'].set_color('maroon')
            # ax.spines['right'].set_color('violet')
            ax.legend(loc='upper left')
            ax.set_ylabel('Force [N]')
            if ax_secondary is not None:
                ax_secondary.clear()
                ax_secondary.plot(range(int(agent.age / config.DT) - len(agent.log.record['reaction_friction_force_angle']),
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
                        np.sum(agent.log.record['energy_excess'])/(agent.age / config.DT), color='coral', alpha=0.5,
                        label='Power gain')
            else:
                ax.plot(range(int(agent.age / config.DT) - len(agent.log.record['energy_excess']),
                              int(agent.age / config.DT)),
                        np.sum(agent.log.record['energy_excess']) / (agent.age / config.DT), color='coral',
                        alpha=0.5,
                        label='Power gain')
            ax.plot(range(int(agent.age / config.DT) - len(agent.log.record['energy_consumption']),
                          int(agent.age / config.DT)),
                    np.sum(agent.log.record['energy_consumption']) / (agent.age / config.DT), color='lawngreen', alpha=0.5,
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
            ax.scatter([agent.adolescence],['age'], color='pink', s=20)
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


"""" 
--------------------------------------------------------------------------------------------------------------
------------------------------------------ balance plots -----------------------------------------------------
--------------------------------------------------------------------------------------------------------------
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

def plot_parametric_variation(f_list, x, args_list, x_label='x', y_label='f(x)', title=None, colors=None,
                              func_labels=None, styles=None, args_name='Args', ax=None):
    """
    Plots multiple functions over the same x and args_list, with varying alpha for args and color for function.

    Parameters:
    - f_list: list of functions of form f(x, *args) -> y
    - x: vector of x values
    - args_list: list of argument tuples to be unpacked into f
    - x_label: label for x-axis
    - y_label: label for y-axis
    - title: optional title for the plot
    - colors: list of colors corresponding to each function
    - func_labels: list of labels for each function (for legend)
    - styles: list of line styles corresponding to each function (e.g. '-', '--', ':', '-.')
    - args_name: name for the varying argument (used in legend)
    - ax: optional matplotlib Axes object to plot into
    """
    if len(f_list) != len(colors) or len(f_list) != len(func_labels) or len(f_list) != len(styles):
        raise ValueError('f_list, colors, func_labels, and styles must have the same length')

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    base_alpha = 0.1
    alpha_step = (1.0 - base_alpha) / max(len(args_list) - 1, 1)

    if colors is None:
        colors = ['black'] * len(f_list)
    if func_labels is None:
        func_labels = [f'f{i}' for i in range(len(f_list))]
    if styles is None:
        styles = ['-'] * len(f_list)

    if len(f_list) == 1:
        for j, (f, color, func_label, style) in enumerate(zip(f_list, colors, func_labels, styles)):
            for i, args in enumerate(args_list):
                y = f(x, args)
                alpha = 1
                label = f'{func_label}, {args_name}={args}'
                ax.plot(x, y, alpha=alpha, color=color, linestyle=style)
    else:
        for j, (f, color, func_label, style) in enumerate(zip(f_list, colors, func_labels, styles)):
            for i, args in enumerate(args_list):
                y = f(x, args)
                alpha = base_alpha + i * alpha_step
                label = f'{func_label}, {args_name}={args}'
                ax.plot(x, y, alpha=alpha, label=label, color=color, linestyle=style)
                ax.legend(title="Function and Args", fontsize=6)

    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)
    if ax is None:
        plt.tight_layout()
        plt.show()

# -------------------------------- Force Calculations --------------------------------

def linear_drag(speed, height):
    # args[0] = height, args[1] = speed
    linear_drag_force = - physical_model.gamma * height ** 2 * speed
    # linear_drag_force = - physical_model.gamma * agent.height * agent.speed
    return linear_drag_force

def quadratic_drag(speed, height):
    # args[0] = height, args[1] = speed
    # quadratic_drag_force = - physical_model.c_drag * agent.height * agent.speed ** 2
    quadratic_drag_force = - physical_model.c_drag * height ** 2 * speed ** 2
    return quadratic_drag_force

def friction(mass, x=0):
    # args[0] = height, args[1] = speed
    # friction_force = - physical_model.mu_kinetic * agent.mass * physical_model.g
    friction_force = - physical_model.mu_kinetic * mass * physical_model.g
    return friction_force

# -------------------------------- Energy Calculations --------------------------------

def calc_propulsion_energy(propulsion_force, x=0):
    eta = physical_model.energy_conversion_factors['activity_efficiency']
    c_heat = physical_model.energy_conversion_factors['heat_loss']
    propulsion_energy = (1 / eta + c_heat) * propulsion_force
    return propulsion_energy


def calc_inner_energy(self):
    c_d = physical_model.energy_conversion_factors['digest']
    c_h = physical_model.energy_conversion_factors['height']
    rest_energy = physical_model.energy_conversion_factors['rest'] * self.mass ** 0.75  # adds mass (BMR) energy
    inner_energy = rest_energy + c_d * np.sum(list(self.digest_dict.values())) + c_h * self.height  # adds height energy
    inner_energy = inner_energy + self.brain.size * physical_model.energy_conversion_factors['brain_consumption']
    return inner_energy


def calc_trait_energy(trait_type, gained_energy, age):
    trait_energy_params = physical_model.trait_energy_params_dict[trait_type]
    factor = trait_energy_params['factor']
    rate = trait_energy_params['rate']
    trait_energy_func = physical_model.trait_energy_func(factor=factor, rate=rate, age=age)
    trait_energy = trait_energy_func * gained_energy
    return trait_energy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def create_dashboard(f_list, x_list, init_args_list, param_limits, x_labels, y_labels):
    """
    Creates a dashboard with sliders for each parameter and subplots for each function.

    Parameters:
    - f_list: list of functions, each of form f(x, *params)
    - x_list: list of x arrays for each function
    - init_args_list: list of initial parameter tuples
    - param_limits: list of tuples (min, max) for each parameter (list of lists per function)
    - x_labels: list of x-axis labels
    - y_labels: list of y-axis labels
    """
    num_funcs = len(f_list)
    fig, axs = plt.subplots(num_funcs, 1, figsize=(8, 3 * num_funcs), squeeze=False)
    sliders = []

    # Initial plot
    plots = []
    for i in range(num_funcs):
        ax = axs[i, 0]
        y = f_list[i](x_list[i], *init_args_list[i])
        line, = ax.plot(x_list[i], y, label='Initial')
        ax.set_xlabel(x_labels[i])
        ax.set_ylabel(y_labels[i])
        ax.grid(True)
        plots.append((line, f_list[i], x_list[i], init_args_list[i]))

    # Add sliders for parameters under all plots
    slider_axes = []
    slider_y = 0.05
    for i, args in enumerate(init_args_list):
        sliders.append([])
        for j, (val, lim) in enumerate(zip(args, param_limits[i])):
            ax_slider = fig.add_axes([0.25, slider_y, 0.65, 0.03])
            slider = Slider(ax_slider, f'Func {i} Param {j}', lim[0], lim[1], valinit=val)
            sliders[i].append(slider)
            slider_y += 0.04

    # Update function
    def update(val):
        for i, (line, f, x, _) in enumerate(plots):
            current_args = tuple(slider.val for slider in sliders[i])
            y = f(x, *current_args)
            line.set_ydata(y)
        fig.canvas.draw_idle()

    # Connect sliders
    for slider_group in sliders:
        for slider in slider_group:
            slider.on_changed(update)

    plt.tight_layout(rect=[0, slider_y, 1, 1])
    plt.show()


if __name__ == '__main__':
    def f1(x, a): return np.sin(a * x)


    def f2(x, b, c): return np.exp(-b * x) * np.cos(c * x)


    create_dashboard(
        f_list=[f1, f2],
        x_list=[np.linspace(0, 2 * np.pi, 200), np.linspace(0, 5, 200)],
        init_args_list=[(1,), (0.5, 3)],
        param_limits=[[(0.1, 5)], [(0.1, 2), (1, 10)]],
        x_labels=['x (rad)', 'time (s)'],
        y_labels=['sin(a·x)', 'damped oscillation']
    )

    # # ---------------------------------------- Force graphs -----------------------------------------------
    # fig, axs = plt.subplots(2,1,figsize=(8, 5))
    # # Drag forces
    # height = [config.INIT_MAX_HEIGHT * 0.01, config.INIT_MAX_HEIGHT * 0.1, config.INIT_MAX_HEIGHT * 0.5]
    # x = np.linspace(0, config.MAX_SPEED, 301)
    # plot_parametric_variation([linear_drag,
    #                            quadratic_drag,
    #                            lambda x, h: (quadratic_drag(x, h) + linear_drag(x, h))],
    #                           x, height, x_label='Speed [m/s]',
    #                           y_label='Force [N]',
    #                           title='Quad Drag vs Speed for multiple heights',
    #                           args_name='Height [m]',
    #                           colors=['teal', 'violet', 'black'], func_labels=['Linear Drag', 'Quadratic Drag', 'Sum'],
    #                           styles=['--', '--', '-'],
    #                           ax=axs[0])
    # # # Friction force
    # dummy = [0]
    # x = np.linspace(config.INIT_MAX_MASS*0.8, config.INIT_MAX_MASS*1, 101)
    # plot_parametric_variation([friction],
    #                           x, dummy, x_label='mass [Kg]',
    #                           y_label='Force [N]',
    #                           title='friction vs mass',
    #                           args_name='',
    #                           colors=['black'], func_labels=['Friction'],
    #                           styles=['-'],
    #                           ax=axs[1])
    # plt.tight_layout()
    # # plt.show()
    #
    # # ---------------------------------------- Energy graphs -----------------------------------------------
    # fig, axs = plt.subplots(2, 1, figsize=(8, 5))
    # # Propulsion energy
    # dummy = [0]
    # x = np.linspace(config.INIT_MAX_MASS*0.8, config.INIT_MAX_MASS*1, 101)
    # x = friction(x)
    # plot_parametric_variation([calc_propulsion_energy],
    #                           x, dummy, x_label='Force [N]',
    #                           y_label='Energy [J]',
    #                           title='Energy vs Force',
    #                           args_name='',
    #                           colors=['black'], func_labels=['Propulsion Energy'],
    #                           styles=['-'],
    #                           ax=axs[0])
    #
