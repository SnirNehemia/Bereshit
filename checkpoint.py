import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from input.codes.config import load_config
from input.codes.physical_model import load_physical_model

# Load config
config_yaml_relative_path = r"input\yamls\2025_04_18_config.yaml"
config = load_config(yaml_relative_path=config_yaml_relative_path)

# Load physical model
physical_model_yaml_relative_path = r"input\yamls\2025_04_18_physical_model.yaml"
physical_model = load_physical_model(yaml_relative_path=physical_model_yaml_relative_path)

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class ParametricDashboard:
    def __init__(self, f_list, x_attr_list, init_struct_list, param_limits, x_labels, y_labels, layout=None,
                 slider_marks=None, slider_names=None, func_colors=None, func_legends=None,
                 shared_slider_labels=None):
        self.f_list = f_list
        self.init_struct_list = init_struct_list  # list of tuples or lists (agent_dict, extra_args_dict) or just args
        self.param_limits = param_limits           # list of lists of (min, max) tuples
        self.x_labels = x_labels
        self.y_labels = y_labels
        self.num_funcs = len(f_list)
        self.x_attr_list = x_attr_list
        self.sliders = []
        self.slider_refs = {}
        self.plots = []
        self.axes = []
        self.layout = layout if layout else (self.num_funcs, 1)
        self.slider_marks = slider_marks
        self.slider_names = slider_names
        self.func_colors = func_colors
        self.func_legends = func_legends
        self.shared_slider_labels = shared_slider_labels if shared_slider_labels else {}
        self.build_dashboard()

    def build_dashboard(self):
        fig_width = 6 * self.layout[1] + 2.5
        fig_height = 3.5 * self.layout[0]
        self.fig, axs = plt.subplots(*self.layout, figsize=(fig_width, fig_height))
        axs = np.array(axs).reshape(-1)

        plt.subplots_adjust(left=0.3, right=0.95)

        for i in range(self.num_funcs):
            ax = axs[i]
            funcs = self.f_list[i] if isinstance(self.f_list[i], (list, tuple)) else [self.f_list[i]]
            colors = self.func_colors[i] if self.func_colors and i < len(self.func_colors) else [None] * len(funcs)
            labels = self.func_legends[i] if self.func_legends and i < len(self.func_legends) else [None] * len(funcs)
            lines = []

            args = self.init_struct_list[i]
            if isinstance(args, (tuple, list)) and isinstance(args[0], dict):
                agent, extra = args
                x_vals = self._x_from_struct(agent, self.x_attr_list[i])
                for f, color, label in zip(funcs, colors, labels):
                    y = f(agent, **extra)
                    line, = ax.plot(x_vals, y, label=label, color=color)
                    lines.append(line)
            else:
                x_index = self.x_attr_list[i]
                if not isinstance(x_index, int):
                    print(f"Warning: x_attr_list[{i}] was not an int. Falling back to index 0.")
                    x_index = 0
                x_vals = np.asarray(args[x_index])
                for f, color, label in zip(funcs, colors, labels):
                    y = f(*args)
                    line, = ax.plot(x_vals, y, label=label, color=color)
                    lines.append(line)

            ax.set_xlabel(self.x_labels[i], labelpad=10)
            ax.set_ylabel(self.y_labels[i], labelpad=10)
            ax.grid(True)
            if any(labels):
                ax.legend()
            self.plots.append((lines, funcs, self.x_attr_list[i]))
            self.axes.append(ax)

        slider_top = 0.95
        slider_height = 0.02
        slider_gap = 0.025

        for i, args in enumerate(self.init_struct_list):
            slider_group = []
            if isinstance(args, (tuple, list)) and isinstance(args[0], dict):
                agent, extra = args
                keys = list(agent.keys()) + list(extra.keys())
                values = {**agent, **extra}
            else:
                keys = [self.slider_names[i][j] if self.slider_names and i < len(self.slider_names) and j < len(self.slider_names[i]) and self.slider_names[i][j] else f'param{j}' for j in range(len(args))]
                values = {k: v for k, v in zip(keys, args)}

            for j, (key, lim) in enumerate(zip(keys, self.param_limits[i])):

                if key == self.x_attr_list[i]:
                    continue
                val = values[key]
                if isinstance(val, (list, np.ndarray)):
                    continue  # skip array-valued keys (e.g. x values)
                if lim[0] is None or lim[1] is None:
                    continue  # skip if slider limits are invalid
                label = self.slider_names[i][j] if self.slider_names and self.slider_names[i][j] else key
                if label in self.slider_refs:
                    slider = self.slider_refs[label]
                    slider_group.append(slider)
                    continue
                slider_bottom = slider_top - slider_height
                ax_slider = self.fig.add_axes([0.05, slider_bottom, 0.15, slider_height])
                ax_slider.set_title(label, fontsize=7, pad=4)
                ax_slider.set_frame_on(False)
                ax_slider.get_xaxis().set_visible(False)
                ax_slider.get_yaxis().set_visible(False)


                slider = Slider(ax_slider, '', lim[0], lim[1], valinit=val)
                slider.on_changed(self.update)
                if self.slider_marks and self.slider_marks[i][j]:
                    for mark in self.slider_marks[i][j]:
                        ax_slider.axvline(x=mark, color='black', linestyle='-', alpha=0.3)
                slider_group.append(slider)
                self.slider_refs[label] = slider
                slider_top -= (slider_height + slider_gap)
            self.sliders.append(slider_group)

        plt.show()

    def update(self, val):
        for i, (lines, funcs, x_attr) in enumerate(self.plots):
            args = self.init_struct_list[i]
            if isinstance(args, (tuple, list)) and isinstance(args[0], dict):
                agent, extra = args
                new_agent = agent.copy()
                new_extra = extra.copy()
                for label, slider in self.slider_refs.items():
                    if label in new_agent:
                        new_agent[label] = slider.val
                    elif label in new_extra:
                        new_extra[label] = slider.val
                x_vals = self._x_from_struct(new_agent, x_attr)
                for line, f in zip(lines, funcs):
                    y = f(new_agent, **new_extra)
                    line.set_ydata(y)
                    line.set_xdata(x_vals)
            else:
                keys = [self.slider_names[i][j] if self.slider_names and i < len(self.slider_names) and j < len(self.slider_names[i]) and self.slider_names[i][j] else f'param{j}' for j in range(len(args))]
                new_args = [self.slider_refs[k].val if k in self.slider_refs else v for k, v in zip(keys, args)]
                x_vals = np.asarray(new_args[x_attr])
                for line, f in zip(lines, funcs):
                    y = f(*new_args)
                    line.set_ydata(y)
                    line.set_xdata(x_vals)

            self.axes[i].relim()
            self.axes[i].autoscale_view()
        self.fig.canvas.draw_idle()

    def _x_from_struct(self, struct, attr):
        val = struct[attr]
        if isinstance(val, (int, float)):
            return np.array([val])
        return np.asarray(val)

    def _x_from_struct(self, struct, attr):
        val = struct[attr]
        if isinstance(val, (int, float)):
            return np.array([val])
        return np.asarray(val)


# ------------------ TODO:The physical model - it should be in a seperate module and imported here ------------------

def linear_drag(speed, height, gamma=0):
    # args[0] = height, args[1] = speed
    linear_drag_force = - gamma * height ** 2 * speed
    # linear_drag_force = - physical_model.gamma * agent.height * agent.speed
    return linear_drag_force

def quadratic_drag(speed, height, c_drag=0):
    # args[0] = height, args[1] = speed
    # quadratic_drag_force = - physical_model.c_drag * agent.height * agent.speed ** 2
    quadratic_drag_force = - c_drag * height ** 2 * speed ** 2
    return quadratic_drag_force

def friction(mass, g=physical_model.g, mu_static=physical_model.mu_static):
    # args[0] = height, args[1] = speed
    # friction_force = - physical_model.mu_kinetic * agent.mass * physical_model.g
    friction_force = - mu_static * mass * g
    return friction_force

# -------------------------------- Energy Calculations --------------------------------

def calc_propulsion_energy(propulsion_force, eta = physical_model.energy_conversion_factors['activity_efficiency'],
                           c_heat = physical_model.energy_conversion_factors['heat_loss']):
    # eta = physical_model.energy_conversion_factors['activity_efficiency']
    # c_heat = physical_model.energy_conversion_factors['heat_loss']
    propulsion_energy = (1 / eta + c_heat) * propulsion_force
    return propulsion_energy


# def calc_inner_energy(mass, height, brain_size=10, digest_dict = config.INIT_DIGEST_DICT,  c_d = physical_model.energy_conversion_factors['digest'],
#                       c_h = physical_model.energy_conversion_factors['height'],
#                       rest_factor = physical_model.energy_conversion_factors['rest'],
#                       brain_consumption = physical_model.energy_conversion_factors['brain_consumption']):
#     # c_d = physical_model.energy_conversion_factors['digest']
#     # c_h = physical_model.energy_conversion_factors['height']
#     # rest_energy = physical_model.energy_conversion_factors['rest'] * self.mass ** 0.75  # adds mass (BMR) energy
#     rest_energy = rest_factor * mass ** 0.75  # adds mass (BMR) energy
#     inner_energy = rest_energy + c_d * digest_dict + c_h * height  # adds height energy
#     inner_energy = inner_energy + brain_size * brain_consumption
#     return inner_energy

# def calc_inner_energy(agent, c_d = physical_model.energy_conversion_factors['digest'],
#                   c_h = physical_model.energy_conversion_factors['height'],
#                   rest_factor = physical_model.energy_conversion_factors['rest'],
#                   brain_consumption = physical_model.energy_conversion_factors['brain_consumption']):
#     c_d = physical_model.energy_conversion_factors['digest']
#     c_h = physical_model.energy_conversion_factors['height']
#     rest_energy = physical_model.energy_conversion_factors['rest'] * agent.mass ** 0.75  # adds mass (BMR) energy
#     inner_energy = rest_energy + c_d * np.sum(list(agent.digest_dict.values())) + c_h * agent.height  # adds height energy
#     inner_energy = inner_energy + agent.brain.size * physical_model.energy_conversion_factors['brain_consumption']
#     return inner_energy

def calc_inner_energy(agent, physical_model):
    c_d = physical_model.energy_conversion_factors['digest']
    c_h = physical_model.energy_conversion_factors['height']
    rest_energy = physical_model.energy_conversion_factors['rest'] * agent.mass ** 0.75  # adds mass (BMR) energy
    inner_energy = rest_energy + c_d * np.sum(list(agent.digest_dict.values())) + c_h * agent.height  # adds height energy
    inner_energy = inner_energy + agent.brain.size * physical_model.energy_conversion_factors['brain_consumption']
    return inner_energy

def calc_trait_energy(trait_type, gained_energy, age):
    trait_energy_params = physical_model.trait_energy_params_dict[trait_type]
    factor = trait_energy_params['factor']
    rate = trait_energy_params['rate']
    trait_energy_func = physical_model.trait_energy_func(factor=factor, rate=rate, age=age)
    trait_energy = trait_energy_func * gained_energy
    return trait_energy


# Example usage
if __name__ == '__main__':
    # import numpy as np
    #
    #
    # def parabola(x, a, b):
    #     return a * x ** 2 + b
    #
    #
    # x_vals = np.linspace(-10, 10, 400)
    # a_init = 1.0
    # b_init = 0.0
    #
    # dashboard = ParametricDashboard(
    #     f_list=[parabola],
    #     x_attr_list=[0],  # x is the first argument in the function
    #     init_struct_list=[(x_vals, a_init, b_init)],
    #     param_limits=[[(None, None), (-5, 5), (-10, 10)]],  # dummy for x, limits for a and b
    #     x_labels=['x'],
    #     y_labels=['y'],
    #     slider_names=[['param0', 'a', 'b']]  # can name the x param arbitrarily; it will be skipped
    # )

    def f1(x, a): return np.sin(a * x)
    def f2(x, b, c): return np.exp(-b * x) * np.cos(c * x)
    # Define the functions and initial arguments
    f_list = []
    x_attr_list = []
    init_args_list = []
    param_limits = []
    x_labels = []
    y_labels = []
    slider_marks = []
    slider_names = []
    func_colors = []
    func_legends = []

    # ---------------------------------- test --------------------------------
    # # Sine wave
    #
    # f_list.append(f2)
    # x_list.append(np.linspace(0, 5, 300))
    # init_args_list.append((0.5, 2))
    # param_limits.append([(0.1, 2), (0.5, 10)])
    # x_labels.append('Time (s)')
    # y_labels.append('Damped Oscillation')
    # slider_marks.append([(0.1, 2), (0.5, 10)])
    # slider_names.append(['Amplitude', 'Frequency'])
    # func_colors.append(['black'])
    # func_legends.append(['Sine Wave'])
    # x_labels = ['Angle (rad)', 'Time (s)']

    # x_vals = np.linspace(0, 10, 500)
    # b_init = 0.5
    # c_init = 3.0
    #
    # ParametricDashboard(
    #     f_list=[f2],
    #     x_attr_list=[0],  # x is at index 0 in the arguments
    #     init_struct_list=[(x_vals, b_init, c_init)],
    #     param_limits=[[(None, None), (0.1, 2.0), (1.0, 10.0)]],  # skip x, define limits for b, c
    #     x_labels=["Time (s)"],
    #     y_labels=["Amplitude"],
    #     slider_names=[['x', 'b', 'c']]
    # )
    # ---------------------------------- Forces --------------------------------

    # Drag forces -     example for parameteric dashboard with several funcs
    f_list.append([])
    f_list[-1].append(lambda x, h, gamma, c_drag: linear_drag(x, h, gamma))
    f_list[-1].append(lambda x, h, gamma, c_drag: quadratic_drag(x, h, c_drag))
    f_list[-1].append(lambda x, h, gamma, c_drag: quadratic_drag(x, h, c_drag) + linear_drag(x, h, gamma))
    x_attr_list.append(0)
    # x_list.append(np.linspace(0, config.MAX_SPEED, 101))
    init_args_list.append((np.linspace(0, config.MAX_SPEED, 101),
                            config.INIT_MAX_HEIGHT * 0.1,
                           physical_model.c_drag,
                           physical_model.gamma))
    param_limits.append([(None, None),
                         (config.INIT_MAX_HEIGHT * 0.01, config.INIT_MAX_HEIGHT),
                         (physical_model.gamma * 0.1, physical_model.gamma * 10),
                         (physical_model.c_drag * 0.1, physical_model.c_drag * 10)])
    x_labels.append('velocity [m/s]')
    y_labels.append('Drag Force [N]')
    slider_marks.append([(), (config.INIT_MAX_HEIGHT * 0.1, config.INIT_MAX_HEIGHT*0.5),(),()])
    slider_names.append(['None', 'Height [m]', 'Gamma [N/(m/s)]', 'C_drag [N/(m/sec)^2]'])
    func_colors.append(['blueviolet', 'violet', 'black'])
    func_legends.append(['Linear Drag', 'Quadratic Drag', 'Linear + Quadratic Drag'])

    # # Friction forces -     example for parameteric dashboard with a single funcs
    f_list.append(lambda m, g, mu_static: friction(m, g, mu_static))
    x_attr_list.append(0)
    # x_list.append(np.linspace(config.INIT_MAX_MASS*0.01, config.INIT_MAX_MASS*10, 101))
    init_args_list.append([np.linspace(config.INIT_MAX_MASS*0.01, config.INIT_MAX_MASS*10, 101),
                           physical_model.g,
                           physical_model.mu_static])
    param_limits.append([(None, None),
                         (physical_model.g*0.1, physical_model.g*10),
                         (physical_model.mu_static*0.1, physical_model.mu_static*10)])
    x_labels.append('Mass [kg]')
    y_labels.append('Friction Force [N]')
    slider_marks.append([(),(physical_model.g*0.1, physical_model.g*0.5), (physical_model.mu_static*0.1, physical_model.mu_static*0.5)])
    slider_names.append(['None','g [m/s^2]', 'mu_static'])
    func_colors.append(['black'])
    func_legends.append(['Friction Force'])

    # ---------------------------------- Energy --------------------------------

    # Energy consumption
    f_list.append([])
    f_list[-1].append(lambda agent, g, mu_static, h, gamma: calc_propulsion_energy(friction(agent.mass, g, mu_static), h, gamma))
    f_list[-1].append(calc_inner_energy)
    f_list[-1].append(lambda x, h, gamma, c_drag: quadratic_drag(x, h, c_drag) + linear_drag(x, h, gamma))
    x_list.append(np.linspace(config.INIT_MAX_MASS * 0.01, config.INIT_MAX_MASS * 10, 101))
    init_args_list.append([physical_model.g, physical_model.mu_static])


    x_list.append(np.linspace(0, config.MAX_SPEED, 101))
    init_args_list.append((config.INIT_MAX_HEIGHT * 0.1, physical_model.c_drag, physical_model.gamma))
    param_limits.append([(config.INIT_MAX_HEIGHT * 0.01, config.INIT_MAX_HEIGHT),
                         (physical_model.gamma * 0.1, physical_model.gamma * 10),
                         (physical_model.c_drag * 0.1, physical_model.c_drag * 10)])
    x_labels.append('velocity [m/s]')
    y_labels.append('Drag Force [N]')
    slider_marks.append([(config.INIT_MAX_HEIGHT * 0.1, config.INIT_MAX_HEIGHT*0.5),(),()])
    slider_names.append(['Height [m]', 'Gamma [N/(m/s)]', 'C_drag [N/(m/sec)^2]'])
    func_colors.append(['blueviolet', 'violet', 'black'])
    func_legends.append(['Linear Drag', 'Quadratic Drag', 'Linear + Quadratic Drag'])

    # # Sine wave
    #
    # f_list.append(f2)
    # x_list.append(np.linspace(0, 5, 300))
    # init_args_list.append((0.5, 2))
    # param_limits.append([(0.1, 2), (0.5, 10)])
    # x_labels.append('Time (s)')
    # y_labels.append('Damped Oscillation')
    # slider_marks.append([(0.1, 2), (0.5, 10)])
    # slider_names.append(['Amplitude', 'Frequency'])
    # func_colors.append(['black'])
    # func_legends.append(['Sine Wave'])
    # # x_labels = ['Angle (rad)', 'Time (s)']


    dashboard = ParametricDashboard(
        f_list=f_list,
        x_attr_list=x_attr_list,
        init_struct_list=init_args_list,
        param_limits=param_limits,
        x_labels=x_labels,
        y_labels=y_labels,
        slider_marks=slider_marks,
        slider_names=slider_names,
        func_colors=func_colors,
        func_legends=func_legends,
        layout=(2, 2)
    )
    plt.show()



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

    for j, (f, color, func_label, style) in enumerate(zip(f_list, colors, func_labels, styles)):
        for i, args in enumerate(args_list):
            y = f(x, *args)
            alpha = base_alpha + i * alpha_step
            label = f'{func_label}, {args_name}={args}'
            ax.plot(x, y, alpha=alpha, label=label, color=color, linestyle=style)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)

    ax.legend(title="Function and Args", fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)

    if ax is None:
        plt.tight_layout()
        plt.show()
