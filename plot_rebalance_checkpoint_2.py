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

    def _flatten_object(self, obj, parent_key='', sep='.'):
        items = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.update(self._flatten_object(v, new_key, sep=sep))
        elif hasattr(obj, '__dict__'):
            for k, v in vars(obj).items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.update(self._flatten_object(v, new_key, sep=sep))
        else:
            items[parent_key] = obj
        return items

    def _set_nested_attr(self, obj, attr_path, value):
        parts = attr_path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def _get_nested_attr(self, obj, attr_path):
        for part in attr_path.split('.'):
            obj = getattr(obj, part)
        return obj

    def f_vector(self, f, attr, x_vector, agent, extra):
        f_vector = [
            f(self._set_nested_attr(agent, attr, x) or agent, extra)
            for x in x_vector
        ]
        return f_vector

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
            if isinstance(args, (tuple, list)): # and isinstance(args[0], dict)
                agent, extra = args
                self.x_vector = {}
                for j, (name, lim) in enumerate(zip(self.slider_names[i], self.param_limits[i])):
                    if name == self.x_attr_list[i]:  # TODO: modify 'j' to WORK!!!
                        k = self.slider_names[i].index(name)
                        self.x_vector[name] = np.linspace(self.param_limits[i][k][0], self.param_limits[i][k][1], 101)
                        continue
                try:
                    x_vals = self._x_from_struct(agent, self.x_attr_list[i])
                    for f, color, label in zip(funcs, colors, labels):
                        # x_vector = self._get_nested_attr(agent, self.x_attr_list[i])
                        x_vector = self.x_vector[self.x_attr_list[i]]
                        if isinstance(x_vector, (np.ndarray, list)):
                            y = np.array(self.f_vector(f, self.x_attr_list[i], x_vector, agent, extra))
                            line, = ax.plot(x_vector, y, label=label, color=color)
                        else:
                            y = f(agent, extra)
                            line, = ax.plot(x_vals, y, label=label, color=color)
                        lines.append(line)
                        continue
                except AttributeError:
                    pass

            else:
                x_index = self.x_attr_list[i]
                if not isinstance(x_index, int):
                    print(f"Warning: x_attr_list[{i}] was not an int. Falling back to index 0.")
                    x_index = 0
                x_vals = np.asarray(args[x_index])
                for f, color, label in zip(funcs, colors, labels):
                    y = np.array([f(agent, **extra) for agent in args[0]]) if isinstance(args[0], (list, np.ndarray)) else f(*args)
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
        slider_height = 0.045
        slider_gap = 0.02
        # self.x_vector = {}
        # TODO: make sure we take physical properties into account, for each key (in physical or agent) check which one it fits.
        for i, args in enumerate(self.init_struct_list):
            slider_group = []
            if isinstance(args, (tuple, list)) and not isinstance(args[0], (int, float, str)):
                agent, extra = args
                agent_flat = self._flatten_object(agent)
                extra_flat = self._flatten_object(extra)
                keys = [self.slider_names[i][j] if self.slider_names and i < len(self.slider_names) and j < len(
                    self.slider_names[i]) and self.slider_names[i][j] else f'param{j}' for j in range(len(args))]
                # keys = list(agent_flat.keys()) + list(extra_flat.keys())
                values = {**agent_flat, **extra_flat}
            else:
                keys = [self.slider_names[i][j] if self.slider_names and i < len(self.slider_names) and j < len(self.slider_names[i]) and self.slider_names[i][j] else f'param{j}' for j in range(len(args))]
                values = {k: v for k, v in zip(keys, args)}

            for j, (key, lim) in enumerate(zip(keys, self.param_limits[i])):
                if key == self.x_attr_list[i]:
                    # self.x_vector[key] = np.linspace(self.param_limits[i][0], self.param_limits[i][1], 101)
                    continue
                if not isinstance(values[key], (int, float)):
                    continue
                if not self.slider_names or not self.slider_names[i][j]: continue
                if key.endswith('id'): continue
                label = self.slider_names[i][j]
                # print(f'{label=} | {key=} | {self.slider_names[i][j]=} | {self.slider_names=}')
                print(f'{label=}')
                if label in self.slider_refs:
                    slider = self.slider_refs[label]
                    slider_group.append(slider)
                    continue
                slider_bottom = slider_top - slider_height
                ax_slider = self.fig.add_axes([0.05, slider_bottom, 0.25, slider_height])
                ax_slider.set_title(label, fontsize=10, pad=4)
                val = values[key]
                if isinstance(val, (list, np.ndarray)):
                    continue  # skip array-valued keys (e.g. x values)
                if lim[0] is None or lim[1] is None:
                    continue  # skip if slider limits are invalid
                slider = Slider(ax_slider, '', lim[0], lim[1], valinit=val)
                slider.on_changed(self.update)
                if self.slider_marks and self.slider_marks[i][j]:
                    for mark in self.slider_marks[i][j]:
                        ax_slider.axvline(x=mark, color='gray', linestyle=':', alpha=0.5)
                slider_group.append(slider)
                self.slider_refs[label] = slider
                slider_top -= (slider_height + slider_gap)
            self.sliders.append(slider_group)

        plt.show()

    def update(self, val):
        for i, (lines, funcs, x_attr) in enumerate(self.plots):
            args = self.init_struct_list[i]
            if isinstance(args, (tuple, list)) and not isinstance(args[0], (int, float, str)):
                agent, extra = args
                new_agent = agent # agent.copy()
                new_extra = extra # extra.copy()
                for label, slider in self.slider_refs.items():
                    try:
                        self._set_nested_attr(agent, label, slider.val)
                    except AttributeError:
                        try:
                            self._set_nested_attr(extra, label, slider.val)
                        except AttributeError:
                            pass
                # x_vals = self._x_from_struct(agent, x_attr)
                # x_vector = self._get_nested_attr(agent, self.x_attr_list[i])
                x_vector = self.x_vector[self.x_attr_list[i]]
                for line, f in zip(lines, funcs):
                    if isinstance(x_vector, (np.ndarray, list)):
                        y = np.array(self.f_vector(f, self.x_attr_list[i], x_vector, agent, extra))
                        line.set_xdata(x_vector)
                    else:
                        y = f(agent, extra)
                        line.set_xdata(x_vals)
                    line.set_ydata(y)

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
        val = getattr(struct, attr) if hasattr(struct, attr) else struct[attr]
        if isinstance(val, (int, float)):
            return np.array([val])
        return np.asarray(val)

import numpy as np
from types import SimpleNamespace

def calc_inner_energy(agent, physical_model):
    c_d = physical_model.energy_conversion_factors['digest']
    c_h = physical_model.energy_conversion_factors['height_energy']
    rest_energy = physical_model.energy_conversion_factors['rest'] * agent.mass ** 0.75
    inner_energy = rest_energy + c_d * np.sum(list(agent.digest_dict.values())) + c_h * agent.height
    inner_energy += agent.brain.size * physical_model.energy_conversion_factors['brain_consumption']
    return inner_energy


# create agent:
from environment import Environment
import simulation_utils

env = Environment(map_filename=config.ENV_PATH,
                               grass_generation_rate=config.GRASS_GENERATION_RATE,
                               leaves_generation_rate=config.LEAVES_GENERATION_RATE)

# Initialize creatures (ensuring they are not in forbidden areas).
agents = simulation_utils.initialize_creatures(num_creatures=1,
                                                       simulation_space=env.size,
                                                       input_size=config.INPUT_SIZE,
                                                       output_size=config.OUTPUT_SIZE,
                                                       eyes_params=config.EYES_PARAMS,
                                                       env=env)

agent = agents[0]
agent.mass = np.linspace(config.INIT_MAX_MASS*0.01, config.INIT_MAX_MASS*10, 101)

ParametricDashboard(
    f_list=[calc_inner_energy],
    x_attr_list=['mass'],
    init_struct_list=[(agent, physical_model)],
    param_limits=[[
        (10, 80),              # mass (x)
        (1.4, 2.0),            # height
        (0.5, 3.0),            # brain.size
        (0.5, 3.0),            # energy_conversion_factors.digest
        (1.0, 3.0),            # energy_conversion_factors.height
        (0.5, 1.5),            # energy_conversion_factors.rest
        (2.0, 5.0),            # energy_conversion_factors.brain_consumption
        (5.0, 15.0)            # physical_model.g
    ]],
    x_labels=["Mass [kg]"],
    y_labels=["Inner Energy [units]"],
    slider_names=[[
        'mass',
        'height',
        'brain.size',
        'physical_model.energy_conversion_factors.digest',
        'physical_model.energy_conversion_factors.height',
        'physical_model.energy_conversion_factors.rest',
        'physical_model.energy_conversion_factors.brain_consumption',
        'physical_model.g'
    ]]
)
