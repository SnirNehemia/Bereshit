import importlib
from math import floor

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from input.codes.config import load_config
from input.codes.physical_model import load_physical_model


class ParametricDashboard:
    def __init__(self, f_list, x_attr_list, init_struct_list, param_limits, x_labels, y_labels, layout=None,
                 slider_marks=None, slider_names=None, func_colors=None, func_legends=None,
                 shared_slider_labels=None, sample_num=None):
        self.f_list = f_list
        self.init_struct_list = init_struct_list  # list of tuples or lists (agent_dict, extra_args_dict) or just args
        if len(param_limits) != len(x_attr_list):  # list of lists of (min, max) tuples
            self.param_limits = [param_limits] * len(x_attr_list)
        else:
            self.param_limits = param_limits
        self.x_labels = x_labels
        self.y_labels = y_labels
        self.num_funcs = len(f_list)
        self.x_attr_list = x_attr_list
        self.sliders = []
        self.slider_refs = {}
        self.plots = []
        self.axes = []
        self.layout = layout if layout else (self.num_funcs, 1)
        if len(slider_marks) != len(x_attr_list):
            self.slider_marks = [slider_marks] * len(x_attr_list)
        else:
            self.slider_marks = slider_marks
        if len(slider_names) != len(x_attr_list):
            self.slider_names = [slider_names] * len(x_attr_list)
        else:
            self.slider_names = slider_names
        self.func_colors = func_colors
        self.func_legends = func_legends
        self.x_vector = {}
        self.shared_slider_labels = shared_slider_labels if shared_slider_labels else {}
        if sample_num == None:
            sample_num = [101] * len(x_attr_list)
        else:
            self.sample_num = sample_num
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
            if isinstance(obj, dict):
                if part not in obj:
                    raise AttributeError('Attribute not found')
                obj = obj[part]
            else:
                if not hasattr(obj, part):
                    raise AttributeError('Attribute not found')
                obj = getattr(obj, part)
        final_part = parts[-1]
        if isinstance(obj, dict):
            if final_part not in obj:
                raise AttributeError('Attribute not found')
            obj[final_part] = value
        else:
            if not hasattr(obj, final_part):
                raise AttributeError('Attribute not found')
            setattr(obj, final_part, value)

    def _get_nested_attr(self, obj, attr_path):
        for part in attr_path.split('.'):
            if isinstance(obj, dict):
                if part not in obj:
                    raise AttributeError('Attribute not found')
                obj = obj[part]
            else:
                if not hasattr(obj, part):
                    raise AttributeError('Attribute not found')
                obj = getattr(obj, part)
        return obj

    # def _set_nested_attr(self, obj, attr_path, value):
    #     parts = attr_path.split('.')
    #     for part in parts[:-1]:
    #         if isinstance(obj, dict):
    #             if part not in obj:
    #                 raise AttributeError('Attribute not found')
    #             obj = obj[part]
    #         else:
    #             obj = getattr(obj, part)
    #     final_part = parts[-1]
    #     if isinstance(obj, dict):
    #         if final_part not in obj:
    #             raise AttributeError('Attribute not found')
    #         obj[final_part] = value
    #     else:
    #         if not hasattr(obj, final_part):
    #             raise AttributeError('Attribute not found')
    #         setattr(obj, final_part, value)
    #
    #
    # def _get_nested_attr(self, obj, attr_path):
    #     for part in attr_path.split('.'):
    #         if isinstance(obj, dict):
    #             obj = obj[part]
    #         else:
    #             obj = getattr(obj, part)
    #     return obj

    # def _set_nested_attr(self, obj, attr_path, value):
    #     parts = attr_path.split('.')
    #     for part in parts[:-1]:
    #         obj = getattr(obj, part)
    #     if not hasattr(obj, parts[-1]):
    #         raise AttributeError('Attribute not found')
    #     setattr(obj, parts[-1], value)
    #
    # def _get_nested_attr(self, obj, attr_path):
    #     for part in attr_path.split('.'):
    #         obj = getattr(obj, part)
    #     return obj

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

        plt.subplots_adjust(wspace=0.25, hspace=0.5, left=0.3, right=0.95)

        for i in range(self.num_funcs):
            ax = axs[i]
            funcs = self.f_list[i] if isinstance(self.f_list[i], (list, tuple)) else [self.f_list[i]]
            colors = self.func_colors[i] if self.func_colors and i < len(self.func_colors) else [None] * len(funcs)
            labels = self.func_legends[i] if self.func_legends and i < len(self.func_legends) else [None] * len(funcs)
            lines = []

            args = self.init_struct_list[i]
            if isinstance(args, (tuple, list)):  # and isinstance(args[0], dict)
                agent, extra = args
                for j, (name, lim) in enumerate(zip(self.slider_names[i], self.param_limits[i])):
                    if name == self.x_attr_list[i]:
                        k = self.slider_names[i].index(name)
                        self.x_vector[name] = np.linspace(self.param_limits[i][k][0], self.param_limits[i][k][1],
                                                          self.sample_num[i])
                        continue
                try:
                    x_vals = self._x_from_struct(agent, self.x_attr_list[i])
                    for f, color, label in zip(funcs, colors, labels):
                        # x_vector = self._get_nested_attr(agent, self.x_attr_list[i])
                        x_vector = self.x_vector[self.x_attr_list[i]]
                        if isinstance(x_vector, (np.ndarray, list)):
                            y = np.array(self.f_vector(f, self.x_attr_list[i], x_vector, agent, extra))
                            if len(y.shape) > 1:
                                y = y[:, 0]
                            line, = ax.plot(x_vector, y, label=label, color=color, alpha=0.5)
                        else:
                            y = f(agent, extra)
                            line, = ax.plot(x_vals, y, label=label, color=color, alpha=0.5)
                        lines.append(line)
                        continue
                except AttributeError:
                    raise Exception("AttributeError at plot stage")
                    pass

            else:
                x_index = self.x_attr_list[i]
                if not isinstance(x_index, int):
                    print(f"Warning: x_attr_list[{i}] was not an int. Falling back to index 0.")
                    x_index = 0
                x_vals = np.asarray(args[x_index])
                for f, color, label in zip(funcs, colors, labels):
                    y = np.array([f(agent, **extra) for agent in args[0]]) if isinstance(args[0],
                                                                                         (list, np.ndarray)) else f(
                        *args)
                    line = ax.plot(x_vals, y, label=label, color=color)
                    lines.append(line)

            ax.set_xlabel(self.x_labels[i], fontsize=8, labelpad=2)
            ax.set_ylabel(self.y_labels[i], fontsize=8, labelpad=2)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            ax.grid(True)
            if any(labels):
                ax.legend()
            self.plots.append((lines, funcs, self.x_attr_list[i]))
            self.axes.append(ax)

        slider_top = 0.95
        slider_height = 0.02
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
                keys = [self.slider_names[i][j] if self.slider_names and i < len(self.slider_names) and j < len(
                    self.slider_names[i]) and self.slider_names[i][j] else f'param{j}' for j in range(len(args))]
                values = {k: v for k, v in zip(keys, args)}

            for j, (key, lim) in enumerate(zip(self.slider_names[i], self.param_limits[i])):
                if key == self.x_attr_list[i]:
                    # self.x_vector[key] = np.linspace(self.param_limits[i][0], self.param_limits[i][1], 101)
                    continue
                # if not isinstance(values[key], (int, float)):
                #     continue
                if not self.slider_names or not self.slider_names[i][j]: continue
                if key.endswith('id'): continue
                label = self.slider_names[i][j]
                # print(f'{label=} | {key=} | {self.slider_names[i][j]=} | {self.slider_names=}')
                # print(f'{label=}')
                if label in self.slider_refs:
                    slider = self.slider_refs[label]
                    slider_group.append(slider)
                    continue
                slider_bottom = slider_top - slider_height
                ax_slider = self.fig.add_axes([0.05, slider_bottom, 0.15, slider_height])
                ax_slider.set_title(label, fontsize=8, pad=2)
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

    # def update(self, val):
    #     for i, (lines, funcs, x_attr) in enumerate(self.plots):
    #         args = self.init_struct_list[i]
    #         if isinstance(args, (tuple, list)) and not isinstance(args[0], (int, float, str)):
    #             agent, extra = args
    #             new_agent = agent # agent.copy()
    #             new_extra = extra # extra.copy()
    #             updated = False
    #             for label, slider in self.slider_refs.items():
    #                 if '.' in label:
    #                     label = label.split('.')[0] + "['" + label.split('.')[1] + "']"
    def update(self, val):
        for i, (lines, funcs, x_attr) in enumerate(self.plots):
            args = self.init_struct_list[i]
            if isinstance(args, (tuple, list)) and not isinstance(args[0], (int, float, str)):
                agent, extra = args
                new_agent = agent  # agent.copy()
                new_extra = extra  # extra.copy()
                for label, slider in self.slider_refs.items():
                    updated = False
                    try:
                        self._set_nested_attr(agent, label, slider.val)
                        updated = True
                    except AttributeError:
                        try:
                            self._set_nested_attr(extra, label, slider.val)
                            updated = True
                        except AttributeError:
                            pass
                    if not updated:
                        raise Exception("AttributeError at update stage")
                # x_vals = self._x_from_struct(agent, x_attr)
                # x_vector = self._get_nested_attr(agent, self.x_attr_list[i])
                # x_vector = self.x_vector[self.x_attr_list[i]]
                for j, (line, f) in enumerate(zip(lines, funcs)):
                    x_vector = self.x_vector[self.x_attr_list[i]]
                    if isinstance(x_vector, (np.ndarray, list)):
                        y = np.array(self.f_vector(f, self.x_attr_list[i], x_vector, agent, extra))
                        line.set_xdata(x_vector)
                    else:
                        y = f(agent, extra)
                        line.set_xdata(x_vals)
                    line.set_ydata(y)

            else:
                keys = [self.slider_names[i][j] if self.slider_names and i < len(self.slider_names) and j < len(
                    self.slider_names[i]) and self.slider_names[i][j] else f'param{j}' for j in range(len(args))]
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


# -------------- Energy functions --------------

def calc_inner_energy(agent, physical_model):
    c_d = physical_model.energy_conversion_factors['digest']
    c_h = physical_model.energy_conversion_factors['height_energy']
    rest_energy = physical_model.energy_conversion_factors['rest'] * agent.mass ** 0.75
    inner_energy = rest_energy + c_d * np.sum(list(agent.digest_dict.values())) + c_h * agent.height
    inner_energy += agent.brain.size * physical_model.energy_conversion_factors['brain_consumption']
    return inner_energy


def calc_propulsion_energy(agent, physical_model):
    propulsion_force = agent.strength
    eta = physical_model.energy_conversion_factors['activity_efficiency']
    c_heat = physical_model.energy_conversion_factors['heat_loss']
    propulsion_energy = (1 / eta + c_heat) * propulsion_force
    return propulsion_energy


# -------------- Force functions --------------

def linear_force(agent, physical_model):
    linear_drag_force = - physical_model.gamma * agent.height ** 2 * agent.speed
    return linear_drag_force


def quadratic_force(agent, physical_model):
    quadratic_drag_force = - physical_model.c_drag * agent.height ** 2 * agent.speed ** 2
    return quadratic_drag_force


def total_drag_force(agent, physical_model):
    linear_drag_force = - physical_model.gamma * agent.height ** 2 * agent.speed
    quadratic_drag_force = - physical_model.c_drag * agent.height ** 2 * agent.speed ** 2
    drag_force = linear_drag_force + quadratic_drag_force
    return drag_force


def sim_linear_force(agent, physical_model):
    index = min([floor(agent.t / config.DT), len(agent.log.record['reaction_friction_force']) - 1])
    return np.linalg.norm(agent.log.record['linear_drag_force'][index])


def sim_total_drag_force(agent, physical_model):
    index = min([floor(agent.t / config.DT), len(agent.log.record['reaction_friction_force']) - 1])
    return np.linalg.norm(agent.log.record['drag_force'][index])


def sim_propulsion_force(agent, physical_model):
    index = min([floor(agent.t / config.DT), len(agent.log.record['reaction_friction_force']) - 1])
    return np.linalg.norm(agent.log.record['reaction_friction_force'][index])


# ---------------- General functions ----------------

def run_simulation(agent, physical_model,
                   debug_energy=False, debug_position=False, debug_force=False):
    average_eating_rate = agent.average_eating_rate
    distance = 5
    angle = np.radians(5)
    eyes_inputs = [np.array([1, distance, angle])
                   for _ in range(len(agent.eyes_channels) * len(agent.eyes_params))]  # change to a specific policy
    if debug_energy: print(f'{agent.t:.1f} starting -> {agent.energy:.1f}')
    if debug_position: print(f'{agent.t:.1f} starting -> {np.linalg.norm(agent.position)=:.1f}')
    if agent.t >= agent.max_age:
        if debug_energy or debug_position: print('finished simulation')
        if debug_energy: print(f'{agent.live_count=} | {agent.dead_count=} | {agent.energy=}'
                               f' | {agent.eat_count=} | {agent.children_count=}')
        if debug_position: print(f'{np.linalg.norm(agent.position)=:.1f}')
    if agent.t == 0:
        if debug_energy: print('restart simulation')
        agent.init_state(balance=True)
        agent.live_count = 0
        agent.dead_count = 0
        agent.eat_count = 0
        agent.children_count = 0
        agent.energy = 0.8 * (
                config.REPRODUCTION_ENERGY + config.MIN_LIFE_ENERGY)
        agent.position = [0, 0]
        agent.velocity = np.array([0.01, 0.01])
        agent.speed = np.linalg.norm(agent.velocity)
        return 0
    if agent.energy <= 0:
        agent.energy = 0
        agent.dead_count += 1
        return 0
    agent.live_count += 1
    brain_input = np.hstack([np.array([agent.hunger, agent.thirst]), agent.speed, np.concatenate(eyes_inputs)])
    if debug_energy: print(f'\t\t{brain_input=}')
    decision = agent.think(brain_input)
    if 0.5 < agent.own_0_const_1_rand_2 < 1.5:
        decision[0], decision[1] = agent.decision, np.pi
    if 1.5 < agent.own_0_const_1_rand_2 < 2.5:
        decision[0], decision[1] = (
            agent.randn_t[floor(agent.t / config.DT)] * agent.decision_rand_mag + agent.decision, np.pi)
    if debug_energy: print(f'\t{agent.t:.1f} after thinking -> \t{agent.energy:.1f}')
    if debug_energy: print(f'\t\t{decision=}')
    agent.move(decision=decision, dt=config.DT,
               debug_energy=debug_energy, debug_position=debug_position, debug_force=debug_force)
    if debug_position: print(f'\t{agent.t:.1f} after moving -> \t{np.linalg.norm(agent.position)=:.1f}')
    if debug_energy: print(f'\t{agent.t:.1f} after moving -> \t{agent.energy:.1f}')
    if agent.own_0_const_1_rand_2 > 0.5:
        if np.linalg.norm(agent.position) >= average_eating_rate:
            agent.position = [0, 0]
            agent.energy += agent.digest_dict['grass'] * physical_model.GRASS_ENERGY
            agent.eat_count += 1
    else:
        if agent.rand_t[floor(agent.t / config.DT)] * average_eating_rate < config.DT:
            # creature.eat(food_type='grass', food_energy=config.GRASS_ENERGY)  # TODO: make it work!
            agent.energy += agent.digest_dict['grass'] * physical_model.GRASS_ENERGY
            agent.eat_count += 1
    if agent.energy > agent.reproduction_energy + config.MIN_LIFE_ENERGY:
        agent.energy -= agent.reproduction_energy
        agent.children_count += 1


def energy_over_time(agent, physical_model, debug_energy=False):
    run_simulation(agent, physical_model, debug_energy=debug_energy)
    return agent.energy


def position_over_time(agent, physical_model, debug_position=False):
    run_simulation(agent, physical_model, debug_position=debug_position, debug_force=debug_position)
    return np.linalg.norm(agent.position)


def speed_over_time(agent, physical_model):
    index = min([floor(agent.t / config.DT), len(agent.log.record['speed']) - 1])
    if index == -1:
        return 0
    else:
        return agent.log.record['speed'][index]


def sim_energy_inner(agent, physical_model):
    index = min([floor(agent.t / config.DT), len(agent.log.record['energy_inner']) - 1])
    return agent.log.record['energy_inner'][index]


def sim_energy_propulsion(agent, physical_model):
    index = min([floor(agent.t / config.DT), len(agent.log.record['energy_propulsion']) - 1])
    return agent.log.record['energy_propulsion'][index]

    # step_num = len(creature.t)
    # energy_log = np.zeros(step_num)
    # for i_t in range(step_num):
    #     brain_input = []
    #     brain_input.append(np.array([creature.hunger, creature.thirst]))
    #     brain_input.append(creature.speed)
    #     brain_input.append(np.concatenate(eyes_inputs))
    #     brain_input = np.hstack(brain_input)
    #     decision = creature.think(brain_input)
    #     creature.move(decision=decision, dt=config.DT)
    #     if creature.t[i_t]%average_eating_rate:
    #         creature.eat(food_type='grass', food_energy=config.GRASS_ENERGY)
    #         eat_count += 1
    #     if creature.energy > creature.reproduction_energy + config.MIN_LIFE_ENERGY:
    #         creature.energy -= creature.reproduction_energy
    #         children_count += 1
    #         if creature.energy <= 0:
    #             break
    #     energy_log[i_t] = creature.energy
    #
    # return energy_log


if __name__ == '__main__':
    # Load config
    config_yaml_relative_path = r"input\yamls\2025_06_20_config.yaml"
    config = load_config(yaml_relative_path=config_yaml_relative_path)

    # Load physical model
    physical_model_yaml_relative_path = r"input\yamls\2025_04_18_physical_model.yaml"
    physical_model = load_physical_model(yaml_relative_path=physical_model_yaml_relative_path)

    # create agent:
    from environment import Environment
    import simulation_utils

    env = Environment(map_filename=config.ENV_PATH,
                      grass_generation_rate=config.GRASS_GENERATION_RATE,
                      leaves_generation_rate=config.LEAVES_GENERATION_RATE)

    # Initialize creatures (ensuring they are not in forbidden areas).
    brain_module = importlib.import_module(f"brain_models.{config.BRAIN_TYPE}")
    brain_obj = getattr(brain_module, 'Brain')
    agents = simulation_utils.initialize_creatures(env=env, brain_obj=brain_obj)

    agent = agents[0]
    agent.t = np.arange(0, agent.max_age) * config.DT  # time vector
    agent.average_eating_rate = 500
    agent.rand_t = np.random.rand(int(agent.max_age // config.DT) + 1)
    agent.randn_t = np.random.randn(int(agent.max_age // config.DT) + 1)
    agent.own_0_const_1_rand_2 = 1
    agent.decision = 1
    agent.decision_rand_mag = 0.5
    agent.move(decision=[agent.decision, 0], dt=config.DT)
    agent.max_age = agent.max_age // 10

    physical_model.GRASS_ENERGY = config.GRASS_ENERGY

    slider_names = ['mass',
                    'height',
                    'strength',
                    'mu_static',
                    'mu_kinetic',
                    'brain.size',
                    'energy_conversion_factors.activity_efficiency',
                    'energy_conversion_factors.heat_loss',
                    'energy_conversion_factors.digest',
                    'energy_conversion_factors.height_energy',
                    'energy_conversion_factors.rest',
                    'energy_conversion_factors.brain_consumption',
                    'energy_conversion_factors.mass_energy',
                    'g',
                    'speed',
                    'gamma',
                    'c_drag',
                    't',
                    'average_eating_rate',
                    'own_0_const_1_rand_2',
                    'decision',
                    'decision_rand_mag',
                    'GRASS_ENERGY']
    param_limits = [
        (0.1, 80),  # mass (x)
        (config.INIT_MAX_HEIGHT * 0.01, config.INIT_MAX_HEIGHT),  # height
        (0.5, 30),  # strength
        (0, 2),  # mu_static
        (0, 2),  # mu_kinetic
        (1, 30),  # brain.size
        (0.1, 1.0),  # energy_conversion_factors.activity_efficiency
        (0, 0.1),  # energy_conversion_factors.heat_loss
        (0.05, 1.0),  # energy_conversion_factors.digest
        (1.0, 30.0),  # energy_conversion_factors.height
        (0.0001, 0.001),  # energy_conversion_factors.rest
        (0.001, 5.0),  # energy_conversion_factors.brain_consumption
        (0.001, 5.0),  # energy_conversion_factors.mass_energy
        (5.0, 15.0),  # physical_model.g
        (0, config.MAX_SPEED * 10),  # speed
        (0, physical_model.gamma * 2),  # physical_model.gamma
        (0, physical_model.c_drag * 2),  # physical_model.c_drag
        (0, agent.max_age),  # t
        (1000, 10000),  # average_eating_rate
        (0, 2),  # override decision with 3 classes
        (0, 1),  # decision
        (0, 0.5),  # decision_rand_mag
        (1000, 10000),  # GRASS_ENERGY
    ]
    slider_marks = [
        [config.INIT_MAX_MASS],  # mass (x)
        [config.INIT_MAX_HEIGHT],  # height
        [config.INIT_MAX_STRENGTH],  # strength
        [physical_model.mu_static],  # mu_static
        [physical_model.mu_kinetic],  # mu_kinetic
        [],  # brain.size
        [physical_model.energy_conversion_factors['activity_efficiency']],  # energy_conversion_factors.activity
        [physical_model.energy_conversion_factors['heat_loss']],  # energy_conversion_factors.heat
        [physical_model.energy_conversion_factors['digest']],  # energy_conversion_factors.digest
        [physical_model.energy_conversion_factors['height_energy']],  # energy_conversion_factors.height
        [physical_model.energy_conversion_factors['rest']],  # energy_conversion_factors.rest
        [physical_model.energy_conversion_factors['brain_consumption']],  # energy_conversion_factors.brain_consumption
        [],  # energy_conversion_factors.mass
        [physical_model.g],  # physical_model.g
        [config.MAX_SPEED],  # speed
        [physical_model.gamma],  # physical_model.gamma
        [physical_model.c_drag * 0.1, physical_model.c_drag * 10],  # physical_model.c_drag
        [0, agent.max_age],  # t
        [1, 500],  # average_eating_rate
        [0, 1],  # override decision
        [],  # decision
        [],  # decision_rand_mag
        [1000, 10000],  # GRASS_ENERGY
    ]

    f_list = []
    x_attr_list = []
    init_struct_list = []
    x_labels = []
    y_labels = []
    func_colors = []
    func_legends = []
    sample_num = []
    layout = (3, 2)

    # energy plot

    f_list.append([])
    f_list[-1].append(calc_inner_energy)
    f_list[-1].append(calc_propulsion_energy)
    x_attr_list.append('height')
    sample_num.append(101)
    init_struct_list.append((agent, physical_model))
    x_labels.append('height [m]')
    y_labels.append('Inner Energy [units]')
    func_colors.append(['blueviolet', 'violet'])
    func_legends.append(['Inner Energy', 'Propulsion Energy'])

    # force plot

    f_list.append([])
    f_list[-1].append(linear_force)
    f_list[-1].append(quadratic_force)
    f_list[-1].append(total_drag_force)
    x_attr_list.append('speed')
    sample_num.append(101)
    init_struct_list.append((agent, physical_model))
    x_labels.append('velocity [m/s]')
    y_labels.append('Drag Force [N]')
    func_colors.append(['blueviolet', 'violet', 'black'])
    func_legends.append(['Linear Drag', 'Quadratic Drag', 'Linear + Quadratic Drag'])

    # Simulated energy plot

    f_list.append(energy_over_time)
    x_attr_list.append('t')
    sample_num.append(int(agent.max_age // config.DT))
    init_struct_list.append((agent, physical_model))
    x_labels.append('time [s]')
    y_labels.append('Energy [J]')
    func_colors.append(['black'])
    func_legends.append(['Energy'])

    # # Simulated position plot
    #
    # f_list.append(position_over_time)
    # x_attr_list.append('t')
    # sample_num.append(int(agent.max_age//config.DT))
    # init_struct_list.append((agent, physical_model))
    # x_labels.append('time [s]')
    # y_labels.append('Position [m]')
    # func_colors.append(['black'])
    # func_legends.append(['|r|'])

    # Simulated force plot

    f_list.append([])
    # f_list[-1].append(sim_linear_force)
    f_list[-1].append(sim_total_drag_force)
    f_list[-1].append(sim_propulsion_force)
    x_attr_list.append('t')
    sample_num.append(int(agent.max_age // config.DT))
    init_struct_list.append((agent, physical_model))
    x_labels.append('time [s]')
    y_labels.append('Force [N]')
    func_colors.append(['blueviolet', 'violet', 'black'])
    func_legends.append(['Linear Drag', 'Linear + Quadratic Drag', 'Propulsion Force'])

    # Simulated energy plot

    f_list.append([])
    f_list[-1].append(sim_energy_inner)
    f_list[-1].append(sim_energy_propulsion)
    x_attr_list.append('t')
    sample_num.append(int(agent.max_age // config.DT))
    init_struct_list.append((agent, physical_model))
    x_labels.append('time [s]')
    y_labels.append('Power [Watt]')
    func_colors.append(['blueviolet', 'violet'])
    func_legends.append(['Inner Energy', 'Propulsion Energy'])

    # Simulated speed plot

    f_list.append(speed_over_time)
    x_attr_list.append('t')
    sample_num.append(int(agent.max_age // config.DT))
    init_struct_list.append((agent, physical_model))
    x_labels.append('time [s]')
    y_labels.append('speed [m/s]')
    func_colors.append(['black'])
    func_legends.append(['speed'])
    print('\nmax age is devided by 10 to make it run smoother.\n\n')
    print('own_0_const_1_rand_2 is about how you treat decision and food:\n'
          'own_0_const_1_rand_2 < 0.5 means own decision, and gets food using random chance with mean of average_eating_rate\n'
          'own_0_const_1_rand_2 > 0.5 & <1.5 means const decision, and gets food every average_eating_rate meters\n'
          'own_0_const_1_rand_2 > 1.5 means random decision in range decision+[0,decision_rand_mag], and gets food every average_eating_rate meters')

    ParametricDashboard(
        f_list=f_list,
        x_attr_list=x_attr_list,
        init_struct_list=init_struct_list,
        param_limits=param_limits,
        x_labels=x_labels,
        y_labels=y_labels,
        slider_names=slider_names,
        slider_marks=slider_marks,
        func_colors=func_colors,
        func_legends=func_legends,
        sample_num=sample_num,
        layout=layout
    )
