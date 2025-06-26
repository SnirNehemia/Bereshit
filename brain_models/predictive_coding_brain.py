import math

import numpy as np
import torch
import torch.nn.functional as F
import random
import copy

from matplotlib import animation, pyplot as plt


class PredictiveCodingBrain:
    def __init__(self, input_dim, hidden_dims, output_dim=2, energy_reward=False):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims[:]
        self.output_dim = output_dim
        self.energy_reward = energy_reward

        # Weight matrices
        self.W_encode = [torch.randn(h, i) * 0.1 for i, h in zip([input_dim] + hidden_dims[:-1], hidden_dims)]
        self.W_decode = [torch.randn(i, h) * 0.1 for i, h in zip([input_dim] + hidden_dims[:-1], hidden_dims)]
        self.W_action = torch.randn(output_dim, hidden_dims[-1]) * 0.1

        # Hidden state per layer
        self.h = [torch.zeros(h) for h in hidden_dims]

        # Energy memory (for delta-energy reward modulation)
        self.last_energy = None

        # for plot
        self.layer_positions = []
        self.encoder_neurons = []
        self.decoder_neurons = []
        self.encoder_arrows = []
        self.decoder_arrows = []

    def forward(self, x, inference_steps=5, lr=0.1,
                early_stopping=True, error_window=3, error_threshold=1e-3):
        x = x.detach()
        h = [layer.clone().detach() for layer in self.h]
        hidden_states_per_step = []
        errors_per_step = []
        total_errors = []

        for step in range(inference_steps):
            prediction = x
            step_errors = []
            total_error = 0.0

            for i in range(len(h)):
                x_hat = self.W_decode[i] @ h[i]
                error = prediction - x_hat
                delta_h = self.W_encode[i] @ error
                h[i] += lr * delta_h

                # Apply nonlinearity
                h[i] = F.tanh(h[i])  # or F.relu(h[i]), or torch.sigmoid(h[i])

                prediction = h[i]
                step_errors.append(error)
                total_error += torch.norm(error).item()

            hidden_states_per_step.append([layer.clone() for layer in h])
            errors_per_step.append(step_errors)
            total_errors.append(total_error)

            # Early stopping condition
            if early_stopping and step >= error_window:
                recent = total_errors[-error_window:]
                deltas = [abs(recent[i + 1] - recent[i]) for i in range(error_window - 1)]
                if max(deltas) < error_threshold:
                    break

        self.h = h
        force = torch.tanh(self.W_action @ h[-1])
        return force, errors_per_step, hidden_states_per_step

    def local_update(self, x, errors, alpha=0.1, energy=None):
        h_prev = x.detach()
        encoders_change = []
        decoders_change = []
        for i in range(len(self.h)):
            e = errors[i].detach()
            # e = (e - e.mean()) / (e.std() + 1e-6)
            h_curr = self.h[i].detach()

            new_W_decode = self.W_decode[i] + alpha * torch.outer(e, h_curr)
            new_W_encode = self.W_encode[i] + alpha * torch.outer(self.W_encode[i] @ e, h_prev)

            encoders_change.append(torch.norm(self.W_encode[i] - new_W_encode))
            decoders_change.append(torch.norm(self.W_decode[i] - new_W_decode))

            self.W_decode[i] = new_W_decode
            self.W_encode[i] = new_W_encode
            h_prev = h_curr

        if self.energy_reward and energy is not None and self.last_energy is not None:
            delta_energy = energy - self.last_energy
            reward_signal = torch.tanh(torch.tensor(delta_energy, dtype=torch.float32))
            self.W_action += alpha * reward_signal * torch.outer(torch.ones(self.output_dim), self.h[-1])

        self.last_energy = energy
        # self.clip_weights()

        return encoders_change, decoders_change

    def clip_weights(self, min_val=-1.0, max_val=1.0):
        for W in self.W_encode + self.W_decode + [self.W_action]:
            W.clamp_(min_val, max_val)

    def record_current_weights(self, weights_history, is_encode: bool):
        current_weights = []
        if is_encode:
            for i in range(len(self.h)):
                if is_encode:
                    current_weights.append(self.W_encode[i])
            current_weights.append(self.W_action)
        else:
            for i in range(len(self.h)):
                current_weights.append(self.W_decode[i])

        weights_history.append(current_weights)

    def mutate(self,
               mutation_rate=0.1,
               mutation_strength=0.05,
               add_neuron_chance=0.7,  # 0.2,
               remove_neuron_chance=0.7,  # 0.1,
               add_layer_chance=0.7,  # 0.1,
               remove_layer_chance=0.7,  # 0.05
               to_print: bool = False):

        mutant = copy.deepcopy(self)

        # Mutate weights
        for i in range(len(mutant.W_encode)):
            if random.random() < mutation_rate:
                mutant.W_encode[i] += torch.randn_like(mutant.W_encode[i]) * mutation_strength
            if random.random() < mutation_rate:
                mutant.W_decode[i] += torch.randn_like(mutant.W_decode[i]) * mutation_strength

        if random.random() < mutation_rate:
            mutant.W_action += torch.randn_like(mutant.W_action) * mutation_strength

        # Add neuron
        if random.random() < add_neuron_chance:
            layer_idx = random.randint(0, len(mutant.hidden_dims) - 1)
            mutant._add_neuron(layer_idx=layer_idx)

            if to_print:
                print(f'--- neuron added at {layer_idx=} ---')
                mutant.print_brain_structure()

        # Remove neuron
        if random.random() < remove_neuron_chance and any(h > 1 for h in mutant.hidden_dims):
            # pick random layer with >1 neurons
            valid_layers = [i for i, size in enumerate(mutant.hidden_dims) if size > 1]
            if valid_layers:
                layer_idx = random.choice(valid_layers)
                mutant._remove_neuron(layer_idx=layer_idx)

            if to_print:
                print(f'--- neuron removed from {layer_idx=} ---')
                mutant.print_brain_structure()

        # Add new layer
        if random.random() < add_layer_chance:
            insert_idx = random.randint(0, len(mutant.hidden_dims))  # can append at end
            new_size = random.randint(2, 8)
            mutant._add_layer(new_size=new_size, insert_idx=insert_idx)

            if to_print:
                print(f'--- layer added at {insert_idx=} with {new_size=} ---')
                mutant.print_brain_structure()

        # Remove layer
        if random.random() < remove_layer_chance and len(mutant.hidden_dims) > 1:
            layer_idx = random.randint(0, len(mutant.hidden_dims) - 1)
            mutant._remove_layer(layer_idx=layer_idx)

            if to_print:
                print(f'--- layer removed at {layer_idx=} ---')
                mutant.print_brain_structure()

        return mutant

    def _add_neuron(self, layer_idx):
        old_size = self.hidden_dims[layer_idx]
        new_size = old_size + 1
        self.hidden_dims[layer_idx] = new_size

        # Expand hidden state
        self.h[layer_idx] = torch.cat([self.h[layer_idx], torch.zeros(1)])

        # W_encode[layer_idx]: add row
        self.W_encode[layer_idx] = torch.cat([
            self.W_encode[layer_idx],
            torch.randn(1, self.W_encode[layer_idx].shape[1]) * 0.1
        ], dim=0)

        # W_decode[layer_idx]: add column
        self.W_decode[layer_idx] = torch.cat([
            self.W_decode[layer_idx],
            torch.randn(self.W_decode[layer_idx].shape[0], 1) * 0.1
        ], dim=1)

        # If there's a next layer, update its W_encode and W_decode
        if layer_idx + 1 < len(self.hidden_dims):
            # W_encode[layer_idx + 1]: add column
            self.W_encode[layer_idx + 1] = torch.cat([
                self.W_encode[layer_idx + 1],
                torch.randn(self.W_encode[layer_idx + 1].shape[0], 1) * 0.1
            ], dim=1)

            # W_decode[layer_idx + 1]: add row
            self.W_decode[layer_idx + 1] = torch.cat([
                self.W_decode[layer_idx + 1],
                torch.randn(1, self.W_decode[layer_idx + 1].shape[1]) * 0.1
            ], dim=0)

        # If this is the last layer, expand W_action
        if layer_idx == len(self.hidden_dims) - 1:
            self.W_action = torch.cat([
                self.W_action,
                torch.randn(self.output_dim, 1) * 0.1
            ], dim=1)

    def _remove_neuron(self, layer_idx, neuron_idx=None):
        """Remove a neuron from the specified layer (random if neuron_idx is None)"""
        layer_size = self.hidden_dims[layer_idx]
        if layer_size <= 1:
            print(f"Cannot remove from layer {layer_idx} — only one neuron remains.")
            return

        # Pick neuron index
        if neuron_idx is None:
            neuron_idx = random.randint(0, layer_size - 1)

        self.hidden_dims[layer_idx] -= 1

        # Remove from hidden state
        self.h[layer_idx] = torch.cat([
            self.h[layer_idx][:neuron_idx],
            self.h[layer_idx][neuron_idx + 1:]
        ])

        # Remove row from W_encode[i]
        self.W_encode[layer_idx] = torch.cat([
            self.W_encode[layer_idx][:neuron_idx],
            self.W_encode[layer_idx][neuron_idx + 1:]
        ], dim=0)

        # Remove column from W_decode[i]
        self.W_decode[layer_idx] = torch.cat([
            self.W_decode[layer_idx][:, :neuron_idx],
            self.W_decode[layer_idx][:, neuron_idx + 1:]
        ], dim=1)

        # Remove column from W_encode[i+1] (if exists)
        if layer_idx + 1 < len(self.hidden_dims):
            self.W_encode[layer_idx + 1] = torch.cat([
                self.W_encode[layer_idx + 1][:, :neuron_idx],
                self.W_encode[layer_idx + 1][:, neuron_idx + 1:]
            ], dim=1)

            # Remove row from W_decode[i+1]
            self.W_decode[layer_idx + 1] = torch.cat([
                self.W_decode[layer_idx + 1][:neuron_idx],
                self.W_decode[layer_idx + 1][neuron_idx + 1:]
            ], dim=0)

        # Remove from W_action if it's the final layer
        if layer_idx == len(self.hidden_dims) - 1:
            self.W_action = torch.cat([
                self.W_action[:, :neuron_idx],
                self.W_action[:, neuron_idx + 1:]
            ], dim=1)

    def _add_layer(self, insert_idx=None, new_size=4):
        """Insert a new hidden layer at insert_idx (or end if None)"""
        if insert_idx is None:
            insert_idx = len(self.hidden_dims)

        # Previous and next sizes
        prev_size = self.input_dim if insert_idx == 0 else self.hidden_dims[insert_idx - 1]
        next_size = self.hidden_dims[insert_idx] if insert_idx < len(self.hidden_dims) else None

        # Insert into hidden_dims and hidden state
        self.hidden_dims.insert(insert_idx, new_size)
        self.h.insert(insert_idx, torch.zeros(new_size))

        # Create new encode/decode matrices
        W_enc = torch.randn(new_size, prev_size) * 0.1
        W_dec = torch.randn(prev_size, new_size) * 0.1

        self.W_encode.insert(insert_idx, W_enc)
        self.W_decode.insert(insert_idx, W_dec)

        # Update connection to next layer if needed
        if next_size is not None:
            # Replace W_encode[next] and W_decode[next] to fit new input
            self.W_encode[insert_idx + 1] = torch.randn(next_size, new_size) * 0.1
            self.W_decode[insert_idx + 1] = torch.randn(new_size, next_size) * 0.1
        else:
            # If this is the new last layer → update W_action
            self.W_action = torch.randn(self.output_dim, new_size) * 0.1

    def _remove_layer(self, layer_idx=None):
        if len(self.hidden_dims) <= 1:
            print("Cannot remove layer — only one hidden layer remains.")
            return

        if layer_idx is None:
            layer_idx = random.randint(0, len(self.hidden_dims) - 1)

        is_last = (layer_idx == len(self.hidden_dims) - 1)
        is_first = (layer_idx == 0)
        prev_size = self.input_dim if is_first else self.hidden_dims[layer_idx - 1]
        next_size = self.hidden_dims[layer_idx + 1] if not is_last else None

        # Remove from lists
        del self.hidden_dims[layer_idx]
        del self.h[layer_idx]
        del self.W_encode[layer_idx]
        del self.W_decode[layer_idx]

        # If there's a next layer, reconnect it to the previous
        if next_size is not None:
            self.W_encode[layer_idx] = torch.randn(next_size, prev_size) * 0.1
            self.W_decode[layer_idx] = torch.randn(prev_size, next_size) * 0.1

        # If removed the last layer, update W_action
        if is_last:
            self.W_action = torch.randn(self.output_dim, self.hidden_dims[-1]) * 0.1

    def print_brain_structure(self):
        print(f"Layers: {self.input_dim}, {self.hidden_dims}, {self.output_dim}")
        print(f"Encode matrices: {[self.W_encode[i].shape for i in range(len(self.hidden_dims))]}")
        print(f"Decode matrices: {[self.W_decode[i].shape for i in range(len(self.hidden_dims))]}")
        print(f'Output matrix: {self.W_action.shape}')
        print('\n')

    def animate_plot(self, x_inputs=[],
                     hidden_states=[],
                     encoders_weights_history=[], decoders_weights_history=[],
                     total_errors=[], weights_change=[],
                     is_inference: bool = False):

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Animation update function
        def update_inference(frame):
            axes[0].set_title(f"Predictive Coding Brain Inference {frame=}")
            if frame == 0:
                self.setup_ax(ax=axes[0], weights=self.W_encode + [self.W_action], is_encode=True)
                self.setup_ax(ax=axes[1], weights=self.W_decode, is_encode=False)

            # update colors
            activations = self.get_activations(x_input=x_inputs, hs=hidden_states[frame])
            self.update_colors(activations=activations, neurons=self.encoder_neurons,
                               weights=self.W_encode + [self.W_action], arrows=self.encoder_arrows)
            self.update_colors(activations=activations, neurons=self.decoder_neurons,
                               weights=self.W_decode, arrows=self.decoder_arrows)

            # another graph
            self.plot_errors(ax=axes[2], error_over_time=total_errors[:frame])

        def init_func():
            return axes

        def update_input(frame):
            axes[0].set_title(f"Predictive Coding Brain Local Update {frame=}")
            if frame == 0:
                self.setup_ax(ax=axes[0], weights=encoders_weights_history[frame], is_encode=True)
                self.setup_ax(ax=axes[1], weights=decoders_weights_history[frame], is_encode=False)

            # update colors
            activations = self.get_activations(x_input=x_inputs[frame], hs=hidden_states[frame])
            self.update_colors(activations=activations, neurons=self.encoder_neurons,
                               weights=encoders_weights_history[frame], arrows=self.encoder_arrows)
            self.update_colors(activations=activations, neurons=self.decoder_neurons,
                               weights=decoders_weights_history[frame], arrows=self.decoder_arrows)

            # another graph
            # self.plot_errors(ax=axes[2], error_over_time=total_errors[:frame])
            self.plot_errors(ax=axes[2], error_over_time=weights_change[frame])

        if is_inference:
            update = update_inference
            num_frames = len(hidden_states)
        else:
            update = update_input
            num_frames = len(x_inputs)
        ani = animation.FuncAnimation(fig=fig, func=update, frames=num_frames, interval=10, repeat=False,
                                      init_func=init_func)
        plt.tight_layout()
        plt.show()

    def draw_neurons(self, ax):
        neurons = []
        for layer in self.layer_positions:
            for x_, y_ in layer:
                circ = plt.Circle((x_, y_), radius=0.1, color='white', ec='black', zorder=4)
                ax.add_patch(circ)
                neurons.append(circ)
        return neurons

    def draw_arrows(self, ax, weight, from_layer, to_layer, label):
        arrows = []
        for j, (x1, y1) in enumerate(from_layer):
            for k, (x2, y2) in enumerate(to_layer):
                w = weight[k, j].item()  # Note: encode: w[k, j] = neuron k receives from j
                color = (1 - abs(w), 1 - abs(w), 1) if w > 0 else (1, 1 - abs(w), 1 - abs(w))
                color = np.clip(color, 0, 1)

                r = 0.1  # neuron radius
                dx, dy = x2 - x1, y2 - y1
                L = math.hypot(dx, dy)

                # avoid zero division
                if L > 0:
                    shrink = r / L
                    x1_adj = x1 + shrink * dx
                    y1_adj = y1 + shrink * dy
                    dx_adj = dx * (1 - 2 * shrink)
                    dy_adj = dy * (1 - 2 * shrink)

                    arrow = ax.arrow(x1_adj, y1_adj, dx_adj, dy_adj,
                                     head_width=0.05, head_length=0.05,
                                     length_includes_head=True,
                                     color=color, alpha=1)
                    arrows.append(arrow)

        ax.set_xlim(-0.5, len(self.layer_positions) - 0.5)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title(label)
        return arrows

    def get_activations(self, x_input, hs):
        activations = [x_input[i].item() for i in range(self.input_dim)]  # input layer fixed
        for h in hs:
            activations.extend(h.tolist())
        out = (self.W_action @ hs[-1]).tolist()
        activations.extend(out)
        return activations

    def setup_ax(self, ax, weights, is_encode: bool):
        # clear and turn off axes
        ax.clear()
        ax.axis("off")

        # Compute layer sizes
        layer_sizes = [self.input_dim] + self.hidden_dims + [self.output_dim]
        self.layer_positions = []

        for i, size in enumerate(layer_sizes):
            y_pos = np.linspace(-1, 1, size)
            self.layer_positions.append([(i, y) for y in y_pos])

        # Draw neurons
        if is_encode:
            self.encoder_neurons = self.draw_neurons(ax=ax)
        else:
            self.decoder_neurons = self.draw_neurons(ax=ax)

        # draw arrows
        for i, weight in enumerate(weights):
            if is_encode:
                arrows = self.encoder_arrows
                from_layer = self.layer_positions[i]
                to_layer = self.layer_positions[i + 1]
                label = "Encode Connections (Prediction Error Forward)"
            else:
                arrows = self.decoder_arrows
                from_layer = self.layer_positions[i + 1]
                to_layer = self.layer_positions[i]
                label = "Decode Connections (Predictions Backward)"
            arrows.append(
                self.draw_arrows(ax=ax, weight=weight, from_layer=from_layer, to_layer=to_layer, label=label))

    @staticmethod
    def update_colors(activations, neurons, weights, arrows):
        # Update neuron colors
        for val, circ in zip(activations, neurons):
            color = (1 - abs(val), 1 - abs(val), 1) if val > 0 else (1, 1 - abs(val), 1 - abs(val))
            circ.set_facecolor(np.clip(color, 0, 1))

        for arrows_list, weight in zip(arrows, weights):
            for a, w in zip(arrows_list, weight.flatten()):
                color = (1 - abs(w), 1 - abs(w), 1) if w > 0 else (1, 1 - abs(w), 1 - abs(w))
                a.set_color(np.clip(color, 0, 1))

    def plot(self):
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        self.setup_ax(ax=axes[0], weights=self.W_encode + [self.W_action], is_encode=True)
        self.setup_ax(ax=axes[1], weights=self.W_decode, is_encode=False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_errors(ax, error_over_time):
        # Plot error convergence
        ax.clear()
        ax.plot(error_over_time, marker='o', label='Total Prediction Error')
        ax.set_xlabel("Inference Step")
        ax.set_ylabel("Error Norm")
        ax.set_title("Prediction Error Convergence Over Time")
        ax.grid(True)
        ax.legend()

    @staticmethod
    def calc_total_errors_per_step(errors_per_step):
        return [np.sum([torch.norm(errors_in_step_i[layer_idx]).item()
                        for layer_idx in range(len(errors_in_step_i))])
                for errors_in_step_i in errors_per_step]

    @staticmethod
    def plot_total_errors_per_input(total_errors_per_input,
                                    xlabel="Inference Step",
                                    ylabel="Error Norm",
                                    title="Prediction Error Convergence Over Inference Steps"):
        num_inputs = len(total_errors_per_input)
        num_inputs_in_one_plot = num_inputs // 10
        jump_input = num_inputs_in_one_plot // 5
        for i in range(0, num_inputs, num_inputs_in_one_plot):
            plt.figure()
            for input_idx in range(i, i + num_inputs_in_one_plot, jump_input):
                plt.plot(total_errors_per_input[input_idx], '.-', label=f'{input_idx}')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.ylim([0, 0.01])
            plt.title(title)
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
        plt.show()


def record_weights(weights):
    conns = []
    for j in range(weights.shape[1]):
        for k in range(weights.shape[0]):
            conns.append(weights[k, j].item())
    return conns


def example_brain():
    brain = PredictiveCodingBrain(input_dim=5, hidden_dims=[10, 6, 3])
    print('--- Before ----')
    brain.print_brain_structure()
    brain.plot()

    # brain._add_layer(new_size=1, insert_idx=2)
    # brain._add_neuron(layer_idx=2)
    # brain._remove_neuron(layer_idx=2)
    # brain._remove_layer(layer_idx=1)

    brain = brain.mutate(to_print=True)

    print('--- After ----')
    brain.print_brain_structure()
    brain.plot()


def example_animation():
    brain = PredictiveCodingBrain(input_dim=5, hidden_dims=[10, 6, 3])

    is_inference = False
    to_plot_inference_animation = False
    to_plot_input_animation = False
    to_plot_summary_graph = True

    num_inputs = 1000
    num_inference_steps = 30

    total_errors_per_input = []  # size=num_inputs. contain lists of size=num_steps
    hidden_states_per_input = []
    encoders_weights_history = []  # size=num_inputs
    decoders_weights_history = []  # size=num_inputs
    x_inputs = []  # size=num_inputs
    weights_change = []  # size=num_inputs. contain lists of size=num_encoders+num_decoders

    # generate inputs
    for input_idx in range(num_inputs):
        # x_input = torch.rand(brain.input_dim)
        x_input = torch.Tensor(input_idx / num_inputs * np.ones(brain.input_dim))
        x_inputs.append(x_input)

    # Run brain
    for input_idx in range(num_inputs):
        print(f"{input_idx=}")
        x_input = x_inputs[input_idx]

        # record current weights
        brain.record_current_weights(weights_history=encoders_weights_history, is_encode=True)
        brain.record_current_weights(weights_history=decoders_weights_history, is_encode=False)

        # run input and update neurons activity (inference)
        force, errors_per_step, hidden_states_per_step = brain.forward(x=x_input,
                                                                       inference_steps=num_inference_steps)

        total_errors_per_step = brain.calc_total_errors_per_step(errors_per_step)
        total_errors_per_input.append(total_errors_per_step)
        hidden_states_per_input.append(hidden_states_per_step[-1])

        # plot
        if to_plot_inference_animation:
            if input_idx % 10 == 0:
                brain.animate_plot(x_inputs=x_input,
                                   hidden_states=hidden_states_per_step, total_errors=total_errors_per_step,
                                   is_inference=is_inference)

        # update weights
        encoders_change, decoders_change = brain.local_update(x=x_input, errors=errors_per_step[-1])

        # brain = brain.mutate()
        weights_change.append(encoders_change + decoders_change)
        x_inputs.append(x_input)

    if to_plot_input_animation:
        brain.animate_plot(x_inputs=x_inputs,
                           hidden_states=hidden_states_per_input,
                           encoders_weights_history=encoders_weights_history,
                           decoders_weights_history=decoders_weights_history,
                           total_errors=total_errors_per_input, weights_change=weights_change,
                           is_inference=is_inference)

    if to_plot_summary_graph:
        if is_inference:
            brain.plot_total_errors_per_input(total_errors_per_input)
        else:
            # brain.plot_total_errors_per_input(total_errors_per_input)
            brain.plot_total_errors_per_input(weights_change,
                                              xlabel="Encoder/decoder",
                                              ylabel="Forbinius Norm",
                                              title="Forbinius Norm for each weights matrix")


if __name__ == '__main__':
    example_brain()
    # example_animation()
