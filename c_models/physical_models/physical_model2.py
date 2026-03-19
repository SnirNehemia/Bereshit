import numpy as np

from b_basic.sim_config import sim_config
from c_models.physical_models.physical_model_abc import PhysicalModel


class PhysicalModel2(PhysicalModel):
    def __init__(self, **params):
        # init config based on data from yaml
        super().__init__()

        for key, value in params.items():
            setattr(self, key, value)

        # make needed adjustments
        self.trait_energy_func = lambda factor, rate, age: factor * np.exp(-rate * age)

    def move_creature(self, creature, env, decision,
                      debug_position: bool = False, debug_energy: bool = False,
                      debug_force: bool = False):
        """
        Update creature position, velocity and energy given decision (brain output).
        :param creature: Creature
        :param env: Environment
        :param decision: brain output, 2 X 1 vector (magnitude, direction)
        :return:
        """
        # Calc total force and propulsion energy
        total_force, propulsion_energy = \
            self._calc_total_force_and_propulsion_energy(creature=creature, decision=decision,
                                                         debug_force=debug_force)

        # Update position and velocity
        self._update_position_and_velocity(creature=creature, env=env,
                                           total_force=total_force,
                                           debug_position=debug_position)

        # update energy
        self._update_energy(creature=creature, propulsion_energy=propulsion_energy, debug_energy=debug_energy)

    def digest_food(self, creature, food_type, food_energy,
                    rebalance: bool = False):
        gained_energy = creature.digest_dict[food_type] * food_energy

        if creature.age <= creature.adolescence_age:
            creature.height, height_energy = self._convert_gained_energy_to_trait(
                trait_type='height_energy',
                old_trait=creature.height,
                gained_energy=gained_energy,
                age=creature.age,
            )
            creature.mass, mass_energy = self._convert_gained_energy_to_trait(
                trait_type='mass_energy',
                old_trait=creature.mass,
                gained_energy=gained_energy,
                age=creature.age,
            )
        else:
            mass_energy, height_energy = 0, 0

        excess_energy = gained_energy - height_energy - mass_energy
        creature.energy += excess_energy

        if rebalance:
            creature.log.add_record('energy_excess', excess_energy)
            creature.log.add_record('energy_gain', gained_energy)
            creature.log.add_record('energy_height', height_energy)
            creature.log.add_record('energy_mass', mass_energy)
        else:
            creature.log.add_record('energy_excess', excess_energy)

    def _transform_propulsion_force(self, creature, decision):
        # Clip propulsion force based on strength
        propulsion_force_mag, relative_propulsion_force_angle = decision
        propulsion_force_mag = np.clip(propulsion_force_mag, 0, 1) * creature.strength

        # transform relative propulsion_force to global cartesian coordinates (x,y)
        current_direction = creature.get_heading()
        cos_angle = np.cos(relative_propulsion_force_angle)
        sin_angle = np.sin(relative_propulsion_force_angle)
        global_propulsion_force_direction = np.array([
            current_direction[0] * cos_angle - current_direction[1] * sin_angle,
            current_direction[0] * sin_angle + current_direction[1] * cos_angle
        ])
        global_propulsion_force = global_propulsion_force_direction * propulsion_force_mag

        return global_propulsion_force

    def _calc_gravity_and_normal_forces(self, creature):
        """
        # Calculate gravity and normal force.
        Right now (2D movement) normal force is only used for friction
        and not directly in equation of motion.
        :param mass: creature mass
        :return:
        """
        gravity_force = creature.mass * self.g
        normal_force = - gravity_force
        return gravity_force, normal_force

    def _calc_reaction_friction_force(self, normal_force, propulsion_force):
        normal_force_mag = np.linalg.norm(normal_force)
        propulsion_force_mag = np.linalg.norm(propulsion_force)
        if propulsion_force_mag > self.mu_static * normal_force_mag:
            propulsion_force_direction = propulsion_force / propulsion_force_mag
            reaction_friction_force = - self.mu_kinetic * normal_force_mag * \
                                      propulsion_force_direction
        else:
            reaction_friction_force = - propulsion_force
        return reaction_friction_force

    def _calc_drag_force(self, creature):
        drag_force = [0, 0]

        if creature.speed > 1e-3:
            current_direction = creature.velocity / creature.speed
            linear_drag_force = - self.gamma * creature.height ** 2 * creature.velocity
            quadratic_drag_force = - self.c_drag * creature.height ** 2 * creature.speed ** 2 * current_direction
            drag_force = linear_drag_force + quadratic_drag_force

        return drag_force

    def _calc_propulsion_energy(self, propulsion_force):
        eta = self.energy_conversion_factors['activity_efficiency']
        c_heat = self.energy_conversion_factors['heat_loss']
        propulsion_energy = (1 / eta + c_heat) * np.linalg.norm(propulsion_force)
        return propulsion_energy

    def _calc_inner_energy(self, creature):
        c_d = self.energy_conversion_factors['digest']
        c_h = self.energy_conversion_factors['height_energy']
        rest_energy = self.energy_conversion_factors['rest'] * creature.mass ** 0.75  # adds mass (BMR) energy
        inner_energy = rest_energy + c_d * np.sum(
            list(creature.digest_dict.values())) + c_h * creature.height  # adds height energy
        inner_energy = inner_energy + creature.brain.size * self.energy_conversion_factors['brain_consumption']
        return inner_energy

    def _calc_trait_energy(self, trait_type, gained_energy, age):
        trait_energy_params = self.trait_energy_params_dict[trait_type]
        factor = trait_energy_params['factor']
        rate = trait_energy_params['rate']
        trait_energy_func = self.trait_energy_func(factor=factor, rate=rate, age=age)
        trait_energy = trait_energy_func * gained_energy
        return trait_energy

    def _convert_gained_energy_to_trait(self, trait_type: str, old_trait: float,
                                        gained_energy: float, age: float,
                                        ):
        trait_energy = self._calc_trait_energy(trait_type=trait_type, gained_energy=gained_energy, age=age)
        c_trait = self.energy_conversion_factors[trait_type]
        new_trait = old_trait + trait_energy / c_trait
        return new_trait, trait_energy

    def _calc_total_force_and_propulsion_energy(self, creature, decision,
                                                debug_force: bool = False):
        # Propulsion force
        global_propulsion_force = self._transform_propulsion_force(creature=creature, decision=decision)
        propulsion_energy = self._calc_propulsion_energy(global_propulsion_force)

        # Gravity and normal force
        gravity_force, normal_force = self._calc_gravity_and_normal_forces(creature=creature)

        # Reaction friction force
        reaction_friction_force = self._calc_reaction_friction_force(
            normal_force=normal_force, propulsion_force=global_propulsion_force)

        # Drag force (air resistence)
        drag_force = self._calc_drag_force(creature=creature)

        total_force = reaction_friction_force + drag_force

        if debug_force:
            print(f'\t\t{reaction_friction_force=}\n'
                  f'\t\t{drag_force=}')

        # if self.is_agent:
        creature.log.add_record('gravity_force', gravity_force)
        creature.log.add_record('normal_force', normal_force)
        creature.log.add_record('reaction_friction_force', reaction_friction_force)
        creature.log.add_record('drag_force', drag_force)

        return total_force, propulsion_energy

    def _update_energy(self, creature, propulsion_energy,
                       debug_energy: bool = False):
        inner_energy = self._calc_inner_energy(creature)
        creature.energy -= propulsion_energy + inner_energy

        # if self.is_agent:
        creature.log.add_record('energy_propulsion', propulsion_energy)
        creature.log.add_record('energy_inner', inner_energy)
        creature.log.add_record('energy_consumption', inner_energy + propulsion_energy)
        if creature.log.record['energy_consumption'][-1] < 0:
            breakpoint('energy consumption < 0')
            raise ValueError('Energy consumption cannot be negative')
        if debug_energy: print(f'\t\t{propulsion_energy=:.1f} | {inner_energy=:.1f}')

    def _update_position_and_velocity(self, creature, env, total_force,
                                      debug_position: bool = False):
        dt = sim_config.config.DT
        acceleration = total_force / creature.mass
        new_velocity = creature.velocity + acceleration * dt

        # Check for collision (obstacle, boundaries of map
        obstacle_mask = env.obstacle_mask
        resolved_pos, is_collided = \
            self._move_on_torus(obstacle=obstacle_mask,
                                p0=creature.position,
                                velocity=new_velocity,
                                dt=dt)

        if debug_position:
            print(f'\t\t{acceleration=}\n'
                  f'\t\t{creature.velocity=} --> {new_velocity=}\n'
                  f'\t\t{creature.position} --> {resolved_pos}')

        # Update position, velocity and speed
        creature.position = resolved_pos
        if is_collided:
            creature.velocity = np.array([0.0, 0.0])
        else:
            creature.velocity = new_velocity
        creature.calc_speed()

    @staticmethod
    def _move_on_torus(obstacle: np.ndarray,
                       p0, velocity, dt, eps=1e-6, wrap_output=True):
        """
        obstacle: (H,W) bool, obstacle[y,x]=True means blocked
        p0: (2,) float world pos (can be unwrapped)
        velocity: (2,) float world units per second (in grid-cell units)
        dt: float seconds
        Returns: (pos (2,), collided (bool))

        Uses direction from velocity (dp = v*dt). Grid indexing wraps (torus).
        """
        H, W = obstacle.shape
        p0 = np.asarray(p0, np.float64)
        v = np.asarray(velocity, np.float64)

        dp = v * float(dt)
        if dp[0] == 0.0 and dp[1] == 0.0:
            ix, iy = int(np.floor(p0[0])) % W, int(np.floor(p0[1])) % H
            hit = bool(obstacle[iy, ix])
            out = np.array([p0[0] % W, p0[1] % H], np.float64) if wrap_output else p0.copy()
            return out, hit

        p1 = p0 + dp

        gx0, gy0 = p0
        dx, dy = dp
        ix, iy = int(np.floor(gx0)), int(np.floor(gy0))

        if obstacle[iy % H, ix % W]:
            out = np.array([p0[0] % W, p0[1] % H], np.float64) if wrap_output else p0.copy()
            return out, True

        sx = 0 if dx == 0.0 else (1 if dx > 0 else -1)
        sy = 0 if dy == 0.0 else (1 if dy > 0 else -1)

        invx = np.inf if dx == 0.0 else 1.0 / abs(dx)
        invy = np.inf if dy == 0.0 else 1.0 / abs(dy)

        tx = (((ix + 1) - gx0) if sx > 0 else (gx0 - ix)) * invx if invx != np.inf else np.inf
        ty = (((iy + 1) - gy0) if sy > 0 else (gy0 - iy)) * invy if invy != np.inf else np.inf
        dtx, dty = invx, invy

        while True:
            if tx < ty:
                t = tx
                tx += dtx
                ix += sx
            else:
                t = ty
                ty += dty
                iy += sy

            if t > 1.0:
                break

            if obstacle[iy % H, ix % W]:
                t = max(0.0, t - eps)
                p = p0 + dp * t
                out = np.array([p[0] % W, p[1] % H], np.float64) if wrap_output else p
                return out, True

        out = np.array([p1[0] % W, p1[1] % H], np.float64) if wrap_output else p1
        return out, False

    @staticmethod
    def _debug_plot_torus_movement(obstacle, p0, velocity, dt, move_func):
        """
        obstacle : (H,W) bool grid
        p0       : start position
        velocity : velocity vector
        dt       : timestep
        move_func: move_torus_point_vel function
        """

        H, W = obstacle.shape
        p0 = np.asarray(p0, float)
        velocity = np.asarray(velocity, float)

        attempted = p0 + velocity * dt
        resolved, collided = move_func(obstacle, p0, velocity, dt)

        fig, ax = plt.subplots(figsize=(6, 5))

        # draw obstacles
        ax.imshow(obstacle, origin="lower", cmap="gray_r", extent=[0, W, 0, H])

        # grid lines
        for x in range(W + 1):
            ax.axvline(x, color="lightgray", linewidth=0.5)
        for y in range(H + 1):
            ax.axhline(y, color="lightgray", linewidth=0.5)

        # draw path copies (for torus visualization)
        shifts = [-W, 0, W]
        for sx in shifts:
            for sy in [-H, 0, H]:
                ax.plot(
                    [p0[0] + sx, attempted[0] + sx],
                    [p0[1] + sy, attempted[1] + sy],
                    "b--",
                    alpha=0.3
                )

        # main path
        ax.plot([p0[0], attempted[0]], [p0[1], attempted[1]],
                "b--", label="attempted")

        # start
        ax.scatter(*p0, c="green", s=120, label="start")

        # attempted end
        ax.scatter(*(attempted % [W, H]), c="blue", s=120, label="target")

        # resolved
        ax.scatter(*resolved, c="red", s=120, label="resolved")

        # velocity arrow
        ax.arrow(p0[0], p0[1],
                 velocity[0] * dt, velocity[1] * dt,
                 head_width=0.15, length_includes_head=True,
                 color="blue")

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")

        title = "Collision detected" if collided else "No collision"
        ax.set_title(title)

        ax.legend()
        plt.show()


if __name__ == '__main__':
    from time import time
    import matplotlib.pyplot as plt
    from matplotlib import use

    use('TkAgg')

    pm = PhysicalModel2()
    old_pos = np.array([0, 1])
    v = np.array([2, 2])
    dt = 3
    obstacle_mask = np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0]])

    start = time()
    resolved_pos, is_collided = pm._move_on_torus(obstacle=obstacle_mask, p0=old_pos, velocity=v, dt=dt)
    end = round(time() - start, 2)
    print(f"{resolved_pos=}, {is_collided=}, {end=}sec")

    # pm._debug_torus_movement(obstacle=obstacle_mask, p0=old_pos, velocity=v, dt=dt, move_func=pm._move_on_torus)
