import numpy as np

from b_basic.sim_config import sim_config
from c_models.physical_models.physical_model_abc import PhysicalModel


class PhysicalModel1(PhysicalModel):
    def __init__(self, **params):
        # init config based on data from yaml
        super().__init__()

        for key, value in params.items():
            setattr(self, key, value)

    def move_creature(self, creature, env, decision, **kwargs):
        """
        Update creature position, velocity and energy given decision (brain output).
        :param creature: Creature
        :param env: Environment
        :param decision: brain output, 2 X 1 vector (magnitude, direction)
        :param kwargs:
        :return:
        """

        # update position, velocity and speed
        force = self._transform_propulsion_force(creature=creature, decision=decision)
        self._update_position_and_velocity(creature=creature, env=env, total_force=force)

        # update energy
        force_mag = decision[0]
        propulsion_energy = self.energy_conversion_factors['activity_efficiency'] * force_mag
        rest_energy = self.energy_conversion_factors['rest'] * creature.mass ** 0.75
        brain_energy = self.energy_conversion_factors['brain_consumption'] * creature.brain.size
        inner_energy = rest_energy + brain_energy
        creature.energy -= propulsion_energy + inner_energy

    def digest_food(self, creature, food_type, food_energy, **kwargs):
        """
        Update creature energy based on given food.
        :param creature: Creature
        :param food_type: str
        :param food_energy: float
        :param kwargs:
        :return:
        """

        creature.energy += creature.digest_dict[food_type] * food_energy

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

    pm = PhysicalModel1()
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
