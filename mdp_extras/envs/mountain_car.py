import numpy as np
import itertools as it

import gym

from gym.envs.classic_control import Continuous_MountainCarEnv


from scipy.stats import multivariate_normal

from mdp_extras import FeatureFunction


# Range of the mountain car state space
MC_POSITION_RANGE = (-1.2, 0.6)
MC_VELOCITY_RANGE = (-0.07, 0.07)


class GaussianBasis(FeatureFunction):
    """Gaussian basis feature function for MountainCar

    A set of Gaussian functions spanning the state space
    """

    def __init__(self, num=5, pos_range=(-1.2, 0.6), vel_range=(-0.07, 0.07)):
        """C-tor"""
        super().__init__(self.Type.OBSERVATION)

        self.dim = num ** 2
        pos_delta = pos_range[1] - pos_range[0]
        vel_delta = vel_range[1] - vel_range[0]

        pos_mean_diff = pos_delta / (num + 1)
        pos_basis_means = (
            np.linspace(pos_mean_diff * 0.5, pos_delta - pos_mean_diff * 0.5, num)
            + pos_range[0]
        )
        pos_basis_std = pos_mean_diff ** 2 / 10

        vel_mean_diff = vel_delta / (num + 1)
        vel_basis_means = (
            np.linspace(vel_mean_diff * 0.5, vel_delta - vel_mean_diff * 0.5, num)
            + vel_range[0]
        )
        vel_basis_std = vel_mean_diff ** 2 / 10

        covariance = np.diag([pos_basis_std, vel_basis_std])
        means = np.array(list(it.product(pos_basis_means, vel_basis_means)))

        self.rvs = [multivariate_normal(m, covariance) for m in means]

    def __len__(self):
        """Get the length of the feature vector"""
        return self.dim

    def __call__(self, o1, a=None, o2=None):
        """Get feature vector given state(s) and/or action"""
        return np.array([rv.pdf(o1) for rv in self.rvs])


class Cylindrical_Continuous_MountainCarEnv(Continuous_MountainCarEnv):
    """A continuous MountainCar environment with a cylindrical position dimension

    This is essentially analogous to the Pendulum environment

    Also see Desmos plot at https://www.desmos.com/calculator/mbrks8vzet
    """

    def __init__(self, goal_pos_tolerance=0.05, goal_velocity=None):
        super().__init__(goal_velocity=goal_velocity)

        self.goal_pos_tolerance = goal_pos_tolerance

        # Scaling factor to convert MountainCar range to a full period
        # Original curve from Gym Continuous_MountainCarEnv
        # return np.sin(3 * xs)*.45+.55
        self.pos_rescale = 2 * np.pi / np.abs(self.min_position - self.max_position)

    def _height(self, xs):
        return np.sin(self.pos_rescale * xs) * 0.45 + 0.55

    def _height_tangent(self, xs):
        # Original curve from Gym Continuous_MountainCarEnv
        # math.cos(3 * position)
        return np.cos(self.pos_rescale * xs)

    def _power_mod(self, xs):
        return -0.0025 * np.cos(self.pos_rescale * xs)

    def step(self, action):
        position, velocity = self.state

        # Convert action to acceleration
        force = min(max(action[0], self.min_action), self.max_action)
        # Original curve from Gym Continuous_MountainCarEnv
        # -0.0025 * math.cos(3 * position)
        velocity += force * self.power - 0.0025 * self._height_tangent(position)

        # Clamp velocity
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed

        # Update position
        position += velocity

        # Clamp position
        if position > self.max_position:
            position = self.min_position
        if position < self.min_position:
            position = self.max_position

        # Convert a possible numpy bool to a Python bool
        near_goal = np.abs(position - self.goal_position) <= self.goal_pos_tolerance
        slow_enough = True
        if self.goal_velocity is not None:
            slow_enough = velocity <= self.goal_velocity

        reward = 0
        if near_goal and slow_enough:
            reward += 10.0
        reward -= (action[0] ** 2) * 0.1

        self.state = np.array([position, velocity])
        return self.state, reward, done, {}

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(backwheel)

            # Render flag 1
            flagx = (
                self.goal_position - self.goal_pos_tolerance - self.min_position
            ) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

            # Render flag 2
            flagx = (
                self.goal_position + self.goal_pos_tolerance - self.min_position
            ) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos - self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(self._height_tangent(pos))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
