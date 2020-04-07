import gym
from gym import spaces
import numpy as np
import cv2

class CartPole_Pixel(gym.Wrapper):
    """
    Wrapper for getting raw pixel in cartpole env
    observation: 400x400x1 => (Width, Height, Colour-chennel)
    we dispose 100pxl from each side of width to make the frame divisible(Square) in CNN
    """

    def __init__(self, env):
        self.width = 400
        self.height = 400

        gym.Wrapper.__init__(self, env)
        self.env = env.unwrapped
        # self.env.seed(123)  # fix the randomness for reproducibility purpose

        """
        start new thread to deal with getting raw image
        """
        from tf_rl.env.cartpole_pixel import RenderThread
        self.renderer = RenderThread(env)
        self.renderer.start()

    def _pre_process(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = np.expand_dims(frame, -1)
        return frame

    def step(self, ac):
        _, reward, done, info = self.env.step(ac)
        self.renderer.begin_render()  # move screen one step
        observation = self._pre_process(self.renderer.get_screen())

        if done:
            reward = -1.0  # reward at a terminal state
        return observation, reward, done, info

    def reset(self, **kwargs):
        self.env.reset()
        self.renderer.begin_render()  # move screen one step
        return self._pre_process(self.renderer.get_screen())  # overwrite observation by raw image pixels of screen

    def close(self):
        self.renderer.stop()  # terminate the threads
        self.renderer.join()  # collect the dead threads and notice all threads are safely terminated
        if self.env:
            return self.env.close()

class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        from PIL import Image
        return np.array(Image.fromarray(observation).resize(self.shape[0:2]))


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward


# Deprecated
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0], action[1]]
        return action_


class PixelWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(PixelWrapper, self).__init__(env)

    def observation(self, observation):
        observation_ = self.env.render(mode="rgb_array")
        return observation_