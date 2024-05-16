import numpy as np
import pickle
import os.path as osp
import pyflex
from softgym.envs.rope_env import RopeNewEnv
from copy import deepcopy
from softgym.utils.pyflex_utils import random_pick_and_place, center_object

class RopeFlattenEnv(RopeNewEnv):
    def __init__(self, cached_states_path='rope_flatten_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """

        super().__init__(**kwargs)
        self.prev_distance_diff = None
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

    def generate_env_variation(self, num_variations=1, config=None, save_to_file=False, **kwargs):
        """ Generate initial states. Note: This will also change the current states! """
        generated_configs, generated_states = [], []
        if config is None:
            config = self.get_default_config()
        default_config = config            
        for i in range(num_variations):
            config = deepcopy(default_config)
            config['segment'] = self.get_random_rope_seg_num()
            self.set_scene(config)

            self.update_camera('default_camera', default_config['camera_params']['default_camera'])
            config['camera_params'] = deepcopy(self.camera_params)
            self.action_tool.reset([0., -1., 0.])

            random_pick_and_place(pick_num=4, pick_scale=0.005)
            center_object()
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def get_random_rope_seg_num(self):
        return np.random.randint(40, 41)

    def _reset(self):
        config = self.current_config
        self.rope_length = config['segment'] * config['radius'] * 0.5

        # set reward range
        self.reward_max = 0
        rope_particle_num = config['segment'] + 1
        self.key_point_indices = self._get_key_point_idx(rope_particle_num)

        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions().reshape([-1, 4])
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.1, cy])

        # set reward range
        self.reward_max = 0
        self.reward_min = -self.rope_length
        self.reward_range = self.reward_max - self.reward_min

        return self._get_obs()

    def _step(self, action):
        if self.action_mode.startswith('picker'):
            self.action_tool.step(action)
            pyflex.step()
        else:
            raise NotImplementedError
        return

    def _get_endpoint_distance(self):
        pos = pyflex.get_positions().reshape(-1, 4)
        p1, p2 = pos[0, :3], pos[-1, :3]
        return np.linalg.norm(p1 - p2).squeeze()

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """ Reward is the distance between the endpoints of the rope"""
        curr_endpoint_dist = self._get_endpoint_distance()
        curr_distance_diff = -np.abs(curr_endpoint_dist - self.rope_length)
        r = curr_distance_diff
        return r

    def _get_info(self):
        curr_endpoint_dist = self._get_endpoint_distance()
        curr_distance_diff = -np.abs(curr_endpoint_dist - self.rope_length)

        performance = curr_distance_diff
        normalized_performance = (performance - self.reward_min) / self.reward_range

        return {
            'performance': performance,
            'normalized_performance': normalized_performance,
            'end_point_distance': curr_endpoint_dist
        }


    def render(self, mode='rgb_array', hide_picker=False):
        if mode == 'rgb_array':
            if hide_picker:
                shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
                prev_shape_states = shape_states.copy()
                for shape_state in shape_states:
                    shape_state[1] = 1e5
                    shape_state[4] = 1e5
                pyflex.set_shape_states(shape_states)
            img, depth = pyflex.render()
            width, height = self.camera_params['default_camera']['width'], self.camera_params['default_camera']['height']
            img = img.reshape(height, width, 4)[::-1, :, :3]  # Need to reverse the height dimension

            if hide_picker:
                pyflex.set_shape_states(prev_shape_states)
            
            return img
        elif mode == 'human':
            raise NotImplementedError


class RopeFlattenEasyEnv(RopeNewEnv):
    def __init__(self, cached_states_path='rope_flatten_easy_1000_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        
        super().__init__(**kwargs)
        self.prev_distance_diff = None
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)
        print('debuggg cached_states_path ', cached_states_path)

    def generate_env_variation(self, num_variations=1, config=None, save_to_file=False, **kwargs):
        """ Generate initial states. Note: This will also change the current states! """
        generated_configs, generated_states = [], []
        if config is None:
            config = self.get_default_config()
        default_config = config            
        for i in range(num_variations):
            config = deepcopy(default_config)
            config['segment'] = self.get_random_rope_seg_num()
            self.set_scene(config)
            
            self.action_tool.reset([0., -1., 0.])

            self.update_camera('default_camera', default_config['camera_params']['default_camera'])
            config['camera_params'] = deepcopy(self.camera_params)
            self.action_tool.reset([0., -1., 0.])

            random_pick_and_place(pick_num=4, pick_scale=0.005)
            center_object()

            all_particle_pos = pyflex.get_positions().reshape(-1, 4)
            endpoint1_particle_pos = all_particle_pos[0, :3]
            endpoint2_particle_pos = all_particle_pos[-1, :3]
            endpoint1_picker_pos = endpoint1_particle_pos + self.action_tool.picker_radius
            endpoint2_picker_pos = endpoint2_particle_pos + self.action_tool.picker_radius
            
            self.action_tool.set_two_picker_pos([endpoint1_picker_pos, endpoint2_picker_pos])
            self.picked_partcile_idx_1 = 0
            self.picked_partcile_idx_2 = len(all_particle_pos) - 1
            self.action_tool.set_picked_particle(0, self.picked_partcile_idx_1) 
            self.action_tool.set_picked_particle(1, self.picked_partcile_idx_2) 
            
            for _ in range(30):
                pyflex.render()

            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def get_random_rope_seg_num(self):
        return np.random.randint(40, 41)

    def _reset(self):
        config = self.current_config
        self.rope_length = config['segment'] * config['radius'] * 0.5

        # set reward range
        self.reward_max = 0
        rope_particle_num = config['segment'] + 1
        self.key_point_indices = self._get_key_point_idx(rope_particle_num)

        if hasattr(self, 'action_tool'):
            # curr_pos = pyflex.get_positions().reshape([-1, 4])
            # cx, cy = self._get_center_point(curr_pos)
            # self.action_tool.reset([cx, 0.1, cy])
            self.action_tool.reset([0, 0, 0])
            all_particle_pos = pyflex.get_positions().reshape(-1, 4)
            endpoint1_particle_pos = all_particle_pos[0, :3]
            endpoint2_particle_pos = all_particle_pos[-1, :3]
            endpoint1_picker_pos = endpoint1_particle_pos + self.action_tool.picker_radius
            endpoint2_picker_pos = endpoint2_particle_pos + self.action_tool.picker_radius
            
            self.action_tool.set_two_picker_pos([endpoint1_picker_pos, endpoint2_picker_pos])
            self.picked_partcile_idx_1 = 0
            self.picked_partcile_idx_2 = len(all_particle_pos) - 1
            self.action_tool.set_picked_particle(0, self.picked_partcile_idx_1) 
            self.action_tool.set_picked_particle(1, self.picked_partcile_idx_2) 
            
        # set reward range
        self.reward_max = 0
        self.reward_min = -self.rope_length
        self.reward_range = self.reward_max - self.reward_min

        return self._get_obs()


    def _step(self, action):
        if self.action_mode.startswith('picker'):
            action[3] = 1
            action[7] = 1
            self.action_tool.step(action)
            pyflex.step()
        else:
            raise NotImplementedError
        return

    def _get_endpoint_distance(self):
        pos = pyflex.get_positions().reshape(-1, 4)
        p1, p2 = pos[0, :3], pos[-1, :3]
        return np.linalg.norm(p1 - p2).squeeze()

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """ Reward is the distance between the endpoints of the rope"""
        curr_endpoint_dist = self._get_endpoint_distance()
        # curr_distance_diff = -np.abs(curr_endpoint_dist - self.rope_length)
        # r = curr_distance_diff
        r = np.minimum(curr_endpoint_dist, self.rope_length * 1.1)
        return r

    def _get_info(self):
        curr_endpoint_dist = self._get_endpoint_distance()
        # curr_distance_diff = -np.abs(curr_endpoint_dist - self.rope_length)
        # performance = curr_distance_diff
        performance = np.minimum(curr_endpoint_dist, self.rope_length * 1.1)
        
        normalized_performance = (performance - self.reward_min) / self.reward_range

        return {
            'performance': performance,
            'normalized_performance': normalized_performance,
            'end_point_distance': curr_endpoint_dist,
            "success": 1 if performance >= 0.95 * self.rope_length else 0
        }


    def render(self, mode='rgb_array', hide_picker=True):
        if mode == 'rgb_array':
            if hide_picker:
                shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
                prev_shape_states = shape_states.copy()
                for shape_state in shape_states:
                    shape_state[1] = 1e5
                    shape_state[4] = 1e5
                pyflex.set_shape_states(shape_states)
            img, depth = pyflex.render()
            width, height = self.camera_params['default_camera']['width'], self.camera_params['default_camera']['height']
            img = img.reshape(height, width, 4)[::-1, :, :3]  # Need to reverse the height dimension

            if hide_picker:
                pyflex.set_shape_states(prev_shape_states)
            
            return img
        elif mode == 'human':
            raise NotImplementedError
        
        
if __name__ == '__main__':
    env_dict =  {
        'observation_mode': 'key_point',
        'action_mode': 'picker',
        'num_picker': 2,
        'render': True,
        'headless': True,
        'horizon': 40,
        'action_repeat': 5,
        'render_mode': 'cloth',
        'num_variations': 1000,
        'use_cached_states': True,
        'deterministic': False,
        "recording": True
    }

    env = RopeFlattenEasyEnv(cached_states_path='rope_flatten_easy_1000_init_states.pkl', **env_dict)
    env_name = 'RopeFlattenEasy'
    

    from softgym.utils.visualization import save_numpy_as_gif
    SAVE_PATH = 'data/'
    all_videos = []
    # for i in range()
    # obs = env.reset()
    
    # for config_id in range(20):
    #     obs = env.reset(config_id=config_id)

    #     for j in range(50):
    #         pyflex.render()
    #         action = env.action_space.sample()
    #         # action = np.zeros(8)
    #         action[0] = 0.005
    #         action[2] = 0.007
    #         action[3] = 1
    #         action[4] = 0.005
    #         action[6] = 0.008
    #         action[7] = 1
    #         # for _ in range(10):
    #         #     pyflex.step()
    #         obs, _, _, info = env.step(action)

    #     images = env.video_frames
    #     # print('debugging ', images.shape)
    #     # save_numpy_as_gif(np.array(images), osp.join(SAVE_PATH, '{}.gif'.format(env_name)))

