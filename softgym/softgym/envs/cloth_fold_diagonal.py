import numpy as np
import pyflex
from copy import deepcopy
from softgym.envs.cloth_env import ClothEnv
from softgym.utils.pyflex_utils import center_object
import copy
import os

class ClothFoldDiagonalEnv(ClothEnv):
    def __init__(self, cached_states_path='cloth_fold_diagnal_init_states.pkl', **kwargs):
        self.fold_group_a = self.fold_group_b = None
        self.init_pos, self.prev_dist = None, None
        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        dirdir_path = os.path.dirname(dir_path)
        self.cached_states_path = os.path.join(dirdir_path, 'cached_initial_states')
        self.goal_particle_pos = np.load(os.path.join(self.cached_states_path, "diagnal_fold_goal.pkl.npy"))
        self._max_episode_steps = self.horizon

    def rotate_particles(self, angle):
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos -= center
        new_pos = pos.copy()
        new_pos[:, 0] = (np.cos(angle) * pos[:, 0] - np.sin(angle) * pos[:, 2])
        new_pos[:, 2] = (np.sin(angle) * pos[:, 0] + np.cos(angle) * pos[:, 2])
        new_pos += center
        pyflex.set_positions(new_pos)

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        particle_radius = self.cloth_particle_radius
        if self.action_mode in ['sawyer', 'franka']:
            cam_pos, cam_angle = np.array([0.0, 1.62576, 1.04091]), np.array([0.0, -0.844739, 0])
        else:
            cam_pos, cam_angle = np.array([-0.0, 0.42, 0]), np.array([0, -90 / 180. * np.pi, 0.])
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [40, 40],
            'ClothStiff': [0.8, 1, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': cam_pos,
                                   'angle': cam_angle,
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'flip_mesh': 0,
            'mass': 1
        }

        return config
    
    def _sample_cloth_size(self):
        return 40, 40

    def generate_env_variation(self, num_variations=2, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 1000  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.2  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()
        # default_config['flip_mesh'] = 1

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']

            self.set_scene(config)
            self.action_tool.reset(np.array([[0., -1., 0.]]))
            pos = pyflex.get_positions().reshape(-1, 4)
            pos[:, :3] -= np.mean(pos, axis=0)[:3]
            if self.action_mode in ['sawyer', 'franka']: # Take care of the table in robot case
                pos[:, 1] = 0.57
            else:
                pos[:, 1] = 0.005
            pos[:, 3] = 1
            pyflex.set_positions(pos.flatten())
            pyflex.set_velocities(np.zeros_like(pos))
            for _ in range(5):  # In case if the cloth starts in the air
                pyflex.step()

            for wait_i in range(max_wait_step):
                pyflex.step()
                self.get_image()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
                    break

            center_object()
            # angle = (np.random.random() - 0.5) * np.pi / 2
            # self.rotate_particles(angle)

            # move picker to one corner
            corner_particle_pos = pyflex.get_positions().reshape(-1, 4)[0, :3]
            picker_pos = corner_particle_pos + self.action_tool.picker_radius
            self.action_tool.set_picker_pos(picker_pos)
            self.picked_partcile_idx = 0
            self.action_tool.set_picked_particle(0, [self.picked_partcile_idx]) # TODO: implement this

            for _ in range(50):
                pyflex.render()

            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states
    
    def get_state(self):
        pos = pyflex.get_positions()
        vel = pyflex.get_velocities()
        shape_pos = pyflex.get_shape_states()
        phase = pyflex.get_phases()
        camera_params = copy.deepcopy(self.camera_params)
        return {'particle_pos': pos, 'particle_vel': vel, 'shape_pos': shape_pos, 'phase': phase, 'camera_params': camera_params,
                'config_id': self.current_config_id, "picked_particle_idx": self.picked_partcile_idx}
    
    def set_state(self, state_dict):
        pyflex.set_positions(state_dict['particle_pos'])
        pyflex.set_velocities(state_dict['particle_vel'])
        pyflex.set_shape_states(state_dict['shape_pos'])
        pyflex.set_phases(state_dict['phase'])
        self.action_tool.set_picked_particle(0, state_dict['picked_particle_idx']) # TODO: implement this
        self.camera_params = copy.deepcopy(state_dict['camera_params'])
        self.update_camera(self.camera_name)

    def set_test_color(self, num_particles):
        """
        Assign random colors to group a and the same colors for each corresponding particle in group b
        :return:
        """
        colors = np.zeros((num_particles))
        rand_size = 30
        rand_colors = np.random.randint(0, 5, size=rand_size)
        rand_index = np.random.choice(range(len(self.fold_group_a)), rand_size)
        colors[self.fold_group_a[rand_index]] = rand_colors
        colors[self.fold_group_b[rand_index]] = rand_colors
        self.set_colors(colors)

    def _reset(self):
        """ Right now only use one initial state. Need to make sure _reset always give the same result. Otherwise CEM will fail."""
        if hasattr(self, 'action_tool'):
            # particle_pos = pyflex.get_positions().reshape(-1, 4)
            # p1, p2, p3, p4 = self._get_key_point_idx()
            # key_point_pos = particle_pos[(p1, p2), :3] # Was changed from from p1, p4.
            # middle_point = np.mean(key_point_pos, axis=0)
            # self.action_tool.reset([middle_point[0], 0.1, middle_point[2]])

            # picker_low = self.action_tool.picker_low
            # picker_high = self.action_tool.picker_high
            # offset_x = self.action_tool._get_pos()[0][0][0] - picker_low[0] - 0.3
            # picker_low[0] += offset_x
            # picker_high[0] += offset_x
            # picker_high[0] += 1.0
            # self.action_tool.update_picker_boundary(picker_low, picker_high)

            self.action_tool.reset(np.array([[0, 0, 0]]))
            corner_particle_pos = pyflex.get_positions().reshape(-1, 4)[0, :3]
            picker_pos = corner_particle_pos + self.action_tool.picker_radius
            self.action_tool.set_picker_pos(picker_pos)
            self.picked_partcile_idx = 0
            self.action_tool.set_picked_particle(0, [self.picked_partcile_idx]) # TODO: implement this


        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]
        x_split = cloth_dimx // 2
        self.fold_group_a = particle_grid_idx[:, :x_split].flatten()
        self.fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()

        colors = np.zeros(num_particles)
        colors[self.fold_group_a] = 1
        # self.set_colors(colors) # TODO the phase actually changes the cloth dynamics so we do not change them for now. Maybe delete this later.

        pyflex.step()
        self.init_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pos_a = self.init_pos[self.fold_group_a, :]
        pos_b = self.init_pos[self.fold_group_b, :]
        self.prev_dist = np.mean(np.linalg.norm(pos_a - pos_b, axis=1))

        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']
        return self._get_obs()

    def _step(self, action):
        self.action_tool.step(action)
        if self.action_mode in ['sawyer', 'franka']:
            print(self.action_tool.next_action)
            pyflex.step(self.action_tool.next_action)
        else:
            pyflex.step()

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """
        The particles are splitted into two groups. The reward will be the minus average eculidean distance between each
        particle in group a and the crresponding particle in group b
        :param pos: nx4 matrix (x, y, z, inv_mass)
        """
        cur_pos  = pyflex.get_positions().reshape(-1, 4)[:, :3]
        return -np.mean(np.linalg.norm(cur_pos - self.goal_particle_pos, axis=1)) * 10

    def _get_info(self):
        # Duplicate of the compute reward function!
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]
        performance = -np.mean(np.linalg.norm(pos - self.goal_particle_pos, axis=1))

        performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance
        normalized_performance = (performance - performance_init) / (0. - performance_init)
        info = {
            'performance': performance,
            'normalized_performance': normalized_performance,
            "success": normalized_performance > 0.9,
        }
        if 'qpg' in self.action_mode:
            info['total_steps'] = self.action_tool.total_steps
        return info

    def _set_to_folded(self):
        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]
        x_split = cloth_dimx // 2
        fold_group_a = particle_grid_idx[:, :x_split].flatten()
        fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()

        curr_pos = pyflex.get_positions().reshape((-1, 4))
        curr_pos[fold_group_a, :] = curr_pos[fold_group_b, :].copy()
        curr_pos[fold_group_a, 1] += 0.05  # group a particle position made tcurr_pos[self.fold_group_b, 1] + 0.05e at top of group b position.

        pyflex.set_positions(curr_pos)
        for i in range(10):
            pyflex.step()
        return self._get_info()['performance']
    
    # Cloth index looks like the following:
    # 0, 1, ..., cloth_xdim -1
    # ...
    # cloth_xdim * (cloth_ydim -1 ), ..., cloth_xdim * cloth_ydim -1

    def _get_key_point_idx(self):
        """ The keypoints are defined as the four corner points of the cloth """
        dimx, dimy = self.current_config['ClothSize']
        # idx_p1 = 0
        # idx_p2 = dimx * (dimy - 1)
        # idx_p3 = dimx - 1
        # idx_p4 = dimx * dimy - 1
        num_particle = 5
        interval_row = dimy // num_particle
        interval_col = dimx // num_particle
        row_idxes = [i * interval_row for i in range(num_particle)] + [dimy - 1]
        col_idxes = [i * interval_col for i in range(num_particle)] + [dimx - 1]
        idxes = []
        for row_idx in row_idxes:
            for col_idx in col_idxes:
                idxes.append(row_idx * dimx + col_idx)

        # import pdb; pdb.set_trace()

        return np.array(idxes)
    

if __name__ == '__main__':
    env_dict = {
        'observation_mode': 'key_point',
        'action_mode': 'pickerplace',
        'num_picker': 1,
        'render': True,
        'headless': False,
        'horizon': 100,
        'action_repeat': 1,
        'render_mode': 'cloth',
        'num_variations': 1,
        'use_cached_states': True,
        'deterministic': True,
        'picker_radius': 0.001,
        'horizon': 1,
        "save_cached_states": True
    }

    env = ClothFoldDiagonalEnv(**env_dict)

    env.reset()
    positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
    max_p, min_p = np.max(positions, axis=0), np.min(positions, axis=0)
    print(max_p)
    print(min_p)


    # image = env.render(hide_picker=True)
    # from matplotlib import pyplot as plt
    # plt.imshow(image)
    # plt.show()
    for _ in range(1):
        all_positions = pyflex.get_positions().reshape(-1, 4)
        target_position = all_positions[-1, [0, 2]]
        _, reward, _, _ = env.step(target_position)

    print(reward)


    # all_positions = pyflex.get_positions().reshape(-1, 4)[:, :3]
    # np.save("/home/yufei/vlm-reward-private/softgym/softgym/cached_initial_states/diagnal_fold_goal.pkl", all_positions)

