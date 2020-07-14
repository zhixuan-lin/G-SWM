from yacs.config import CfgNode

default = CfgNode({
    'name': 'default',
    'path': './data/',
    'gif_path': './gif/',
    'gif_num': 10,
    'gif_fps': 10,
    'split': {
        'train': 1,
        'val': 1,
        'test': 1
    },
    'split_seeds': {
        'train': 0,
        'val': 1,
        'test': 2
    },
    'render_options': {
        # Whether to round positions to integers values. This will affect the looking
        # Please turn this off if you need smooth and accurate trajectoires
        'round_position': False
    },
    'options': {
        'seqlen': 100,
        'canvas_size': (64, 64),
        'camera_size': (64, 64),
        'maxnum': 15,
        # In general, (type, value)
        # Unless specified, value should be a list of value to be choose from
        # For num_objs, value should be (low, high)
        'num_objs': ('random', (10, 15)),
        'obj_shapes': ('random', ('circle16', 'star16', 'diamond16', 'cross16', 'diamond_rotate16')),
        'obj_sizes': ('random', (6,)),
        'obj_layers': ('random', (0,)),
        'obj_masses': ('random', (1,)),
        # See here https://htmlcolorcodes.com/
        'obj_colors': ('random', ('blue', 'red', 'yellow', 'fuchsia', 'aqua')),
        # How many pixels to move per frame
        'speed': 2,
        'updates_per_second': 10,
        'max_attempts': 1000,
        'interaction': False,
        # For the first frame, whether all objects should be within camera
        'restricted': False,
        'first_nonoverlap': False
    },
})

config_list = {
    'default': CfgNode({}),
    'occlusion_fixed': CfgNode({
        'name': 'occlusion_fixed',
        'gif_num': 5,
        'gif_fps': 10,
        'split': {
            'train': 5000,
            'val': 200,
            'test': 200,
        },
        'split_seeds': {
            'train': 1,
            'val': 2,
            'test': 3
        },
        'render_options': {
            'round_position': False,
        },
        'options': {
            'canvas_size': (64, 64),
            'camera_size': (64, 64),
            'maxnum': 3,
            'num_objs': ('random', (3, 3)),
            'obj_shapes': ('random', ('circle16', 'star16', 'diamond16', 'cross16', 'cross_rotate16')),
            # 'obj_shapes': ('random', ('circle16',)),
            'obj_sizes': ('random', (8.0,)),
            'obj_layers': ('random', (0,)),
            'obj_masses': ('random', (1,)),
            # See here https://htmlcolorcodes.com/
            'obj_colors': ('random_noreplace', ('blue', 'red', 'yellow', 'fuchsia', 'aqua')),
            # How many pixels to move per frame
            'speed': 2,
            'updates_per_second': 10,
            'max_attempts': 1000,
            'interaction': False,
            # For the first frame, whether all objects should be within camera
            'restricted': False,
            'first_nonoverlap': False
        }
    }),
    'balls_occlusion': CfgNode({
        'name': 'balls_occlusion',
        'gif_num': 5,
        'gif_fps': 3,
        'split': {
            'train': 5000,
            'val': 200,
            'test': 200,
        },
        'split_seeds': {
            'train': 1,
            'val': 2,
            'test': 3
        },
        'render_options': {
            'round_position': False,
        },
        'options': {
            'canvas_size': (64, 64),
            'camera_size': (64, 64),
            'maxnum': 3,
            'num_objs': ('random', (3, 3)),
            # 'obj_shapes': ('random', ('circle16', 'star16', 'diamond16', 'cross16', 'cross_rotate16')),
            'obj_shapes': ('random', ('circle16',)),
            'obj_sizes': ('random', (8.0,)),
            'obj_layers': ('random', (0,)),
            'obj_masses': ('random', (1,)),
            # See here https://htmlcolorcodes.com/
            'obj_colors': ('random_noreplace', ('blue', 'red', 'yellow', 'fuchsia', 'aqua')),
            # How many pixels to move per frame
            'speed': 3,
            'updates_per_second': 10,
            'max_attempts': 1000,
            'interaction': False,
            # For the first frame, whether all objects should be within camera
            'restricted': False,
            'first_nonoverlap': False
        }
    }),
    'occlusion_many': CfgNode({
        'name': 'occlusion_many',
        'gif_num': 5,
        'gif_fps': 10,
        'split': {
            'train': 5000,
            'val': 200,
            'test': 200
        },
        'split_seeds': {
            'train': 4,
            'val': 5,
            'test': 6
        },
        'render_options': {
            'round_position': False,
        },
        'options': {
            'canvas_size': (64, 64),
            'camera_size': (64, 64),
            'maxnum': 10,
            'num_objs': ('random', (5, 10)),
            'obj_shapes': ('random', ('circle16', 'star16', 'diamond16', 'cross16', 'cross_rotate16')),
            # 'obj_shapes': ('random', ('circle16',)),
            'obj_sizes': ('random', (6.0,)),
            'obj_layers': ('random', (0,)),
            'obj_masses': ('random', (1,)),
            # See here https://htmlcolorcodes.com/
            'obj_colors': ('random', ('blue', 'red', 'yellow', 'fuchsia', 'aqua')),
            # How many pixels to move per frame
            'speed': 2,
            'updates_per_second': 10,
            'max_attempts': 1000,
            'interaction': False,
            # For the first frame, whether all objects should be within camera
            'restricted': False,
            'first_nonoverlap': False
        }
    }),
    'interaction_fixed': CfgNode({
        'name': 'interaction_fixed',
        'gif_num': 5,
        'gif_fps': 3,
        'split': {
            'train': 5000,
            'val': 200,
            'test': 200
        },
        'split_seeds': {
            'train': 7,
            'val': 8,
            'test': 9
        },
        'render_options': {
            'round_position': False,
        },
        'options': {
            'canvas_size': (64, 64),
            'camera_size': (64, 64),
            'maxnum': 3,
            'num_objs': ('random', (3, 3)),
            'obj_shapes': ('random', ('circle16',)),
            'obj_sizes': ('random', (8.0,)),
            'obj_layers': ('random', (0,)),
            'obj_masses': ('random', (1,)),
            # See here https://htmlcolorcodes.com/
            'obj_colors': ('random', ('blue', 'red', 'yellow', 'fuchsia', 'aqua')),
            # How many pixels to move per frame
            'speed': 3,
            'updates_per_second': 10,
            'max_attempts': 1000,
            'interaction': True,
            # For the first frame, whether all objects should be within camera
            'restricted': False,
            'first_nonoverlap': True
        }
    }),
    'balls_interaction': CfgNode({
        'name': 'balls_interaction',
        'gif_num': 5,
        'gif_fps': 5,
        'split': {
            'train': 5000,
            'val': 200,
            'test': 200
        },
        'split_seeds': {
            'train': 7,
            'val': 8,
            'test': 9
        },
        'render_options': {
            'round_position': False,
        },
        'options': {
            'canvas_size': (64, 64),
            'camera_size': (64, 64),
            'maxnum': 3,
            'num_objs': ('random', (3, 3)),
            'obj_shapes': ('random', ('circle16',)),
            'obj_sizes': ('random', (8.0,)),
            'obj_layers': ('random', (0,)),
            'obj_masses': ('random', (1,)),
            # See here https://htmlcolorcodes.com/
            'obj_colors': ('random', ('blue', 'red', 'yellow', 'fuchsia', 'aqua')),
            # How many pixels to move per frame
            'speed': 3,
            'updates_per_second': 10,
            'max_attempts': 1000,
            'interaction': True,
            # For the first frame, whether all objects should be within camera
            'restricted': False,
            'first_nonoverlap': True
        }
    }),
    'interaction_many': CfgNode({
        'name': 'interaction_many',
        'gif_num': 5,
        'gif_fps': 10,
        'split': {
            'train': 5000,
            'val': 200,
            'test': 200
        },
        'split_seeds': {
            'train': 10,
            'val': 11,
            'test': 12
        },
        'render_options': {
            'round_position': False,
        },
        'options': {
            'canvas_size': (64, 64),
            'camera_size': (64, 64),
            'maxnum': 10,
            'num_objs': ('random', (5, 10)),
            'obj_shapes': ('random', ('circle16',)),
            'obj_sizes': ('random', (5.0,)),
            'obj_layers': ('random', (0,)),
            'obj_masses': ('random', (1,)),
            # See here https://htmlcolorcodes.com/
            'obj_colors': ('random', ('blue', 'red', 'yellow', 'fuchsia', 'aqua')),
            # How many pixels to move per frame
            'speed': 2,
            'updates_per_second': 10,
            'max_attempts': 1000,
            'interaction': True,
            # For the first frame, whether all objects should be within camera
            'restricted': False,
            'first_nonoverlap': True
        }
    }),
    'two_layer': CfgNode({
        'name': 'two_layer',
        'gif_num': 5,
        'gif_fps': 10,
        'split': {
            
            'train': 5000,
            'val': 200,
            'test': 200
        },
        'split_seeds': {
            'train': 13,
            'val': 14,
            'test': 15
        },
        'render_options': {
            'round_position': False,
        },
        'options': {
            'canvas_size': (64, 64),
            'camera_size': (64, 64),
            'maxnum': 6,
            'num_objs': ('random', (6, 6)),
            'obj_shapes': ('random', ('circle16',)),
            'obj_sizes': ('random', (8.0,)),
            'obj_layers': ('fixed', (0,0,0,1,1,1)),
            'obj_masses': ('random', (1,)),
            # See here https://htmlcolorcodes.com/
            'obj_colors': ('fixed', ('blue', 'blue', 'blue', 'red', 'red', 'red')),
            # How many pixels to move per frame
            'speed': 2,
            'updates_per_second': 10,
            'max_attempts': 1000,
            'interaction': True,
            # For the first frame, whether all objects should be within camera
            'restricted': False,
            'first_nonoverlap': True
        }
    }),
    'balls_two_layer': CfgNode({
        'name': 'balls_two_layer',
        'gif_num': 5,
        'gif_fps': 5,
        'split': {
            'train': 5000,
            'val': 200,
            'test': 200
        },
        'split_seeds': {
            'train': 13,
            'val': 14,
            'test': 15
        },
        'render_options': {
            'round_position': False,
        },
        'options': {
            'canvas_size': (64, 64),
            'camera_size': (64, 64),
            'maxnum': 6,
            'num_objs': ('random', (6, 6)),
            'obj_shapes': ('random', ('circle16',)),
            'obj_sizes': ('random', (8.0,)),
            'obj_layers': ('fixed', (0, 0, 0, 1, 1, 1)),
            'obj_masses': ('random', (1,)),
            # See here https://htmlcolorcodes.com/
            'obj_colors': ('fixed', ('blue', 'blue', 'blue', 'red', 'red', 'red')),
            # How many pixels to move per frame
            'speed': 3,
            'updates_per_second': 10,
            'max_attempts': 1000,
            'interaction': True,
            # For the first frame, whether all objects should be within camera
            'restricted': False,
            'first_nonoverlap': True
        }
    }),
    'occlusion_discovery_many': CfgNode({
        'name': 'occlusion_discovery_many',
        'gif_num': 5,
        'gif_fps': 10,
        'split': {
            'train': 5000,
            'val': 200,
            'test': 200
        },
        'split_seeds': {
            'train': 16,
            'val': 17,
            'test': 18
        },
        'render_options': {
            'round_position': False,
        },
        'options': {
            'canvas_size': (96, 96),
            'camera_size': (64, 64),
            'maxnum': 15,
            'num_objs': ('random', (15, 15)),
            'obj_shapes': ('random', ('circle16', 'star16', 'diamond16', 'cross16', 'cross_rotate16')),
            # 'obj_shapes': ('random', ('circle16',)),
            'obj_sizes': ('random', (5.0,)),
            'obj_layers': ('random', (0,)),
            'obj_masses': ('random', (1,)),
            # See here https://htmlcolorcodes.com/
            'obj_colors': ('random', ('blue', 'red', 'yellow', 'fuchsia', 'aqua')),
            # How many pixels to move per frame
            'speed': 3,
            'updates_per_second': 10,
            'max_attempts': 1000,
            'interaction': False,
            # For the first frame, whether all objects should be within camera
            'restricted': False,
            'first_nonoverlap': False
        }
    }),
    'occlusion_discovery_many_fast': CfgNode({
        'name': 'occlusion_discovery_many_fast',
        'gif_num': 5,
        'gif_fps': 5,
        'split': {
            'train': 1,
            'val': 1,
            'test': 1
        },
        'split_seeds': {
            'train': 16,
            'val': 17,
            'test': 18
        },
        'render_options': {
            'round_position': False,
        },
        'options': {
            'canvas_size': (96, 96),
            'camera_size': (64, 64),
            'maxnum': 15,
            'num_objs': ('random', (10, 10)),
            'obj_shapes': ('random', ('circle16', 'star16', 'diamond16', 'cross16', 'cross_rotate16')),
            # 'obj_shapes': ('random', ('circle16',)),
            'obj_sizes': ('random', (7.0,)),
            'obj_layers': ('random', (0,)),
            'obj_masses': ('random', (1,)),
            # See here https://htmlcolorcodes.com/
            'obj_colors': ('random', ('blue', 'red', 'yellow', 'fuchsia', 'aqua')),
            # How many pixels to move per frame
            'speed': 3,
            'updates_per_second': 10,
            'max_attempts': 1000,
            'interaction': False,
            # For the first frame, whether all objects should be within camera
            'restricted': False,
            'first_nonoverlap': False
        }
    }),
    'two_layer_many': CfgNode({
        'name': 'two_layer_many',
        'gif_num': 5,
        'gif_fps': 3,
        'split': {
            
            'train': 5000,
            'val': 200,
            'test': 200
        },
        'split_seeds': {
            'train': 19,
            'val': 20,
            'test': 21
        },
        'render_options': {
            'round_position': False,
        },
        'options': {
            'canvas_size': (96, 96),
            'camera_size': (64, 64),
            'maxnum': 20,
            'num_objs': ('random', (20, 20)),
            'obj_shapes': ('random', ('circle16',)),
            'obj_sizes': ('random', (6.0,)),
            'obj_layers': ('fixed', (0,) * 10 + (1,) * 10),
            'obj_masses': ('random', (1,)),
            # See here https://htmlcolorcodes.com/
            'obj_colors': ('fixed', ('blue',) * 10 + ('red',) * 10),
            # How many pixels to move per frame
            'speed': 3,
            'updates_per_second': 10,
            'max_attempts': 1000,
            'interaction': True,
            # For the first frame, whether all objects should be within camera
            'restricted': False,
            'first_nonoverlap': True
        }
    }),
    'balls_two_layer_dense': CfgNode({
        'name': 'balls_two_layer_dense',
        'gif_num': 5,
        'gif_fps': 5,
        'split': {
            
            'train': 5000,
            'val': 200,
            'test': 200
        },
        'split_seeds': {
            'train': 19,
            'val': 20,
            'test': 21
        },
        'render_options': {
            'round_position': False,
        },
        'options': {
            'canvas_size': (64, 64),
            'camera_size': (64, 64),
            'maxnum': 20,
            'num_objs': ('random', (16, 16)),
            'obj_shapes': ('random', ('circle16',)),
            'obj_sizes': ('random', (5.0,)),
            'obj_layers': ('fixed', (0,) * 8 + (1,) * 8),
            'obj_masses': ('random', (1,)),
            # See here https://htmlcolorcodes.com/
            'obj_colors': ('fixed', ('blue',) * 8 + ('red',) * 8),
            # How many pixels to move per frame
            'speed': 3,
            'updates_per_second': 10,
            'max_attempts': 1000,
            'interaction': True,
            # For the first frame, whether all objects should be within camera
            'restricted': False,
            'first_nonoverlap': True
        }
    }),
}
