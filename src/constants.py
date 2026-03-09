# Default Retracted Position for Experiments (joint space)
# HOME = [287.87, 79.29, 152.622, 253.799, 281.052, 116.668, 90.08] # This is for original shelving task

HOME = [222.163, 78.754, 197.395, 269.46, 220.621, 84.104, 107.406] # This is for the new shelving task with new shelf
RETRACT = [-1.2589, 1.3840, -1.8535, -1.3778, 2.663, 2.0362, 1.5722]
# HOME = [0.0, 0.26191, -2.2690, 0.0, 3.14158, 0.95982, 1.57082]
# TOP_DOWN = [358.813, 8.906, 179.688, 290.291, 0.668, 258.976, 88.211]
TOP_DOWN = [31.072, 340.426, 160.379, 253.633, 353.481, 271.626, 104.495]

# Controls the speed of the EE.
# CAUTION: ANYTHING OVER 0.5 IS VERY FAST FOR KINOVA
SPEED_CONTROL = 0.5
# SPEED_CONTROL = 0.05

# Stop Scan Threshold (absolute x pos) -- Prevents proximity inaccuracies in depth camera data
STOP_SCAN_THRESHOLD = 0.12

# this threshold is to pop out the goals from goal list when end-effector is close enough to the goal
GOAL_POP_THRESHOLD = 0.05

# Placement Region Thresholds
# Assumes that every y value less than the threshold is a valid "pick" location, and everything greater is a valid "place" location.
PLACEMENT_THRESHOLDS = {
    "familiarity": 0.25,
    "pickandplace": 0.0,
    "shelving": 0.25,
    "deceptive_grasping": 0.25,
    "sorting": -1.0,
}

# This is a set of hard-coded goal locations for the various tasks in VOSA.
# This is assumed to be oracle knowledge that the robot can use to navigate and perform tasks that VOSA does not have access to.
ORACLE_GOAL_SET = {
    'familiarity': {
        'object_locations': [
            [],  # No GT for familiarity task
        ],
        'placement_locations': [
            [0.72, -0.043, 0.09],  # Familiarity placement location
        ]
    },
    'pickandplace': {
        'object_locations': [
            [0.627, 0.314, 0.05],  # Object 1
            [0.606, 0.488, 0.05],  # Object 2
        ],
        'placement_locations': [
            [0.57, 0.072, 0.107],  # Goal 1
            [0.816, -0.312, 0.09],  # Goal 2
        ]
    },

    # In shelving, we only know about one object location
    'shelving': {
        'object_locations': [
            [0.447, 0.351, 0.073],  # Blue bottle GT pick location
            [0.41, 0.519, 0.07]     # Mustard bottle GT pick location
        ],
        'placement_locations': [
            [0.782, 0.142, 0.083], # Bottom right shelf
            # [0.784, 0.315, 0.082], # Bottom middle shelf
            [0.773, 0.469, 0.071], # Bottom left shelf
            [0.78, 0.448, 0.488], # Top left shelf
            [0.794, 0.284, 0.477], # Top middle shelf
            # [0.795, 0.155, 0.48], # Top right shelf

        ]
    },
    
    
    # OLD SHELVING GOALS (old shelf)
    # 'shelving': {
    #     'object_locations': [
    #         [0.814, 0.417, 0.07],  # Object 1
    #     ],
    #     'placement_locations': [
    #         [0.912, -0.189, 0.26],  # Goal 1
    #         [0.898, -0.029, 0.26],  # Goal 2
    #     ]
    # },
    # In deceptive grasp, we do not know about the hidden object, only the front 2 objects
    'deceptive_grasping': {
        'object_locations': [
            [0.613, 0.482, 0.071],  # Object 1
            [0.613, 0.258, 0.116],  # Object 2
        ],
        'placement_locations': []
    },
    'sorting': {
        'object_locations': [
            [0.234, -0.154, 0.105],  # Pasta box
            [0.37, 0.181, 0.052], # Blue bottle
            [0.389, 0.329, 0.035],  # Soda bottle
        ],
        'placement_locations': [
            [0.623, -0.243, 0.195],  # Right Bin
            [0.623, 0.244, 0.195],   # Left Bin
            [0.623, 0.0, 0.195],     # Center Bin
        ]
    }
}
