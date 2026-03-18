"""Episode indices for each LIBERO suite, used for single-suite fine-tuning."""

# libero_10: task_index 0-9, 379 episodes
LIBERO_10_EPISODES = list(range(0, 379))

# libero_goal: task_index 10-19, 428 episodes
LIBERO_GOAL_EPISODES = list(range(379, 807))

# libero_object: task_index 20-29, 454 episodes
LIBERO_OBJECT_EPISODES = list(range(807, 1261))

# libero_spatial: task_index 30-39, 432 episodes
LIBERO_SPATIAL_EPISODES = list(range(1261, 1693))
