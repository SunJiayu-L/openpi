"""Episode indices for each LIBERO suite, used for single-suite fine-tuning."""

# libero_10: task_index 0-9, 379 episodes
LIBERO_10_EPISODES = list(range(0, 379))

# libero_goal: task_index 10-19, 428 episodes
LIBERO_GOAL_EPISODES = list(range(379, 807))

# libero_object: task_index 20-29, 454 episodes
LIBERO_OBJECT_EPISODES = list(range(807, 1261))

# libero_spatial: task_index 30-39, 432 episodes
LIBERO_SPATIAL_EPISODES = list(range(1261, 1693))

# libero_10 task 8 — "put both moka pots on the stove" (29 eps)
LIBERO_10_TASK8_EPISODES = [
    10, 20, 23, 46, 51, 54, 57, 67, 70, 73, 86, 100, 106, 115, 143,
    149, 179, 187, 194, 204, 214, 270, 283, 288, 306, 314, 316, 341, 376,
]

# libero_10 task 9 — "put the yellow and white mug in the microwave and close it" (34 eps)
LIBERO_10_TASK9_EPISODES = [
    2, 3, 34, 35, 44, 59, 80, 84, 99, 123, 126, 139, 140, 142, 145, 164, 173,
    178, 186, 193, 217, 226, 230, 245, 269, 276, 282, 285, 312, 324, 339, 342, 357, 365,
]

# libero_10 task 8 + task 9 combined (63 eps) — for targeted fine-tuning on weak tasks
LIBERO_10_TASK89_EPISODES = sorted(LIBERO_10_TASK8_EPISODES + LIBERO_10_TASK9_EPISODES)
