# Selects what the program will run.
MODE = "train"

# Enables the graphical museum simulation.
RENDER = True

# Selects the model used in play or evaluate mode.
MODEL = "latest"

# Selects the museum layout.
TOPOLOGY = "maze6x6"

# Controls how strongly the thief avoids recently watched rooms.
BETA = 4.0

# Controls exploration in the guard's Softmax policy.
TAU = 0.6

# Sets the number of training episodes.
TRAIN_EPISODES = 15000

# Sets the policy-gradient learning rate.
LEARNING_RATE = 0.02

# Discounts future rewards during training.
GAMMA = 0.99

# Limits the maximum length of each episode.
MAX_STEPS = 120

# Fixes randomness for reproducible runs.
SEED = 7

# Sets evaluation episodes after training finishes.
EVAL_EPISODES_AFTER_TRAINING = 2000

# Sets evaluation episodes in evaluate mode.
EVALUATE_EPISODES = 1000

# Sets how many episodes are shown in play mode.
PLAY_EPISODES = 5

# Saves the trained model after training.
SAVE_MODEL = True

# Sets the model output path; None creates an automatic name.
SAVE_PATH = None

# Sets how often training progress is printed.
LOG_EVERY = 250

# Sets how often the render updates during training.
RENDER_EVERY = 100

# Sets the metric window used for rolling averages.
ROLLING_WINDOW = 500

# Sets the delay between rendered frames.
RENDER_SPEED = 0.25

# Reward given when the guard catches the thief.
REWARD_CATCH = 1.0

# Reward given when the thief escapes.
REWARD_ESCAPE = -1.0

# Small penalty applied at each step.
STEP_PENALTY = -0.002
