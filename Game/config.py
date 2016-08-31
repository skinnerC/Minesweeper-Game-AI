# config.py

# configuration for game
game = dict(
    width               = 800, # game width
    height              = 800, # game height
    g_name              = "GAME", # game name
    g_time              = 4000, # game time
    t_time              = 4000, # game testing time
    fps                 = 60, # frame per second
    delay               = 1000, # terminal update delay 20
    n_best              = 5, # number of best agents
    n_agents            = 30, # number of agents
    n_targets           = 20, # number of targets
    s_agent             = 28, # size of an agent
    s_target            = 2, # size of a target
    l_track             = 2., # default speed of left track
    r_track             = 2., # default speed of right track
    r_min               = -0.1, # minimum rotation rate
    r_max               = 0.1, # maximum rotation rate
)

# configuration for image sources
image = dict(
    icon        = "asset/icon.png", # path of icon image file
    best        = "asset/best.png", # path of best image file
    agent       = "asset/agent.png", # path of agent image file
    target      = "asset/target.png", # path of target image file
)

# configuration for neural network
nnet = dict(
    n_inputs            = 4,# number of inputs
    n_outputs           = 2,# number of outputs
    n_h_neurons         = 4# number of neurons in hidden layers
#    bias                = # bias
)

