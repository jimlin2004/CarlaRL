import itertools

steerActions = [
    -1, -0.75, -0.5, -0.25, -0.1,
    0, 0.1, 0.25, 0.5, 0.75, 1
]

# speedActions = [10, 20, 30]

# actionProducts = [product for product in itertools.product(steerActions, speedActions)]

actionMap = {
    # i: action for i, action in enumerate(actionProducts)
    i: action for i, action in enumerate(steerActions)
}

learningRate = 0.01
batchSize = 16
stateShpae = (128, 128)
episodes = 1000
