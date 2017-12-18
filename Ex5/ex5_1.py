import pickle
import numpy as np
import copy

import sys


rf = pickle.load(open("./rf_surrogate_cnn.pkl", "rb"))
cost_rf = pickle.load(open("./rf_cost_surrogate_cnn.pkl", "rb"))
rnd=np.random.RandomState()


def objective_function(x, epoch=40):
    """
        Function wrapper to approximate the validation error of the hyperparameter configurations x
        by the prediction of a surrogate regression model,
        which was trained on the validation error of randomly sampled hyperparameter configurations.
        The original surrogate predicts the validation error after a given epoch.
        Since all hyperparameter configurations were trained for a total amount of
        40 epochs, we will query the performance after epoch 40.
    """

    # Normalize all hyperparameter to be in [0, 1]
    x_norm = copy.deepcopy(x)
    x_norm[0] = (x[0] - (-6)) / (0 - (-6))
    x_norm[1] = (x[1] - 32) / (512 - 32)
    x_norm[2] = (x[2] - 4) / (10 - 4)
    x_norm[3] = (x[3] - 4) / (10 - 4)
    x_norm[4] = (x[4] - 4) / (10 - 4)

    x_norm = np.append(x_norm, epoch)
    y = rf.predict(x_norm[None, :])[0]

    return y


def runtime(x, epoch=40):
    """
        Function wrapper to approximate the runtime of the hyperparameter configurations x.
    """

    # Normalize all hyperparameter to be in [0, 1]
    x_norm = copy.deepcopy(x)
    x_norm[0] = (x[0] - (-6)) / (0 - (-6))
    x_norm[1] = (x[1] - 32) / (512 - 32)
    x_norm[2] = (x[2] - 4) / (10 - 4)
    x_norm[3] = (x[3] - 4) / (10 - 4)
    x_norm[4] = (x[4] - 4) / (10 - 4)

    x_norm = np.append(x_norm, epoch)
    y = cost_rf.predict(x_norm[None, :])[0]

    return y


def random_search(objective_function, lower, upper, num_iterations=50):

    incumbents = []
    incumbents_values = []

    X = []
    Y = []
    x_best=[]
    f_min=0

    for i in range(num_iterations):

        new_x = rnd.uniform(lower, upper)
        new_y = objective_function(new_x)

        X.append(new_x.tolist())
        Y.append(new_y)

        best_idx = np.argmin(Y)
        x_best = X[best_idx]
        f_min = Y[best_idx]

        incumbents.append(x_best)
        incumbents_values.append(f_min)

    results = dict()
    results["x_opt"] = x_best
    results["f_opt"] = f_min
    results["incumbents"] = [inc for inc in incumbents]
    results["incumbent_values"] = [val for val in incumbents_values]
    results["X"] = [x for x in X]
    results["y"] = [y for y in Y]
    return results


lower = np.array([-6, 32, 4, 4, 4])
upper = np.array([0, 512, 10, 10, 10])

orig_stdout = sys.stdout

##objective_function
f = open('rs_custom_out.txt', 'w')
sys.stdout = f
s = []
for i in range(10):
    res = random_search(objective_function, lower, upper, num_iterations=50)
    print("Iteration ", str(i))
    print("\n")
    print(res)
    s.append(res["incumbent_values"])
    print("\n" + 50 * "*" + "\n\n")
print(s)
print("\n" + 50 * "*" + "\n\n")
print("\n" + 50 * "*" + "\n\n")
print(np.mean(s,axis=0))
sys.stdout = orig_stdout
f.close()


##runtime
f = open('rs_custom_out_runtime.txt', 'w')
sys.stdout = f
res = random_search(runtime, lower, upper, num_iterations=50)
print(res)
print("\n" + 50 * "*" + "\n\n")
print(res["incumbent_values"])


sys.stdout = orig_stdout
f.close()
