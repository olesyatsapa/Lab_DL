import numpy as np
import pickle
import argparse
import ConfigSpace as CS
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.ERROR)
from copy import deepcopy

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario

import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker


def create_config_space():
    cs = CS.ConfigurationSpace()

    # TODO: Implement the configuration space
    # Uniformly distributed categorical hyperparameters
    optimizer = CS.CategoricalHyperparameter("optimizer",
                                             ['Adam', 'SGD'],
                                             default_value='Adam')
    cs.add_hyperparameter(optimizer)

    output_activation = CS.CategoricalHyperparameter("output_activation",
                                                     ['softmax'],
                                                     default_value='softmax')
    cs.add_hyperparameter(output_activation)

    learning_rate_schedule = CS.CategoricalHyperparameter("learning_rate_schedule",
                                                          ['ExponentialDecay', 'StepDecay'],
                                                          default_value='ExponentialDecay')
    cs.add_hyperparameter(learning_rate_schedule)

    loss_function = CS.CategoricalHyperparameter("loss_function",
                                                 ['categorical_crossentropy'],
                                                 default_value='categorical_crossentropy')
    cs.add_hyperparameter(loss_function)

    activation = CS.CategoricalHyperparameter("activation",
                                              ['relu', 'tanh'],
                                              default_value='relu')
    cs.add_hyperparameter(activation)

    # Uniformly distributed discrete (Integer) hyperparameters
    StepDecay_epochs_per_step = CS.UniformIntegerHyperparameter("StepDecay_epochs_per_step",
                                                                lower=1,
                                                                default_value=16,
                                                                upper=128,
                                                                log=True)
    cs.add_hyperparameter(StepDecay_epochs_per_step)
    cond = CS.EqualsCondition(StepDecay_epochs_per_step, learning_rate_schedule, 'StepDecay')
    cs.add_condition(cond)

    batch_size = CS.UniformIntegerHyperparameter("batch_size",
                                                 lower=8,
                                                 default_value=16,
                                                 upper=256,
                                                 log=True)
    cs.add_hyperparameter(batch_size)

    num_layers = CS.UniformIntegerHyperparameter("num_layers",
                                                 lower=1,
                                                 default_value=2,
                                                 upper=4)
    cs.add_hyperparameter(num_layers)

    ################
    num_units_0 = CS.UniformIntegerHyperparameter("num_units_0",
                                                  lower=16,
                                                  default_value=32,
                                                  upper=256,
                                                  log=True)
    cs.add_hyperparameter(num_units_0)

    num_units_1 = CS.UniformIntegerHyperparameter("num_units_1",
                                                  lower=16,
                                                  default_value=32,
                                                  upper=256,
                                                  log=True)
    cs.add_hyperparameter(num_units_1)
    cond = CS.GreaterThanCondition(num_units_1, num_layers, 1)
    cs.add_condition(cond)

    num_units_2 = CS.UniformIntegerHyperparameter("num_units_2",
                                                  lower=16,
                                                  default_value=32,
                                                  upper=256,
                                                  log=True)
    cs.add_hyperparameter(num_units_2)
    cond = CS.GreaterThanCondition(num_units_2, num_layers, 2)
    cs.add_condition(cond)

    num_units_3 = CS.UniformIntegerHyperparameter("num_units_3",
                                                  lower=16,
                                                  default_value=32,
                                                  upper=256,
                                                  log=True)
    cs.add_hyperparameter(num_units_3)
    cond = CS.EqualsCondition(num_units_3, num_layers, 4)
    cs.add_condition(cond)

    # Uniformly distributed float hyperparameters

    Adam_final_lr_fraction = CS.UniformFloatHyperparameter('Adam_final_lr_fraction',
                                                           lower=1e-4,
                                                           upper=1e-0,
                                                           default_value=1e-2,
                                                           log=True)

    cs.add_hyperparameter(Adam_final_lr_fraction)
    cond = CS.EqualsCondition(Adam_final_lr_fraction, optimizer, 'Adam')
    cs.add_condition(cond)

    Adam_final_lr = CS.UniformFloatHyperparameter('Adam_final_lr',
                                                  lower=1e-4,
                                                  upper=1e-2,
                                                  default_value=1e-3,
                                                  log=True)

    cs.add_hyperparameter(Adam_final_lr)
    cond = CS.EqualsCondition(Adam_final_lr, optimizer, 'Adam')
    cs.add_condition(cond)

    ################
    SGD_final_lr_fraction = CS.UniformFloatHyperparameter('SGD_final_lr_fraction',
                                                          lower=1e-4,
                                                          upper=1e-0,
                                                          default_value=1e-2,
                                                          log=True)

    cs.add_hyperparameter(SGD_final_lr_fraction)
    cond = CS.EqualsCondition(SGD_final_lr_fraction, optimizer, 'SGD')
    cs.add_condition(cond)

    SGD_final_lr = CS.UniformFloatHyperparameter('SGD_final_lr',
                                                 lower=1e-3,
                                                 upper=0.5,
                                                 default_value=1e-1,
                                                 log=True)

    cs.add_hyperparameter(SGD_final_lr)
    cond = CS.EqualsCondition(SGD_final_lr, optimizer, 'SGD')
    cs.add_condition(cond)

    SGD_momentum = CS.UniformFloatHyperparameter('SGD_momentum',
                                                 lower=0.0,
                                                 upper=0.99,
                                                 default_value=0.9,
                                                 log=False)

    cs.add_hyperparameter(SGD_momentum)
    cond = CS.EqualsCondition(SGD_momentum, optimizer, 'SGD')
    cs.add_condition(cond)

    ##################
    dropout_0 = CS.UniformFloatHyperparameter('dropout_0',
                                              lower=0.0,
                                              upper=0.5,
                                              default_value=0.0,
                                              log=False)

    cs.add_hyperparameter(dropout_0)

    dropout_1 = CS.UniformFloatHyperparameter('dropout_1',
                                              lower=0.0,
                                              upper=0.5,
                                              default_value=0.0,
                                              log=False)

    cs.add_hyperparameter(dropout_1)
    cond = CS.GreaterThanCondition(dropout_1, num_layers, 1)
    cs.add_condition(cond)

    dropout_2 = CS.UniformFloatHyperparameter('dropout_2',
                                              lower=0.0,
                                              upper=0.5,
                                              default_value=0.0,
                                              log=False)

    cs.add_hyperparameter(dropout_2)
    cond = CS.GreaterThanCondition(dropout_2, num_layers, 2)
    cs.add_condition(cond)

    dropout_3 = CS.UniformFloatHyperparameter('dropout_3',
                                              lower=0.0,
                                              upper=0.5,
                                              default_value=0.0,
                                              log=False)

    cs.add_hyperparameter(dropout_3)
    cond = CS.EqualsCondition(dropout_3, num_layers, 4)
    cs.add_condition(cond)

    ############
    l2_reg_0 = CS.UniformFloatHyperparameter('l2_reg_0',
                                             lower=1e-6,
                                             upper=1e-2,
                                             default_value=1e-4,
                                             log=True)

    cs.add_hyperparameter(l2_reg_0)

    l2_reg_1 = CS.UniformFloatHyperparameter('l2_reg_1',
                                             lower=1e-6,
                                             upper=1e-2,
                                             default_value=1e-4,
                                             log=True)

    cs.add_hyperparameter(l2_reg_1)

    cond = CS.GreaterThanCondition(l2_reg_1, num_layers, 1)
    cs.add_condition(cond)

    l2_reg_2 = CS.UniformFloatHyperparameter('l2_reg_2',
                                             lower=1e-6,
                                             upper=1e-2,
                                             default_value=1e-4,
                                             log=True)

    cs.add_hyperparameter(l2_reg_2)

    cond = CS.GreaterThanCondition(l2_reg_2, num_layers, 2)
    cs.add_condition(cond)

    l2_reg_3 = CS.UniformFloatHyperparameter('l2_reg_3',
                                             lower=1e-6,
                                             upper=1e-2,
                                             default_value=1e-4,
                                             log=True)

    cs.add_hyperparameter(l2_reg_3)

    cond = CS.EqualsCondition(l2_reg_3, num_layers, 4)
    cs.add_condition(cond)

    return cs


def objective_function(config, epoch=127, **kwargs):
    # Cast the config to an array such that it can be forwarded to the surrogate
    x = deepcopy(config.get_array())
    x[np.isnan(x)] = -1
    lc = rf.predict(x[None, :])[0]
    c = cost_rf.predict(x[None, :])[0]

    return lc[epoch], {"cost": c, "learning_curve": lc[:epoch].tolist()}


class WorkerWrapper(Worker):
    def compute(self, config, budget, *args, **kwargs):
        cfg = CS.Configuration(cs, values=config)
        loss, info = objective_function(cfg, epoch=int(budget))

        return ({
            'loss': loss,
            'info': {"runtime": info["cost"],
                     "lc": info["learning_curve"]}
        })


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_smac', action='store_true')
    parser.add_argument('--run_hyperband', action='store_true')
    parser.add_argument('--n_iters', default=50, type=int)
    args = vars(parser.parse_args())

    n_iters = args['n_iters']

    cs = create_config_space()
    rf = pickle.load(open("./rf_surrogate_paramnet_mnist.pkl", "rb"))
    cost_rf = pickle.load(open("./rf_cost_surrogate_paramnet_mnist.pkl", "rb"))

    # plot for loss trajectory
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # plot for learning curves
    f2, axarr = plt.subplots(2, sharex=True)


    if args["run_smac"]:

        scenario = Scenario({"run_obj": "quality",
                             "runcount-limit": n_iters,
                             "cs": cs,
                             "deterministic": "true",
                             "output_dir": ""})

        smac = SMAC(scenario=scenario, tae_runner=objective_function)
        smac.optimize()

        # The following lines extract the incumbent strategy and the estimated wall-clock time of the optimization
        rh = smac.runhistory
        incumbents = []
        incumbent_performance = []
        inc = None
        inc_value = 1
        idx = 1
        t = smac.get_trajectory()

        wall_clock_time = []
        cum_time = 0
        for d in rh.data:
            cum_time += rh.data[d].additional_info["cost"]
            wall_clock_time.append(cum_time)
        for i in range(n_iters):

            if idx < len(t) and i == t[idx].ta_runs - 1:
                inc = t[idx].incumbent
                inc_value = t[idx].train_perf
                idx += 1

            incumbents.append(inc)
            incumbent_performance.append(inc_value)



        # TODO: save and plot the wall clock time and the validation of the incumbent after each iteration here
        ax1.plot(wall_clock_time, incumbent_performance,label='SMAC', linewidth=2.0)
        ax1.set_title('SMAC')



        lc_smac = []
        for d in rh.data:
            lc_smac.append(rh.data[d].additional_info["learning_curve"])

        # TODO: save and plot all learning curves here
        for y in lc_smac:
            axarr[0].plot(y, linewidth=2.0)
        axarr[0].set_title('SMAC')


    if args["run_hyperband"]:
        nameserver, ns_port = hpbandster.distributed.utils.start_local_nameserver()

        # starting the worker in a separate thread
        w = WorkerWrapper(nameserver=nameserver, ns_port=ns_port)
        w.run(background=True)

        CG = hpbandster.config_generators.RandomSampling(cs)

        # instantiating Hyperband with some minimal configuration
        HB = hpbandster.HB_master.HpBandSter(
            config_generator=CG,
            run_id='0',
            eta=2,  # defines downsampling rate
            min_budget=1,  # minimum number of epochs / minimum budget
            max_budget=127,  # maximum number of epochs / maximum budget
            nameserver=nameserver,
            ns_port=ns_port,
            job_queue_sizes=(0, 1),
        )
        # runs one iteration if at least one worker is available
        res = HB.run(10, min_n_workers=1)

        # shutdown the worker and the dispatcher
        HB.shutdown(shutdown_workers=True)

        # extract incumbent trajectory and all evaluated learning curves
        traj = res.get_incumbent_trajectory()
        wall_clock_time = []
        cum_time = 0

        for c in traj["config_ids"]:
            cum_time += res.get_runs_by_id(c)[-1]["info"]["runtime"]
            wall_clock_time.append(cum_time)

        lc_hyperband = []
        for r in res.get_all_runs():
            c = r["config_id"]
            lc_hyperband.append(res.get_runs_by_id(c)[-1]["info"]["lc"])

        incumbent_performance = traj["losses"]

        # TODO: save and plot the wall clock time and the validation of the incumbent after each iteration here

        ax2.plot(wall_clock_time, incumbent_performance, label='Hyperband', linewidth=2.0)
        ax2.set_title('Hyperband')

        for y in lc_hyperband:
            axarr[1].plot(y, linewidth=2.0)
        axarr[1].set_title('Hyperband')

    f2.savefig("./plots/learning_curves.png")
    f.savefig("./plots/incumbent_performance.png")
