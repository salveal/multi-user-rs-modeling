import numpy as np
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

"""
Functions taken from algo_confounding
"""
def transform_relative_to_ideal(train_results, metric_key, model_keys, absolute_measure=True):
    relative_dist = defaultdict(lambda: defaultdict(list))

    if absolute_measure:
        ideal_dist = np.array(train_results[metric_key]["ideal"])
    else:
        model_key = list(train_results[metric_key].keys())[0]
        # zeros for all timsteps
        trials = len(train_results[metric_key][model_key])
        timesteps = len(train_results[metric_key][model_key][0])
        ideal_dist = np.zeros((trials, timesteps))
        relative_dist[metric_key]["ideal"] = ideal_dist

    for model_key in model_keys:
        if model_key is "ideal" and not absolute_measure:
            continue

        abs_dist = np.array(train_results[metric_key][model_key])
        print("abs_dist", abs_dist)
        print("ideal_dist", ideal_dist)
        if absolute_measure:
            abs_dist = abs_dist - ideal_dist
        relative_dist[metric_key][model_key] = abs_dist
    return relative_dist

"""
Custom functions
"""
def graph_metrics_and_models(train_results, metric_keys, model_keys, mean_sigma=0, mult_sd=0, conf_sigma=0, relative_to_ideal=False):
    num_subplots = len(metric_keys)
    plt.figure(figsize=(10, 6 * num_subplots))
    for i in range(len(metric_keys)):
        me = metric_keys[i]
        plt.subplot(len(metric_keys), 1, i+1)
        plt.title(me)
        if me == "rmse":
            plt.yscale('log')
        results = train_results
        metric_independent_from_ideal = not "relative" in me
        if relative_to_ideal:
            results = transform_relative_to_ideal(results, me, model_keys, metric_independent_from_ideal)
        else:
            if not metric_independent_from_ideal:
                first_model_key = list(train_results[me].keys())[0]
                sims = len(results[me][first_model_key])
                iters = len(results[me][first_model_key][0])
                results[me]["ideal"] = np.zeros((sims, iters))
        for j in range(len(model_keys)):
            mo = model_keys[j]
            if not isinstance(results[me][mo], np.ndarray):
                results[me][mo] = np.array(results[me][mo])
            if mean_sigma > 0:
                values = gaussian_filter1d(results[me][mo].mean(axis=0), sigma=mean_sigma)
            else:
                values = results[me][mo].mean(axis=0)
            #print("Metric", me, "model", mo)
            if me == "rmse":
                line = plt.plot(values, label=mo)
            else:
                line = plt.plot(values, label=mo)
            line_color = line[0].get_color()
            if mult_sd > 0:
                std = results[me][mo].std(axis=0)
                timesteps = np.arange(len(std))
                low = values - mult_sd * std
                high = values + mult_sd * std
                if conf_sigma > 0:
                    low = gaussian_filter1d(low, sigma=conf_sigma)
                    high = gaussian_filter1d(high, sigma=conf_sigma)
                plt.fill_between(timesteps, low, high, color = line_color, alpha=0.3)
        plt.legend(facecolor='white', framealpha=1, loc='upper right', bbox_to_anchor=(1.7, 1.0))
        plt.grid(True, linestyle='--', color='gray', linewidth=0.5)

def graph_metrics_models_and_mtu(mtu_results, metric_keys, model_keys, labels, mean_sigma=0, mult_sd=0, conf_sigma=0, relative_to_ideal=False):
    num_subplots = len(metric_keys)
    plt.figure(figsize=(10, 6 * num_subplots))
    for i in range(len(metric_keys)):
        me = metric_keys[i]
        model_keys = list(mtu_results[me].keys())
        plt.subplot(len(metric_keys), 1, i+1)
        plt.subplots_adjust(hspace=0.3)
        plt.title(labels["metrics_to_readable"][me])
        results = mtu_results
        metric_independent_from_ideal = not "relative" in me
        if relative_to_ideal:
            results = transform_relative_to_ideal_mtu(results, me, model_keys, metric_independent_from_ideal)
        else:
            if not metric_independent_from_ideal:
                first_model_key = list(mtu_results[me].keys())[0]
                new_dict = defaultdict(list)
                for k, v in mtu_results[me][first_model_key].items():
                    new_dict[k] = np.zeros(len(v))
                results[me]["ideal"] = new_dict
                model_keys = ["ideal"] + list(results[me].keys())[:-1]
        for j in range(len(model_keys)):
            mo = model_keys[j]
            #print("me", me)
            #print("me", me)
            #print("mo", mo)
            x_axis = [labels["x_axis_results_to_readable"][r] for r in list(results[me][mo].keys())]
            y_axis = list(results[me][mo].values())
            if not isinstance(y_axis, np.ndarray):
                y_axis = np.array(y_axis)
            if mean_sigma > 0:
                values = gaussian_filter1d(y_axis.mean(axis=1), sigma=mean_sigma)
            else:
                #print("y_axis", y_axis)
                values = y_axis.mean(axis=1)
            line = plt.plot(x_axis, values, label=labels["id_to_readable"][mo])
            plt.xticks(rotation=0)
            line_color = line[0].get_color()
            if mult_sd > 0:
                std = y_axis.std(axis=1)
                mtus = np.arange(len(std))
                low = values - mult_sd * std
                high = values + mult_sd * std
                if conf_sigma > 0:
                    low = gaussian_filter1d(low, sigma=conf_sigma)
                    high = gaussian_filter1d(high, sigma=conf_sigma)
                plt.fill_between(mtus, low, high, color = line_color, alpha=0.3)
        plt.legend(facecolor='white', framealpha=1, loc='upper right', bbox_to_anchor=(1.7, 1.0))
        plt.grid(True, linestyle='--', color='gray', linewidth=0.5)

def transform_relative_to_ideal_mtu(train_results, metric_key, model_keys, absolute_measure=True):
    relative_dist = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    if absolute_measure:
        ideal_dist = np.array(list(train_results[metric_key]["ideal"].values()))
    else:
        new_ideal = defaultdict(list)
        model_key = list(train_results[metric_key].keys())[0]
        for k, v in train_results[metric_key][model_key].items():
            ideal_dist = np.zeros(len(v))
            new_ideal[k] = ideal_dist
        relative_dist[metric_key]["ideal"] = new_ideal
        ideal_dist = np.array(list(relative_dist[metric_key]["ideal"].values()))

    for model_key in model_keys:
        if model_key is "ideal" and not absolute_measure:
            continue

        abs_dist = np.array(list(train_results[metric_key][model_key].values()))
        print("abs_dist", abs_dist)
        print("ideal_dist", ideal_dist)
        if absolute_measure:
            abs_dist = abs_dist - ideal_dist
        i=0
        new_defaultdict = defaultdict(list)
        for k, _ in train_results[metric_key][model_key].items():
            new_defaultdict[k] = abs_dist[i]
            i+=1
        relative_dist[metric_key][model_key] = new_defaultdict
    return relative_dist
