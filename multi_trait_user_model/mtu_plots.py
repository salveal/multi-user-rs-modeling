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
    print(matplotlib.__version__)
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
