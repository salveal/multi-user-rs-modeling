import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
from trecs.models import (
    ContentFiltering,
    PopularityRecommender,
    SocialFiltering,
)
from trecs.random import Generator
from mtu_creators import NewItemFactory
from mtu_metrics import (
    InteractionTracker,
    CumulativeListDecision,
    TraitTracker,
    UtilityTracker,
    RankTracker,
    UtilityRankTracker,
)
from mtu_rs import (
    IdealRecommender,
    ChaneyContent,
    RandomRecommender,
    ContentFilteringWithTags,
    RandomRecommenderPatched,
    ImplicitMF,
)
from mtu_users import MultiTraitUsers
from mtu_utils import (
    mu_sigma_to_alpha_beta,
    gen_social_network,
    perfect_scores,
    exclude_new_items,
    get_sim_users_pairs,
    gen_social_network_1_attr,
    distances_from_users_to_mean_user,
    measures_of_distances_from_consumed_items_to_mean_item,
    interleave_new_items as startup_interleave_fn
)
import argparse
import os
import errno
import pickle as pkl
import pprint
import warnings
import types
from multiprocessing import Process, Manager, Pool
warnings.simplefilter("ignore")


def sort_by_second_value(x):
    return x[1]

def string_to_unique_int(s):
    return int.from_bytes(s.encode('utf-8'), byteorder='big')

def interleave_new_items(new_items_per_iter, generator):
    """ Chooses the most recent, newest items to interleave
        with the recommendation set. This custom interleaving method
        ensures that all of the most recently created items (i.e.,
        the newest items) are the ones interleaved with the recommendations.
    """
    def interleaving_fn(k, item_indices):
        num_users = item_indices.shape[0]
        indices = item_indices[:, -new_items_per_iter:]
        values = generator.random(indices.shape)
        order = values.argsort(axis=1) # randomly sort indices within rows
        rows = np.tile(np.arange(num_users).reshape((-1, 1)), indices.shape[1])
        return indices[rows, order][:,-k:]
    return interleaving_fn

def set_set_num_items_per_iter(new_items_per_iter, random_items_per_iter):
    """
    Added the new_items_per_iter so that random items per iteration
    can be distinct from new items each iteration.
    """
    def set_num_items_per_iter(self, num_items_per_iter):
            """Change the number of items that will be shown
            to each user per iteration. This function had to
            be made since when setting T-RECS BaseRecommender
            attribute repeated_items parameter to False it messed
            this part and it was a nightmare to debug.
            """
            if num_items_per_iter == "all":
                self.num_items_per_iter = self.num_items - (self.indices < 0).sum(axis=1).max() - new_items_per_iter + random_items_per_iter
                self.expand_items_per_iter = True
                #print("self.num_items_per_iter", self.num_items_per_iter)
            else:
                self.expand_items_per_iter = False
                self.num_items_per_iter = num_items_per_iter
    return set_num_items_per_iter

def startup_and_train(self, timesteps=50, no_new_items=False, **kwargs):
    if self.is_verbose():
        self.log("Startup -- recommend random items")
    self.run(timesteps, startup=True, train_between_steps=False, no_new_items=no_new_items, **kwargs)
    self.train()

def init_sim_state(true_scores, noisy_scores, item_representation, social_network, traits, seed, args):
    u = MultiTraitUsers(
        num_users=args["num_users"],
        true_scores=true_scores,
        known_scores=np.copy(noisy_scores),
        social_network=social_network,
        traits=traits,
        size=(args["num_users"], args["num_attrs"]),
        attention_exp=-0.8,
        repeat_interactions=False,
        seed=seed
    )
    #print("true_scores pre model init", u.true_scores)
    item_factory = NewItemFactory(np.copy(item_representation), args["new_items_per_iter"])
    empty_item_set = np.array([]).reshape((args["num_attrs"], 0))
    return u, item_factory, empty_item_set

def run_ideal_sim(thread_id, user_prefs, true_utils, init_params, args, lock=None):
    rng = Generator(init_params["seed"])
    u, item_factory, empty_items = init_sim_state(**init_params, args=args)
    if not args["repeated_training"]:
        post_startup_num_items_per_iter = (args["startup_iters"] + 1) * args["new_items_per_iter"]
    else:
        post_startup_num_items_per_iter = "all"

    model_params = {
        "creators": item_factory,
        "actual_item_representation": empty_items,
        "actual_user_representation": u,
        "num_items_per_iter": "all",
        "num_users": args["num_users"],
        "num_items": 0,
        "interleaving_fn": startup_interleave_fn(rng),
        "verbose": True,
        "seed": init_params["seed"]
    }
    model_params["user_representation"] = user_prefs
    model_params["score_fn"] = perfect_scores(args["new_items_per_iter"], true_utils)

    run_params = {
        "random_items_per_iter": args["rand_items_per_iter"], # THIS VALUE SHOULDN'T BE GREATER THAN args["new_items_per_iter"]
        "vary_random_items_per_iter": False,
        "repeated_items": False,
        "disable_tqdm": True,
    }

    m = IdealRecommender(**model_params)

    m.set_num_items_per_iter = types.MethodType(set_set_num_items_per_iter(0, 0), m)
    m.startup_and_train = types.MethodType(startup_and_train, m)

    metrics = [
        InteractionTracker(),
        CumulativeListDecision(),
        UtilityTracker(),
        TraitTracker(),
        RankTracker(),
        UtilityRankTracker()
    ]
    m.add_metrics(*metrics)
    m.add_state_variable(m.users.actual_user_profiles)
    m.add_state_variable(m.users_hat)
    m.startup_and_train(timesteps=args["startup_iters"], no_new_items=False, repeated_items=run_params["repeated_items"], disable_tqdm=True)
    #m.set_num_items_per_iter = types.MethodType(set_set_num_items_per_iter(args["new_items_per_iter"], run_params["random_items_per_iter"]), m)
    m.interleaving_fn = interleave_new_items(args["new_items_per_iter"], rng)
    m.set_num_items_per_iter(post_startup_num_items_per_iter)
    m.run(timesteps=args["total_iters"] - args["startup_iters"], train_between_steps=args["repeated_training"], **run_params)
    m.close()

    metric_results = dict()
    for me in metrics:
        measurements = m.get_measurements()[me.name][1:]
        metric_results[me.name] = measurements
    
    with lock:
        raw_data_file = os.path.join(args["output_dir"], "raw_data_file.pkl")
        try:
            with open(raw_data_file, "rb") as file:
                existing_data = pkl.load(file)  # Load the existing list
        except FileNotFoundError:
            # If the file doesn't exist, start with an empty list
            existing_data = []
        existing_data.append([thread_id[0], thread_id[1], metric_results])
        with open(raw_data_file, "wb") as file:
            pkl.dump(existing_data, file)


def run_sim(thread_id, item_attrs, init_params, args, model=ContentFiltering, user_representation=None, lock=None):
    rng = Generator(init_params["seed"])
    u, item_factory, empty_items = init_sim_state(**init_params, args=args)
    if not args["repeated_training"]:
        post_startup_num_items_per_iter = (args["startup_iters"] + 1) * args["new_items_per_iter"]
    else:
        post_startup_num_items_per_iter = "all"
        
    model_params = {
        "creators": item_factory,
        "actual_item_representation": empty_items,
        "actual_user_representation": u,
        "num_items_per_iter": "all",
        "num_users": args["num_users"],
        "num_items": 0,
        "interleaving_fn": startup_interleave_fn(rng),
        "verbose": True,
        "seed": init_params["seed"]
    }
    #model_params["score_fn"] = exclude_new_items(args["new_items_per_iter"])
    if model != PopularityRecommender:
        if model != SocialFiltering:
            if model != ImplicitMF:
                model_params["num_attributes"] = args["num_attrs"]
                if not (model == RandomRecommender or model == RandomRecommenderPatched):
                    model_params["item_rep_for_threshold"] = item_attrs
            else:
                model_params["num_latent_factors"] = args["num_attrs"] if args["num_attrs"] > 2 else 10
        else:
            model_params["user_representation"] = user_representation
    
    run_params = {
        "random_items_per_iter": args["rand_items_per_iter"], # THIS VALUE SHOULDN'T BE GREATER THAN args["new_items_per_iter"]
        "vary_random_items_per_iter": False,
        "repeated_items": False,
        "disable_tqdm": True,
    }

    m = model(**model_params)

    m.set_num_items_per_iter = types.MethodType(set_set_num_items_per_iter(0, 0), m)
    m.startup_and_train = types.MethodType(startup_and_train, m)

    metrics = [
        InteractionTracker(),
        CumulativeListDecision(),
        UtilityTracker(),
        TraitTracker(),
        RankTracker(),
        UtilityRankTracker(),
    ]
    m.add_metrics(*metrics)
    m.add_state_variable(m.users.actual_user_profiles)
    m.add_state_variable(m.users_hat)
    m.startup_and_train(timesteps=args["startup_iters"], no_new_items=False, repeated_items=run_params["repeated_items"], disable_tqdm=True)
    m.interleaving_fn = interleave_new_items(args["new_items_per_iter"], rng)
    m.set_num_items_per_iter(post_startup_num_items_per_iter)
    m.run(timesteps=args["total_iters"] - args["startup_iters"], train_between_steps=args["repeated_training"], **run_params)
    m.close()

    metric_results = dict()
    for me in metrics:
        measurements = m.get_measurements()[me.name][1:]
        metric_results[me.name] = measurements
    
    with lock:
        raw_data_file = os.path.join(args["output_dir"], "raw_data_file.pkl")
        try:
            with open(raw_data_file, "rb") as file:
                existing_data = pkl.load(file)  # Load the existing list
        except FileNotFoundError:
            # If the file doesn't exist, start with an empty list
            existing_data = []
        existing_data.append([thread_id[0], thread_id[1], metric_results])
        with open(raw_data_file, "wb") as file:
            pkl.dump(existing_data, file)


def simulation_function(t, world_params, true_user_items_per_sim, args, lock, lock2):
    print("Starting simulation", t)
    users, true_utils, known_utils, items, social_networks, mo_to_object, model_keys = world_params.values()

    sim_seed = (args["seed"] + string_to_unique_int(t[0])%10000)*(len(model_keys)*t[1] + 1) 

    true_prefs = users[t[1]]
    true_scores = true_utils[t[1]]
    noisy_scores = known_utils[t[1]]
    item_representation = items[t[1]].T
    social_network = social_networks[t[1]]

    init_params = {
        "seed": sim_seed,
        "true_scores": true_scores,
        "noisy_scores": noisy_scores,
        "item_representation": item_representation,
        "social_network": social_network,
        "traits": np.array([
            args["t_sr"],
            args["t_rand"],
            args["t_pop"],
            args["t_trend"],
            args["t_new"],
            args["t_social"],
            args["t_ideal"]
        ]),
    }

    with lock:
        if len(true_user_items_per_sim[t[1]]) == 0:
            true_user_items_per_sim[t[1]].append(true_prefs)
            true_user_items_per_sim[t[1]].append(item_representation)

    print("Running " + t[0] + " simulation " + str(t[1]) + ":")

    run_function = mo_to_object[t[0]][0]
    if t[0] == "ideal":
        run_function(t, true_prefs, true_scores, init_params, args, lock2)
    else:
        model_class = mo_to_object[t[0]][1]
        if "content" in t[0]:
            model_class = mo_to_object[t[0]][1](mo_to_object[t[0]][2])
        mo_to_object[t[0]][0](t, item_representation, init_params, args, model_class, social_network, lock2)


if __name__ == "__main__":

    #######################
    ##  parse arguments  ##
    #######################

    parser = argparse.ArgumentParser(description='running Chaney replication simulations')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_users', type=int, default=100)
    parser.add_argument('--num_items', type=int, default=10000)
    parser.add_argument('--num_attrs', type=int, default=20)
    parser.add_argument('--num_sims', type=int, default=5)
    parser.add_argument('--mu_n', type=float, default=0.98)
    parser.add_argument('--sigma', type=float, default=1e-5)
    parser.add_argument('--new_items_per_iter', type=int, default=10)
    parser.add_argument('--rand_items_per_iter', type=int, default=10)
    parser.add_argument('--repeated_training', dest='repeated_training', action='store_true')
    parser.add_argument('--single_training', dest='repeated_training', action='store_false')
    parser.add_argument('--total_iters', type=int, default=100)
    parser.add_argument('--t_sr', type=float, default=1.0)
    parser.add_argument('--t_rand', type=float, default=0.0)
    parser.add_argument('--t_pop', type=float, default=0.0)
    parser.add_argument('--t_trend', type=float, default=0.0)
    parser.add_argument('--t_new', type=float, default=0.0)
    parser.add_argument('--t_social', type=float, default=0.0)
    parser.add_argument('--t_ideal', type=float, default=0.0)

    parsed_args = parser.parse_args()
    args = vars(parsed_args)
    args["startup_iters"] = 50 if not args["repeated_training"] else 10


    ############################
    ##  create output folder  ##
    ############################

    print("Creating experiment output folder... 💻")
    try:
        os.makedirs(args["output_dir"])
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    # write experiment arguments to file
    with open(os.path.join(args["output_dir"], "args.txt"), "w") as log_file:
        pprint.pprint(args, log_file)


    ############################################
    ##  sample initial user / item  profiles  ##
    ############################################
    
    print("Sampling initial user and item profiles... 🔬")

    rng = Generator(args["seed"])

    user_params = rng.dirichlet(np.ones(args["num_attrs"]), size=args["num_sims"]) * 10
    item_params = rng.dirichlet(np.ones(args["num_attrs"]) * 100, size=args["num_sims"]) * 0.1

    users, items, true_utils, known_utils, social_networks = [], [], [], [], []

    for sim_index in range(args["num_sims"]):
        # generate user preferences and item attributes
        if args["num_attrs"] > 1:
            user_prefs = rng.dirichlet(user_params[sim_index, :], size=args["num_users"])
            item_attrs = rng.dirichlet(item_params[sim_index, :], size=args["num_items"])
        else:
            user_prefs = rng.normal(0, 0.1, size=args["num_users"]).reshape((-1, 1))
            item_attrs = rng.normal(0, 0.1, size=args["num_items"]).reshape((-1, 1))

        # mean of the utility distribution
        true_utils_mu = user_prefs @ item_attrs.T
        true_utils_mu = np.clip(true_utils_mu, 1e-9, None) # avoid numerical stability issues

        alphas, betas = mu_sigma_to_alpha_beta(true_utils_mu, args["sigma"])
        true_util = rng.beta(alphas, betas, size=(args["num_users"], args["num_items"]))
        # assert support of true utilities; should be within 0 and 1
        assert true_util.min() >= 0 and true_util.max() <= 1

        # calculate known utility for each user
        alpha, beta = mu_sigma_to_alpha_beta(args["mu_n"], 1e-1) # parameters for beta function governing percentage of utility known to users
        perc_known = rng.beta(alpha, beta, size=(args["num_users"], args["num_items"]))
        known_util = true_util * perc_known

        # add all synthetic data to list
        users.append(user_prefs)
        if args["num_attrs"] > 1:
            social_networks.append(gen_social_network(user_prefs))
        else:
            soc = gen_social_network_1_attr(user_prefs)
            social_networks.append(soc)
        items.append(item_attrs)
        true_utils.append(true_util)
        known_utils.append(known_util)
    
    ################################################
    ##  checking user-item utility distributions  ##
    ################################################

    #Vsui = np.array(true_utils)

    #item_values = np.sort(Vsui.sum(axis=1), axis=1)[:,::-1]
    #plt.grid()
    #plt.ylim(-5, 70)
    #for v in item_values:
    #    plt.plot(v, marker=',', markersize=1.5)
    #plt.show()

    #mean_item_values = item_values.mean(axis=0)
    #plt.grid()
    #plt.ylim(-5, 70)
    #plt.plot(mean_item_values, marker=',', markersize=1.5, c='k')
    #std = item_values.std(axis=0)
    #mtus = np.arange(len(std))
    #low = mean_item_values - 1. * std
    #high = mean_item_values + 1. * std
    #low = gaussian_filter1d(low, sigma=0.1)
    #high = gaussian_filter1d(high, sigma=0.1)
    #plt.fill_between(mtus, low, high, color='k', alpha=0.3)
    #plt.show()

    #user_values = np.sort(Vsui.sum(axis=2), axis=1)[:,::-1]
    #plt.grid()
    #plt.ylim(350, 630)
    #for v in user_values:
    #    plt.plot(v, marker=',', markersize=1.5)
    #plt.show()

    ###########################
    ##  running simulations  ##
    ###########################

    mo_to_object = {
        #"ideal" : [run_ideal_sim],
        #"content_5": [run_sim, ContentFilteringWithTags, 5],
        #"content_10": [run_sim, ContentFilteringWithTags, 10],
        #"pop": [run_sim, PopularityRecommender],
        #"mf": [run_sim, ImplicitMF],
        #"sf": [run_sim, SocialFiltering],
        "random" : [run_sim, RandomRecommender],
        "random_patched": [run_sim, RandomRecommenderPatched],
    }

    model_keys = list(mo_to_object.keys())
    
    world_params = {
        "users": users,
        "true_utils" : true_utils,
        "known_utils" : known_utils,
        "items": items,
        "social_networks": social_networks,
        "mo_to_object": mo_to_object,
        "model_keys": model_keys
    }

    metric_list = [
        "interaction_history",
        "decisions",
        "utility_history",
        "trait_history",
        "rank_history",
        "utility_rank_history",
        ]
    result_metrics = {k: defaultdict(list) for k in ["user_prefs", "item_attrs", *metric_list]}

    manager = Manager()
    true_user_items_per_sim = manager.list([manager.list() for _ in range(args["num_sims"])])
    lock = manager.Lock()
    lock2 = manager.Lock()
    threads = list([mo, i] for i in range(args["num_sims"]) for mo in model_keys)

    raw_data_file = os.path.join(args["output_dir"], "raw_data_file.pkl")
    if os.path.exists(raw_data_file):
        os.remove(raw_data_file)

    print("Running simulations...👟")
    task_args = [(t, world_params, true_user_items_per_sim, args, lock, lock2) for t in threads]
    with Pool(processes=4) as pool:
        pool.starmap(simulation_function, task_args)

    print("Getting results...")
    for a in list(true_user_items_per_sim):
        result_metrics["user_prefs"]["all"].append(a[0])
        result_metrics["item_attrs"]["all"].append(a[1])

    unsorted_results = pkl.load(open(raw_data_file, "rb"))
    unsorted_results.sort(key=sort_by_second_value)
    for r in unsorted_results:
        for me_key, me_value in r[2].items():
            result_metrics[me_key][r[0]].append(me_value)

    print("")

    ####################################
    ##  write results to pickle file  ##
    ####################################
    
    if os.path.exists(raw_data_file):
        os.remove(raw_data_file)
    output_file = os.path.join(args["output_dir"], "sim_results.pkl")
    pkl.dump(result_metrics, open(output_file, "wb"), -1)
    print("Done with simulation! 🎉")
