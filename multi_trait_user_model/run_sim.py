import numpy as np
from collections import defaultdict
from trecs.metrics import InteractionSimilarity
from trecs.models import (
    ContentFiltering,
    PopularityRecommender,
    SocialFiltering,
    ImplicitMF
)
from trecs.random import Generator
from mtu_creators import NewItemFactory
from mtu_metrics import (
    InteractionTracker,
    RMSEMeasurementFixed,
    MeanInteractionDistance,
    SimilarUserInteractionSimilarity,
    MeanDistanceSimUsers
)
from mtu_rs import (
    IdealRecommender,
    ChaneyContent,
    RandomRecommender
)
from mtu_users import MultiTraitUsers
from mtu_utils import (
    mu_sigma_to_alpha_beta,
    gen_social_network,
    interleave_new_items,
    perfect_scores,
    exclude_new_items
)
import argparse
import os
import errno
import pickle as pkl
import pprint
import warnings
warnings.simplefilter("ignore")


def init_sim_state(true_scores, noisy_scores, item_representation, social_network, traits, args):
    u = MultiTraitUsers(
        num_users=args["num_users"],
        true_scores=true_scores,
        known_scores=np.copy(noisy_scores),
        social_network=social_network,
        traits=traits,
        size=(args["num_users"], args["num_attrs"]),
        attention_exp=-0.8,
        repeat_interactions=False
    )
    #print("true_scores pre model init", u.true_scores)
    item_factory = NewItemFactory(np.copy(item_representation), args["new_items_per_iter"])
    empty_item_set = np.array([]).reshape((args["num_attrs"], 0))
    return u, item_factory, empty_item_set

def run_ideal_sim(user_prefs, true_utils, pairs, init_params, args, rng):
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
        "interleaving_fn": interleave_new_items(rng),
        "verbose": True,
        "seed": args["seed"]
    }
    model_params["user_representation"] = user_prefs
    model_params["score_fn"] = perfect_scores(args["new_items_per_iter"], true_utils)

    m = IdealRecommender(**model_params)

    metrics = [
        InteractionTracker(),
        # InteractionSpread(), no es tan útil, es mostly random
        RMSEMeasurementFixed(),
        InteractionSimilarity(pairs),
        MeanInteractionDistance(pairs),
    ]
    run_params = {
        "random_items_per_iter": args["new_items_per_iter"],
        "vary_random_items_per_iter": False
    }
    m.add_metrics(*metrics)
    m.startup_and_train(timesteps=args["startup_iters"])
    m.set_num_items_per_iter(post_startup_num_items_per_iter)
    m.run(timesteps=args["total_iters"] - args["startup_iters"], train_between_steps=args["repeated_training"], **run_params)
    m.close()
    return m

def run_sim(item_attrs, pairs, ideal_interaction_history, init_params, args, rng, model=ContentFiltering, user_representation=None):
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
        "interleaving_fn": interleave_new_items(rng),
        "verbose": True,
        "seed": args["seed"]
    }
    model_params["score_fn"] = exclude_new_items(args["new_items_per_iter"])
    if model != PopularityRecommender:
        if model != SocialFiltering:
            if model != ImplicitMF:
                model_params["num_attributes"] = args["num_attrs"]
            else:
                model_params["num_latent_factors"] = args["num_attrs"]
        else:
            model_params["user_representation"] = user_representation
    
    m = model(**model_params)

    metrics = [
        # InteractionSpread(), no es tan útil, es mostly random
        RMSEMeasurementFixed(),
        InteractionSimilarity(pairs),
        MeanInteractionDistance(pairs),
        SimilarUserInteractionSimilarity(ideal_interaction_history),
        MeanDistanceSimUsers(ideal_interaction_history, item_attrs)
    ]
    run_params = {
        "random_items_per_iter": args["new_items_per_iter"],
        "vary_random_items_per_iter": False
    }
    m.add_metrics(*metrics)
    m.add_state_variable(m.users_hat)
    m.startup_and_train(timesteps=args["startup_iters"], no_new_items=False)
    m.set_num_items_per_iter(post_startup_num_items_per_iter)
    m.run(timesteps=args["total_iters"] - args["startup_iters"], train_between_steps=args["repeated_training"], **run_params)
    m.close()
    return m

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
    parser.add_argument('--repeated_training', dest='repeated_training', action='store_true')
    parser.add_argument('--single_training', dest='repeated_training', action='store_false')
    parser.add_argument('--total_iters', type=int, default=100)

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
        user_prefs = rng.dirichlet(user_params[sim_index, :], size=args["num_users"])
        item_attrs = rng.dirichlet(item_params[sim_index, :], size=args["num_items"])

        # mean of the utility distribution
        true_utils_mu = user_prefs @ item_attrs.T
        true_utils_mu = np.clip(true_utils_mu, 1e-9, None) # avoid numerical stability issues

        alphas, betas = mu_sigma_to_alpha_beta(true_utils_mu, args["sigma"])
        true_util = rng.beta(alphas, betas, size=(args["num_users"], args["num_items"]))
        # assert support of true utilities; should be within 0 and 1
        assert true_util.min() >= 0 and true_util.max() <= 1

        # calculate known utility for each user
        alpha, beta = mu_sigma_to_alpha_beta(args["mu_n"], args["sigma"]) # parameters for beta function governing percentage of utility known to users
        perc_known = rng.beta(alpha, beta, size=(args["num_users"], args["num_items"]))
        known_util = true_util * perc_known

        # add all synthetic data to list
        users.append(user_prefs)
        social_networks.append(gen_social_network(user_prefs))
        items.append(item_attrs)
        true_utils.append(true_util)
        known_utils.append(known_util)
    

    ###########################
    ##  running simulations  ##
    ###########################
        
    model_keys = ["ideal", "content", "chaney_content", "pop", "mf", "sf", "random"]
    #ideal_model_metric_list = ["interaction_spread", "mse", "interaction_similarity", "mean_interaction_dist"]
    metric_list = ["rmse", "interaction_similarity", "mean_interaction_dist", "similar_user_jaccard", "sim_user_dist"]
    result_metrics = {k: defaultdict(list) for k in metric_list}
    models = {}

    print("Running simulations...👟")

    for i in range(args["num_sims"]):
        print("Starting simulation", i+1)
        true_prefs = users[i] # underlying true preferences
        true_scores = true_utils[i]
        noisy_scores = known_utils[i]
        item_representation = items[i].T
        social_network = social_networks[i]

        pairs = [rng.choice(args["num_users"], 2, replace=False) for _ in range(800)]

        init_params = {
            "true_scores": true_scores,
            "noisy_scores": noisy_scores,
            "item_representation": item_representation,
            "social_network": social_network,
            "traits": np.array([0.34, 0.33, 0.33]),
        }

        print("Running ideal:")
        models["ideal"] = run_ideal_sim(true_prefs, true_scores, pairs, init_params, args, rng)
        ideal_interaction_history = np.hstack(models["ideal"].get_measurements()["interaction_history"][1:])
        # ideal_interaction_history.shape = (total_iters, num_users, 1)

        print("Running content:")
        models["content"] = run_sim(item_representation, pairs, ideal_interaction_history, init_params, args, rng, model=ContentFiltering)
        print("Running chaney_content:")
        models["chaney_content"] = run_sim(item_representation, pairs, ideal_interaction_history, init_params, args, rng, model=ChaneyContent)
        print("Running pop:")
        models["pop"] = run_sim(item_representation, pairs, ideal_interaction_history, init_params, args, rng, model=PopularityRecommender)
        print("Running mf:")
        models["mf"] = run_sim(item_representation, pairs, ideal_interaction_history, init_params, args, rng, model=ImplicitMF)
        print("Running sf:")
        models["sf"] = run_sim(item_representation, pairs, ideal_interaction_history, init_params, args, rng, model=SocialFiltering, user_representation=social_network)
        print("Running random:")
        models["random"] = run_sim(item_representation, pairs, ideal_interaction_history, init_params, args, rng, model=RandomRecommender)

        print("Getting results...")
        for model_key in model_keys:
            model = models[model_key]
            for metric_key in metric_list:
                if not (model_key == "ideal" and (metric_key == "similar_user_jaccard" or metric_key == "sim_user_dist")):
                    measurements = model.get_measurements()[metric_key][1:]
                    #print("For simulation", i, "model", model_key, "and metric", metric_key, "the result is", measurements)
                    result_metrics[metric_key][model_key].append(measurements)
        print("")
    

    ####################################
    ##  write results to pickle file  ##
    ####################################
    
    output_file = os.path.join(args["output_dir"], "sim_results.pkl")
    pkl.dump(result_metrics, open(output_file, "wb"), -1)
    print("Done with simulation! 🎉")
