from collections import defaultdict
import numpy as np
from trecs.matrix_ops import inner_product
from sklearn.metrics.pairwise import cosine_similarity

"""
Functions taken from algo_confounding
"""
def gaussian_similarity(arr, sigma=1.0):
    """
    Computes the Gaussian similarity matrix for an array of shape (N, 1).
    
    Parameters:
    arr (numpy.ndarray): Input array of shape (N, 1).
    sigma (float): The standard deviation of the Gaussian function.
    
    Returns:
    numpy.ndarray: Gaussian similarity matrix of shape (N, N).
    """
    # Ensure the input is a 2D array with shape (N, 1)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    
    # Compute the pairwise squared Euclidean distances
    dist_matrix = np.abs(arr - arr.T)
    
    # Compute the Gaussian similarity
    sim_matrix = np.exp(-dist_matrix**2 / (2 * sigma**2))

    return sim_matrix

def mu_sigma_to_alpha_beta(mu, sigma):
    """ For Chaney's custom Beta' function, we convert
        a mean and variance to an alpha and beta parameter
        of a Beta function. See footnote 3 page 3 of Chaney
        et al. for details.
    """
    alpha = ((1-mu) / (sigma**2) - (1/mu)) * mu**2
    beta = alpha * (1/mu - 1)
    return alpha, beta

def gen_social_network(user_prefs):
    """ Generates a |U|x|U| social network of connections
        as specified in Chaney et al.
    """
    user_cov = np.cov(user_prefs)
    possible_thresholds = np.sort(user_cov.flatten())[::-1]
    user_connections = None
    num_users = user_prefs.shape[0]
    for thresh in possible_thresholds[num_users:]:
        num_connected = (user_cov >= thresh).any(axis=1).sum()
        if num_connected == num_users:
            return (user_cov >= thresh).astype(int) # final adjacency matrix
    raise RuntimeError("Could not find a suitable threshold.")

def gen_social_network_1_attr(user_prefs):
    user_sim = gaussian_similarity(user_prefs)
    np.fill_diagonal(user_sim, 0.0)
    possible_thresholds = np.sort(user_sim.flatten())[::-1]
    user_connections = None
    num_users = user_prefs.shape[0]
    for thresh in possible_thresholds[num_users:]:
        num_connected = (user_sim >= thresh).any(axis=1).sum()
        if num_connected == num_users:
            return (user_sim >= thresh).astype(int) # final adjacency matrix
    raise RuntimeError("Could not find a suitable threshold.")

def calculate_avg_jaccard(pairs, interactions):
    """ Calculates average Jaccard index over specified pairs of users.
    """
    similarity = 0
    num_pairs = len(pairs)
    for user1, user2 in pairs:
        itemset_1 = set(interactions[user1, :])
        itemset_2 = set(interactions[user2, :])
        common = len(itemset_1.intersection(itemset_2))
        union = len(itemset_1.union(itemset_2))
        similarity += common / union / num_pairs
    return similarity

def mean_item_dist_pairs(pairs, interaction_history, item_attributes):
    """
    For each pair, calculates the distance between the mean item
    interacted with by each member of the pair. Then averages these
    distances across all pairs.
    """
    dist = 0
    for pair in pairs:
        itemset_1 = interaction_history[pair[0], :].flatten()
        itemset_2 = interaction_history[pair[1], :].flatten()
        dist += distance_of_mean_items(itemset_1, itemset_2, item_attributes) / len(pairs)
    return dist

def distance_of_mean_items(items1, items2, item_attributes):
    """
    Returns the difference between the average vector of the items
    in set 1 and the average vector of the items in set 2.

    Assume items matrix is |A| x |I|
    """
    mean1 = item_attributes[:, items1].mean(axis=1)
    mean2 = item_attributes[:, items2].mean(axis=1)
    return np.linalg.norm(mean1 - mean2)

def interleave_new_items(generator):
    """ Chooses the most recent, newest items to interleave
        with the recommendation set. This custom interleaving method
        ensures that all of the most recently created items (i.e.,
        the newest items) are the ones interleaved with the recommendations.
    """
    def interleaving_fn(k, item_indices):
        num_users = item_indices.shape[0]
        indices = item_indices[:, -k:]
        values = generator.random(indices.shape)
        order = values.argsort(axis=1) # randomly sort indices within rows
        rows = np.tile(np.arange(num_users).reshape((-1, 1)), indices.shape[1])
        return indices[rows, order]
    return interleaving_fn

def perfect_scores(num_items_per_iter, true_scores):
    """ This custom scoring function ensures that all items in the system
        that are created after a certain point are given a score of negative
        infinity, ensuring that they will be at the very bottom of any recommendation
        list. Otherwise, we return the "true scores" specified in the
        true_scores array. Additionally, it ensures that the true utilities are
        returned for each item. This is the scoring used by the IdealRecommender.
    """
    score_copy = np.copy(true_scores)
    def score_fn(users, items):
        predicted_scores = np.copy(score_copy)
        num_users, num_items = users.shape[0], items.shape[1]
        # all predicted scores for these "new" items will be negative infinity,
        # ensuring they never get recommended; instead, they are interleaved into the recommendation
        # set
        predicted_scores = predicted_scores[:num_users, :num_items] # subset to correct dimensions
        if items.shape[1] == num_items_per_iter: # EDGE CASE: when num_items_per_iter = num_items
            # all predicted scores for these "new" items will be negative infinity,
            # ensuring they never get recommended
            predicted_scores[:, :] = float('-inf')
        return predicted_scores
    return score_fn

def exclude_new_items(num_items_per_iter):
    """ This custom scoring function ensures that all items in the system
        that are created after a certain point are given a score of negative
        infinity, ensuring that they will be at the very bottom of any recommendation
        list. This ensures that only the items in the training phase get recommended;
        new items are only recommended via the interleaving procedure.
    """
    # score_fn is called by process_new_items and by train.
    # therefore, when score_fn is being called the first time, we will give all new items
    # scores of negative infinity; then, when train() is called, the actual
    # scores will be supplied.
    def score_fn(users, items):
        predicted_scores = inner_product(users, items)
        if items.shape[1] == num_items_per_iter: # EDGE CASE: when num_items_per_iter = num_items
            # all predicted scores for these "new" items will be negative infinity,
            # ensuring they never get recommended
            predicted_scores[:, :] = float('-inf')
        return predicted_scores
    return score_fn

def distances_from_users_to_mean_user(item_attrs, interaction_hist):
    item_attrs_interactions = np.transpose(item_attrs[:, interaction_hist], (1,2,0))
    mean_item_attrs_interactions = item_attrs_interactions.mean(axis=1)
    mean_mean_item_attrs_interactions = mean_item_attrs_interactions.mean(axis=0)
    mean_mean_item_attrs_interactions_reshaped = np.tile(mean_mean_item_attrs_interactions[np.newaxis, :], (mean_item_attrs_interactions.shape[0], 1))
    distances_from_means_to_mean_mean = np.linalg.norm(mean_item_attrs_interactions - mean_mean_item_attrs_interactions_reshaped, axis=1)
    return distances_from_means_to_mean_mean

def measures_of_distances_from_consumed_items_to_mean_item(item_attrs, interaction_hist):
    item_attrs_interactions = np.transpose(item_attrs[:, interaction_hist], (1,2,0))
    mean_item_attrs_interactions = item_attrs_interactions.mean(axis=1)
    mean_item_attrs_interactions_reshaped = np.tile(mean_item_attrs_interactions[:, np.newaxis, :], (1, item_attrs_interactions.shape[1], 1))
    distances_from_attrs_to_means = np.linalg.norm(item_attrs_interactions - mean_item_attrs_interactions_reshaped, axis=2)
    mean_distances_from_attrs_to_means = distances_from_attrs_to_means.mean(axis=1)
    var_distances_from_attrs_to_means = np.var(distances_from_attrs_to_means, axis=1)
    return mean_distances_from_attrs_to_means, var_distances_from_attrs_to_means

def get_inter_user_metrics(item_attrs, interaction_hist):
    distances_from_means_to_mean_mean = distances_from_users_to_mean_user(item_attrs, interaction_hist)
    inter_user_mean_dispersion = distances_from_means_to_mean_mean.mean()
    inter_user_diversity = np.var(distances_from_means_to_mean_mean)
    return inter_user_mean_dispersion, inter_user_diversity

def get_intra_user_metrics(item_attrs, interaction_hist):
    mean_distances_from_attrs_to_means, var_distances_from_attrs_to_means = measures_of_distances_from_consumed_items_to_mean_item(item_attrs, interaction_hist)
    intra_user_mean_dispersion = mean_distances_from_attrs_to_means.mean()
    intra_user_dispersion_variance = np.var(mean_distances_from_attrs_to_means)
    intra_user_diversity = var_distances_from_attrs_to_means.mean()
    return intra_user_mean_dispersion, intra_user_dispersion_variance, intra_user_diversity

def get_sim_users_pairs(user_prefs, rng):
    sim_matrix = cosine_similarity(user_prefs, user_prefs)
    # set diagonal entries to zero
    num_users = sim_matrix.shape[0]
    sim_matrix[np.arange(num_users), np.arange(num_users)] = 0
    # add random perturbation to break ties
    sim_tiebreak = np.zeros(
        sim_matrix.shape, dtype=[("score", "f8"), ("random", "f8")]
    )
    sim_tiebreak["score"] = sim_matrix
    sim_tiebreak["random"] = rng.random(sim_matrix.shape)
    # array where element x at index i represents the "most similar" user to user i
    closest_users = np.argsort(sim_tiebreak, axis=1, order=["score", "random"])[:, -1]
    pairs = list(enumerate(closest_users))

    return pairs
