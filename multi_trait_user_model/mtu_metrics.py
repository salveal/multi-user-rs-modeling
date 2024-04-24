import numpy as np
from trecs.metrics import Measurement, MSEMeasurement
from trecs.random import Generator
from sklearn.metrics.pairwise import cosine_similarity
from mtu_utils import calculate_avg_jaccard, mean_item_dist_pairs

"""
Metrics taken from algo_confounding
"""
class InteractionTracker(Measurement):
    """ Tracks all user interactions up to the current timepoint. In the context of replication,
        it will be used to track the items that users interact with in the ideal
        recommender. We need this in order to calculate homogenization of other non-ideal
        RS algorithms, relative to the ideal recommender.
    """
    def __init__(self, name="interaction_history", verbose=False):
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return
        self.observe(np.copy(interactions).reshape((-1, 1)))

class MeanInteractionDistance(Measurement):
    """
    Cacluates the mean distance between items in each users' recommendation list based on their item attributes
    This class inherits from :class:`.Measurement`.
    Parameters
    -----------
        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.
    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`
        name: str (optional, default: "mean_rec_distance")
            Name of the measurement component.
    """
    def __init__(self, pairs, name="mean_interaction_dist", verbose=False):
        Measurement.__init__(self, name, verbose)
        self.pairs = pairs
        self.interaction_hist = None

    def measure(self, recommender):
        """
        Based on the pairings provided by the user, calculates the distance between
        the average item interacted by user 1 and user 2. These distances are averaged
        over all pairs. See mean_item_dist_pairs for more details.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return
        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack([self.interaction_hist, interactions.reshape((-1, 1))])

        avg_dist = mean_item_dist_pairs(self.pairs, self.interaction_hist, recommender.actual_item_attributes)
        self.observe(avg_dist)

class SimilarUserInteractionSimilarity(Measurement):
    """
    Measures the homogenization of users deemed most similar by the RS algorithm,
    relative to the homogenization those users face under the ideal recommender.
    """
    def __init__(self, ideal_interaction_hist, name="similar_user_jaccard", seed=None, verbose=False):
        self.ideal_hist = ideal_interaction_hist
        self.interaction_hist = None
        self.timestep = 0
        self.rng = Generator(seed)
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.timestep += 1
            self.observe(None)
            return
        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack([self.interaction_hist, interactions.reshape((-1, 1))])
        # generate cosine similarity matrix for all users
        assert recommender.users_hat.get_timesteps() == self.timestep + 1 # ensure that the users_hat variable is storing copies at each timestep
        user_representation = recommender.users_hat.state_history[-1]
        sim_matrix = cosine_similarity(user_representation, user_representation)
        # set diagonal entries to zero
        num_users = sim_matrix.shape[0]
        sim_matrix[np.arange(num_users), np.arange(num_users)] = 0
        # add random perturbation to break ties
        sim_tiebreak = np.zeros(
            sim_matrix.shape, dtype=[("score", "f8"), ("random", "f8")]
        )
        sim_tiebreak["score"] = sim_matrix
        sim_tiebreak["random"] = self.rng.random(sim_matrix.shape)
        # array where element x at index i represents the "most similar" user to user i
        closest_users = np.argsort(sim_tiebreak, axis=1, order=["score", "random"])[:, -1]
        pairs = list(enumerate(closest_users))
        # calculate average jaccard similarity
        ideal_similarity = calculate_avg_jaccard(pairs, self.ideal_hist[:, :(self.timestep + 1)]) # compare
        this_similarity = calculate_avg_jaccard(pairs, self.interaction_hist)
        self.observe(this_similarity - ideal_similarity)
        self.timestep += 1 # increment timestep

class MeanDistanceSimUsers(Measurement):
    """
    Cacluates the mean distance between items in each users' interaction list based on their item attributes
    This class inherits from :class:`.Measurement`.
    Parameters
    -----------
        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.
    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`
        name: str (optional, default: "mean_rec_distance")
            Name of the measurement component.
    """
    def __init__(self, ideal_interaction_hist, ideal_item_attrs, seed=None, name="sim_user_dist", verbose=False):
        self.ideal_hist = ideal_interaction_hist
        self.ideal_item_attrs = ideal_item_attrs
        self.interaction_hist = None
        self.timestep = 0
        self.rng = Generator(seed)
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        """
        Based on pairs generated by finding the most similar user to each user (by cosine
        similarity of algorithmic representation), calculates the distance between
        the average item interacted by user 1 and user 2. These distances are averaged
        over all pairs. Finally, we subtract the same metric for the pairings provided from
        the ideal algorithmic simulation.

        See mean_item_dist_pairs for more details.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return
        interactions = recommender.interactions
        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack([self.interaction_hist, interactions.reshape((-1, 1))])

        # get value of user matrix
        user_representation = recommender.users_hat.state_history[-1]
        # find most similar users
        sim_matrix = cosine_similarity(user_representation, user_representation)
        # set diagonal entries to zero
        num_users = sim_matrix.shape[0]
        sim_matrix[np.arange(num_users), np.arange(num_users)] = 0
        # add random perturbation to break ties
        sim_tiebreak = np.zeros(
            sim_matrix.shape, dtype=[("score", "f8"), ("random", "f8")]
        )
        sim_tiebreak["score"] = sim_matrix
        sim_tiebreak["random"] = self.rng.random(sim_matrix.shape)
        # array where element x at index i represents the "most similar" user to user i
        closest_users = np.argsort(sim_tiebreak, axis=1, order=["score", "random"])[:, -1]
        pairs = list(enumerate(closest_users))
        # calculate average jaccard similarity
        ideal_hist = self.ideal_hist[:, :(self.timestep + 1)]
        ideal_dist = mean_item_dist_pairs(pairs, ideal_hist, self.ideal_item_attrs)
        this_dist = mean_item_dist_pairs(pairs, self.interaction_hist, recommender.actual_item_attributes)
        self.observe(this_dist - ideal_dist)
        self.timestep += 1 # increment timestep

"""
Custom metrics
"""
class RMSEMeasurementFixed(Measurement):
    def __init__(self, verbose=False):
        Measurement.__init__(self, "rmse", verbose)
    
    def measure(self, recommender):
        predicted_scores_by_RS = recommender.score_fn(recommender.predicted_user_profiles, recommender.predicted_item_attributes)
        actual_user_scores_resized = recommender.users.actual_user_scores.value[:, :predicted_scores_by_RS.shape[1]]
        diff = predicted_scores_by_RS - actual_user_scores_resized
        if diff.size == 0:
            self.observe(None)
        else:
            result_data = (diff ** 2).mean()
            if result_data != float('-inf'):
                self.observe(result_data ** (0.5), copy=False)
            else:
                self.observe(0.0)
