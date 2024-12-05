import numpy as np
from trecs.metrics import Measurement, MSEMeasurement, InteractionSimilarity
from trecs.random import Generator
from sklearn.metrics.pairwise import cosine_similarity
from mtu_utils import calculate_avg_jaccard, mean_item_dist_pairs, get_inter_user_metrics, get_intra_user_metrics, get_sim_users_pairs

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

class UserJaccard(InteractionSimilarity):
    def __init__(self, pairs, name="user_jaccard", verbose=False):
        super().__init__(pairs, name, verbose)

class UserMeanDistance(Measurement):
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
    def __init__(self, pairs, name="user_mean_dist", verbose=False):
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

class SimilarUserJaccard(Measurement):
    """
    Measures the homogenization of users deemed most similar by the RS algorithm,
    relative to the homogenization those users face under the ideal recommender.
    """
    def __init__(self, name="sim_user_jaccard", seed=None, verbose=False):
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
        pairs = get_sim_users_pairs(user_representation, self.rng)
        # calculate average jaccard similarity
        similarity = calculate_avg_jaccard(pairs, self.interaction_hist)
        self.observe(similarity)
        self.timestep += 1 # increment timestep

class SimilarUserMeanDistance(Measurement):
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
    def __init__(self, seed=None, name="sim_user_mean_dist", verbose=False):
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
        pairs = get_sim_users_pairs(user_representation, self.rng)
        # calculate average jaccard similarity
        dist = mean_item_dist_pairs(pairs, self.interaction_hist, recommender.actual_item_attributes)
        self.observe(dist)
        self.timestep += 1 # increment timestep

class TrueSimilarUserJaccard(Measurement):
    """
    Measures the homogenization of users deemed most similar by the RS algorithm,
    relative to the homogenization those users face under the ideal recommender.
    """
    def __init__(self, name="true_sim_user_jaccard", seed=None, verbose=False):
        self.interaction_hist = None
        self.timestep = 0
        self.rng = Generator(seed)
        self.sim_matrix = None
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
        if self.sim_matrix is None:
            assert recommender.users.actual_user_profiles.get_timesteps() == self.timestep + 1 # ensure that the users variable is storing copies at each timestep
            user_representation = recommender.users.actual_user_profiles.state_history[-1]
            self.sim_matrix = cosine_similarity(user_representation, user_representation)
            # set diagonal entries to zero
            num_users = self.sim_matrix.shape[0]
            self.sim_matrix[np.arange(num_users), np.arange(num_users)] = 0
        # add random perturbation to break ties
        sim_tiebreak = np.zeros(
            self.sim_matrix.shape, dtype=[("score", "f8"), ("random", "f8")]
        )
        sim_tiebreak["score"] = self.sim_matrix
        sim_tiebreak["random"] = self.rng.random(self.sim_matrix.shape)
        # array where element x at index i represents the "most similar" user to user i
        closest_users = np.argsort(sim_tiebreak, axis=1, order=["score", "random"])[:, -1]
        pairs = list(enumerate(closest_users))
        # calculate average jaccard similarity
        similarity = calculate_avg_jaccard(pairs, self.interaction_hist)
        self.observe(similarity)
        self.timestep += 1 # increment timestep

class TrueSimilarUserMeanDistance(Measurement):
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
    def __init__(self, seed=None, name="true_sim_user_mean_dist", verbose=False):
        self.interaction_hist = None
        self.timestep = 0
        self.rng = Generator(seed)
        self.sim_matrix = None
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

        if self.sim_matrix is None:
            # get value of user matrix
            user_representation = recommender.users.actual_user_profiles.state_history[-1]
            # find most similar users
            self.sim_matrix = cosine_similarity(user_representation, user_representation)
            # set diagonal entries to zero
            num_users = self.sim_matrix.shape[0]
            self.sim_matrix[np.arange(num_users), np.arange(num_users)] = 0
        # add random perturbation to break ties
        sim_tiebreak = np.zeros(
            self.sim_matrix.shape, dtype=[("score", "f8"), ("random", "f8")]
        )
        sim_tiebreak["score"] = self.sim_matrix
        sim_tiebreak["random"] = self.rng.random(self.sim_matrix.shape)
        # array where element x at index i represents the "most similar" user to user i
        closest_users = np.argsort(sim_tiebreak, axis=1, order=["score", "random"])[:, -1]
        pairs = list(enumerate(closest_users))
        # calculate average jaccard similarity
        dist = mean_item_dist_pairs(pairs, self.interaction_hist, recommender.actual_item_attributes)
        self.observe(dist)
        self.timestep += 1 # increment timestep

class SimilarUserJaccardRelative(Measurement):
    """
    Measures the homogenization of users deemed most similar by the RS algorithm,
    relative to the homogenization those users face under the ideal recommender.
    """
    def __init__(self, ideal_interaction_hist, name="sim_user_jaccard_relative", seed=None, verbose=False):
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
        print("user_representation", user_representation)
        print("user_representation.shape", user_representation.shape)
        pairs = get_sim_users_pairs(user_representation, self.rng)
        # calculate average jaccard similarity
        ideal_similarity = calculate_avg_jaccard(pairs, self.ideal_hist[:, :(self.timestep + 1)]) # compare
        this_similarity = calculate_avg_jaccard(pairs, self.interaction_hist)
        self.observe(this_similarity - ideal_similarity)
        self.timestep += 1 # increment timestep

class SimilarUserMeanDistanceRelative(Measurement):
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
    def __init__(self, ideal_interaction_hist, ideal_item_attrs, seed=None, name="sim_user_mean_dist_relative", verbose=False):
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
        pairs = get_sim_users_pairs(user_representation, self.rng)
        # calculate average jaccard similarity
        ideal_hist = self.ideal_hist[:, :(self.timestep + 1)]
        ideal_dist = mean_item_dist_pairs(pairs, ideal_hist, self.ideal_item_attrs)
        this_dist = mean_item_dist_pairs(pairs, self.interaction_hist, recommender.actual_item_attributes)
        self.observe(this_dist - ideal_dist)
        self.timestep += 1 # increment timestep

"""
Custom metrics
"""
class RMSEMeasurement(Measurement):
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

class InterUserMeanDispersion(Measurement):
    def __init__(self, verbose=False):
        self.interaction_hist = None
        Measurement.__init__(self, "inter_user_mean_dispersion", verbose)
    
    def measure(self, recommender):
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return
        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack([self.interaction_hist, interactions.reshape((-1, 1))])
        #print("InterUserMeanDispersion, interactions", interactions)
        inter_user_mean_dispersion, _ = get_inter_user_metrics(recommender.actual_item_attributes, self.interaction_hist)
        self.observe(inter_user_mean_dispersion)

class InterUserDiversity(Measurement):
    def __init__(self, verbose=False):
        self.interaction_hist = None
        Measurement.__init__(self, "inter_user_diversity", verbose)
    
    def measure(self, recommender):
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return
        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack([self.interaction_hist, interactions.reshape((-1, 1))])
        _, inter_user_diversity = get_inter_user_metrics(recommender.actual_item_attributes, self.interaction_hist)
        self.observe(inter_user_diversity)

class IntraUserMeanDispersion(Measurement):
    def __init__(self, verbose=False):
        self.interaction_hist = None
        Measurement.__init__(self, "intra_user_mean_dispersion", verbose)
    
    def measure(self, recommender):
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return
        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack([self.interaction_hist, interactions.reshape((-1, 1))])
        intra_user_mean_dispersion, _, _ = get_intra_user_metrics(recommender.actual_item_attributes, self.interaction_hist)
        self.observe(intra_user_mean_dispersion)

class IntraUserDispersionVariance(Measurement):
    def __init__(self, verbose=False):
        self.interaction_hist = None
        Measurement.__init__(self, "intra_user_dispersion_variance", verbose)
    
    def measure(self, recommender):
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return
        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack([self.interaction_hist, interactions.reshape((-1, 1))])
        _, intra_user_dispersion_variance, _ = get_intra_user_metrics(recommender.actual_item_attributes, self.interaction_hist)
        self.observe(intra_user_dispersion_variance)

class IntraUserDiversity(Measurement):
    def __init__(self, verbose=False):
        self.interaction_hist = None
        Measurement.__init__(self, "intra_user_diversity", verbose)
    
    def measure(self, recommender):
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return
        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack([self.interaction_hist, interactions.reshape((-1, 1))])
        _, _, intra_user_diversity = get_intra_user_metrics(recommender.actual_item_attributes, self.interaction_hist)
        self.observe(intra_user_diversity)

class AnwarFilterBubbleEffect(Measurement):
    def __init__(self, verbose=False):
        self.interaction_hist = None
        Measurement.__init__(self, "anwar_fbe", verbose)
    
    def measure(self, recommender):
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return
        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack([self.interaction_hist, interactions.reshape((-1, 1))])
        _, inter_user_diversity = get_inter_user_metrics(recommender.actual_item_attributes, self.interaction_hist)
        _, _, intra_user_diversity = get_intra_user_metrics(recommender.actual_item_attributes, self.interaction_hist)
        self.observe(inter_user_diversity / intra_user_diversity)
    
class AnwarHomogeneity(Measurement):
    def __init__(self, verbose=False):
        self.interaction_hist = None
        Measurement.__init__(self, "anwar_homogeneity", verbose)
    
    def measure(self, recommender):
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return
        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack([self.interaction_hist, interactions.reshape((-1, 1))])
        _, inter_user_diversity = get_inter_user_metrics(recommender.actual_item_attributes, self.interaction_hist)
        _, _, intra_user_diversity = get_intra_user_metrics(recommender.actual_item_attributes, self.interaction_hist)
        self.observe(1.0 / np.sqrt(inter_user_diversity**2 + intra_user_diversity**2))

class UtilityTracker(Measurement):
    def __init__(self, verbose=False):
        self.true_scores = None
        Measurement.__init__(self, "utility_history", verbose)
    
    def measure(self, recommender):
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.true_scores = recommender.users.scores_after_interaction
            self.observe(None)
            return
        new_scores = self.true_scores[np.arange(self.true_scores.shape[0]), interactions]
        self.observe(new_scores)

class MeanUtilityPerIteration(Measurement):
    def __init__(self, verbose=False):
        self.true_scores = None
        self.last_three = []
        Measurement.__init__(self, "mean_utility", verbose)
    
    def measure(self, recommender):
        interactions = recommender.interactions
        # print("interactions", interactions)
        if interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.true_scores = recommender.users.scores_after_interaction
            self.observe(None)
            return
        self.last_three += [interactions.tolist()]
        if len(self.last_three) > 3:
            self.last_three = self.last_three[1:]
        last_three_scores = self.true_scores[np.arange(self.true_scores.shape[0]), self.last_three]
        mean_score = last_three_scores.mean(axis=0).mean()
        self.observe(mean_score)

class MeanUtilityPerIterationIncrement(Measurement):
    def __init__(self, verbose=False):
        self.true_scores = None
        self.mean_sum = 0
        Measurement.__init__(self, "mean_utility_increment", verbose)
    
    def measure(self, recommender):
        interactions = recommender.interactions
        # print("interactions", interactions)
        if interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.true_scores = recommender.users.scores_after_interaction
            self.observe(None)
            return
        mean_score = self.true_scores[np.arange(self.true_scores.shape[0]), interactions].mean()
        self.mean_sum += mean_score
        self.observe(self.mean_sum)

class VarUtilityPerIteration(Measurement):
    def __init__(self, verbose=False):
        self.true_scores = None
        self.last_three = []
        Measurement.__init__(self, "var_utility", verbose)
    
    def measure(self, recommender):
        interactions = recommender.interactions
        # print("interactions", interactions)
        if interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.true_scores = recommender.users.scores_after_interaction
            self.observe(None)
            return
        self.last_three += [interactions.tolist()]
        if len(self.last_three) > 3:
            self.last_three = self.last_three[1:]
        last_three_scores = self.true_scores[np.arange(self.true_scores.shape[0]), self.last_three]
        var_score = last_three_scores.mean(axis=0).var()
        self.observe(var_score)

class MeanTotalUserUtility(Measurement):
    def __init__(self, verbose=False):
        self.total_scores = None
        self.true_scores = None
        Measurement.__init__(self, "mean_total_user_utility", verbose)
    
    def measure(self, recommender):
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.true_scores = recommender.users.scores_after_interaction
            self.observe(None)
            return
        if self.total_scores is None:
            self.total_scores = np.zeros(interactions.shape)
        new_scores = self.true_scores[np.arange(self.true_scores.shape[0]), interactions]
        self.total_scores = self.total_scores + new_scores
        mean_score_per_user = self.total_scores.mean()
        self.observe(mean_score_per_user)

class VarTotalUserUtility(Measurement):
    def __init__(self, verbose=False):
        self.total_scores = None
        self.true_scores = None
        Measurement.__init__(self, "var_total_user_utility", verbose)
    
    def measure(self, recommender):
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.true_scores = recommender.users.scores_after_interaction
            self.observe(None)
            return
        if self.total_scores is None:
            self.total_scores = np.zeros(interactions.shape)
        new_scores = self.true_scores[np.arange(self.true_scores.shape[0]), interactions]
        self.total_scores = self.total_scores + new_scores
        var_score_per_user = self.total_scores.var()
        self.observe(var_score_per_user)

class GiniCoefficient(Measurement):
    def __init__(self, verbose=False):
        self.interaction_hist = None
        Measurement.__init__(self, "gini", verbose)

    def measure(self, recommender):
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return
        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack([self.interaction_hist, interactions.reshape((-1, 1))])
        numerator = 0.0
        denominator = 0.0
        for i in range(recommender.num_items):
            user_interactions_for_item = (self.interaction_hist == i).any(axis=1).sum()
            #print(self.interaction_hist)
            item_rank = np.argsort(np.argsort(np.bincount(self.interaction_hist.flatten(), minlength=recommender.num_items))[::-1])[i] + 1
            numerator += ((2*item_rank) - recommender.num_items - 1) * user_interactions_for_item
            denominator += user_interactions_for_item
            #print("numerator", numerator)
            #print("denominator", denominator)
        gini = - numerator / (recommender.num_items * denominator)
        self.observe(gini)

class CumulativeListDecision(Measurement):
    def __init__(self, name="decisions", verbose=False):
        self.cumulative = None
        super().__init__(name, verbose)
    
    def measure(self, recommender):
        if recommender.interactions.size == 0:
            self.cumulative = np.zeros(recommender.users.trait_distribution.shape[1])
            #print(self.cumulative)
            self.observe(None)
            return
        new_cumulative = np.zeros(self.cumulative.shape[0])
        trait_count = np.bincount(recommender.users.chosen_lists[-1])
        new_cumulative[:trait_count.shape[0]] = trait_count
        self.cumulative += new_cumulative
        #print(self.cumulative)
        self.observe(self.cumulative)

class TraitTracker(Measurement):
    def __init__(self, name="trait_history", verbose=False):
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        trait_distribution = recommender.users.trait_distribution
        self.observe(trait_distribution)
