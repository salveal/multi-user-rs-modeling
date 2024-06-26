import numpy as np
from trecs.components import Users, ActualUserScores

"""
Classes taken from algo_confounding
"""
class ChaneyUsers(Users):
    def __init__(self, known_scores, *args, **kwargs):
        self.known_scores = np.copy(known_scores)
        super().__init__(*args, **kwargs)
    
    def compute_user_scores(self, items):
        self.actual_user_scores = ActualUserScores(self.known_scores)
    
    def score_new_items(self, items):
        pass

    def get_user_feedback(self, *args, **kwargs):
        interactions = super().get_user_feedback(*args, **kwargs)
        return interactions

"""
Custom Classes
"""
class MultiTraitUsers(ChaneyUsers):
    def __init__(self, num_users, true_scores, social_network, traits, *args, **kwargs):
        self.trait_distribution = np.tile(traits, (num_users, 1))
        self.trait_distribution[:, 0] += 1 - self.trait_distribution.sum(axis=1)
        self.total_item_interactions = np.array([], dtype=int)
        self.social_network = social_network
        self.shared_items = np.empty(num_users, dtype=object)
        self.shared_items[...] = [[] for _ in range(self.shared_items.shape[0])]
        self.scores_after_interaction = true_scores
        super().__init__(*args, **kwargs)
    
    def score_new_items(self, items):
        new_items_interactions = np.zeros(items.shape[1], dtype=int)
        self.total_item_interactions = np.hstack((self.total_item_interactions, new_items_interactions))
    
    def get_user_feedback(self, items_shown):
        trait_distribution_no_items_shared = self.trait_distribution.copy()
        b = 1/(trait_distribution_no_items_shared[:, 0] + trait_distribution_no_items_shared[:, 1])
        trait_distribution_no_items_shared[:, [0, 1]] *= b[:, np.newaxis]
        trait_distribution_no_items_shared[:, 2] = 0.0
        trait_distribution_no_items_shared[:, 0] += 1 - trait_distribution_no_items_shared.sum(axis=1)

        cond = np.array([x != [] for x in self.shared_items])
        trait_distribution_no_items_shared[cond] = self.trait_distribution[cond]

        users_selections = np.array([self.rng.choice(np.arange(trait_distribution_no_items_shared.shape[1]), size=1, p=row) for row in trait_distribution_no_items_shared])
        users_selections = users_selections.flatten()

        if not self.repeat_interactions:
            prev_interacted_scores = self.actual_user_scores.get_item_scores(self.user_interactions)
            self.actual_user_scores.set_item_scores_to_value(self.user_interactions, float("-inf"))
        
        popular_items_tiebreak = np.zeros(
            self.total_item_interactions.shape, dtype=[("rank", "u4"), ("random", "f8")]
        )
        popular_items_tiebreak["rank"] = self.total_item_interactions
        popular_items_tiebreak["random"] = self.rng.random(self.total_item_interactions.shape)
        popular_items_rank = np.argsort(popular_items_tiebreak, order=["rank", "random"])[::-1]

        interactions = []
        for i in range(users_selections.shape[0]):
            if users_selections[i] == 0:
                item_list = items_shown[i]
            elif users_selections[i] == 1:
                item_list = popular_items_rank
            elif users_selections[i] == 2:
                item_list = np.array(self.shared_items[i])
            user_item_scores_list = self.actual_user_scores.value[np.repeat(i, item_list.shape[0]), item_list]
            user_item_scores_list = self.attention_transform(user_item_scores_list)
            chosen_item = item_list[np.argmax(user_item_scores_list)]
            interactions += [chosen_item]
            if users_selections[i] == 2:
                self.shared_items[i] = list(filter(lambda x: x != chosen_item, self.shared_items[i]))
        interactions = np.array(interactions, dtype=int)
        
        new_item_interactions = np.bincount(interactions)
        self.total_item_interactions[:len(new_item_interactions)] += new_item_interactions

        if not self.repeat_interactions:
            self.actual_user_scores.set_item_scores_to_value(
                self.user_interactions, prev_interacted_scores
            )
            interactions_col = interactions.reshape((-1, 1))
            self.user_interactions = np.hstack([self.user_interactions, interactions_col])
        
        true_scores_from_interactions = self.scores_after_interaction[np.arange(self.scores_after_interaction.shape[0]), interactions]
        threshold = 1.0 # ??? debe haber mejor forma de hacer esto
        for i in range(true_scores_from_interactions.shape[0]):
            if true_scores_from_interactions[i] > threshold:
                connected_indices = np.where(self.social_network[i] == 1)[0]
                for index in connected_indices:
                    if interactions[i] not in self.shared_items[index]:
                        if not self.repeat_interactions:
                            if interactions[i] not in self.user_interactions[index]:
                                self.shared_items[index].append(interactions[i])
                        else:
                            self.shared_items[index].append(interactions[i])
        
        # print("shared items:", self.shared_items)
        return interactions
    
    def attention_transform(self, recommended_item_scores):
        if len(recommended_item_scores.shape) > 1:
            return super().attention_transform(recommended_item_scores)
        if self.attention_exp != 0:
            num_items = recommended_item_scores.shape[0]
            idxs = np.arange(num_items) + 1
            multiplier = np.power(idxs, self.attention_exp)
            return recommended_item_scores * multiplier
        else:
            return recommended_item_scores
