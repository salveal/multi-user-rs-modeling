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
        #print("self.trait_distribution",self.trait_distribution)
        self.total_item_interactions = np.array([], dtype=int)
        self.interactions_previous_iter = np.array([], dtype=int)
        self.new_feed = np.array([], dtype=int)
        self.social_network = social_network
        self.shared_items = np.empty(num_users, dtype=object)
        self.shared_items[...] = [[] for _ in range(self.shared_items.shape[0])]
        self.scores_after_interaction = true_scores
        self.chosen_lists = []
        self.sharing_threshold = np.zeros(num_users)
        super().__init__(*args, **kwargs)
    
    def score_new_items(self, items):
        new_items_interactions = np.zeros(items.shape[1], dtype=int)
        self.new_feed = np.hstack((np.arange(self.new_feed.shape[0], self.new_feed.shape[0]+items.shape[1])[::-1], self.new_feed))
        self.total_item_interactions = np.hstack((self.total_item_interactions, new_items_interactions))
        self.interactions_previous_iter = np.hstack((self.interactions_previous_iter, new_items_interactions))
    
    def get_user_feedback(self, items_shown):
        #print("\nitems_shown:\n", items_shown[0])
        if not self.repeat_interactions:
            prev_interacted_scores = self.actual_user_scores.get_item_scores(self.user_interactions)
            self.actual_user_scores.set_item_scores_to_value(self.user_interactions, float("-inf"))

        random_feed = np.array([self.rng.permutation(self.total_item_interactions.shape[0]) for _ in range(self.trait_distribution.shape[0])])

        popular_items_tiebreak = np.zeros(
            self.total_item_interactions.shape, dtype=[("rank", "u4"), ("random", "f8")]
        )
        popular_items_tiebreak["rank"] = self.total_item_interactions
        popular_items_tiebreak["random"] = self.rng.random(self.total_item_interactions.shape)
        popular_feed = np.argsort(popular_items_tiebreak, order=["rank", "random"])[::-1]
        resized_popular_feed = np.tile(popular_feed, (self.trait_distribution.shape[0], 1))
        if not self.repeat_interactions:
            resized_popular_feed = np.array([row[~np.isin(row, self.user_interactions[i])] for i, row in enumerate(resized_popular_feed)])
        resized_popular_feed = resized_popular_feed.tolist()
        #print("pop_feed",resized_popular_feed[0])

        trending_items_tiebreak = np.zeros(
            self.interactions_previous_iter.shape, dtype=[("rank", "u4"), ("random", "f8")]
        )
        trending_items_tiebreak["rank"] = self.interactions_previous_iter
        trending_items_tiebreak["random"] = self.rng.random(self.interactions_previous_iter.shape)
        trending_feed = np.argsort(trending_items_tiebreak, order=["rank", "random"])[::-1]
        resized_trending_feed = np.tile(trending_feed, (self.trait_distribution.shape[0], 1)).tolist()

        resized_new_feed = np.tile(self.new_feed, (self.trait_distribution.shape[0], 1)).tolist()

        ideal_feed = np.argsort(self.scores_after_interaction[:,:self.total_item_interactions.shape[0]])[:, ::-1]

        user_lists = [[items_shown[i],
                       random_feed[i],
                       resized_popular_feed[i],
                       resized_trending_feed[i],
                       resized_new_feed[i],
                       [x[0] for x in self.shared_items[i]],
                       ideal_feed[i]
                       ] for i in range(self.trait_distribution.shape[0])]
        weighted_user_lists = [[self.attention_transform(self.actual_user_scores.value[np.repeat(i, len(user_lists[i][j])), user_lists[i][j]]) * self.trait_distribution[i][j] for j in range(len(user_lists[i]))] for i in range(len(user_lists))]
        #print("user_lists[0] sr: ", user_lists[0][0][:min(len(user_lists[0][0]), 50)])
        #print("user_lists[0] pop: ", user_lists[0][1][:min(len(user_lists[0][1]), 50)])
        #print("user_lists[0] new: ", user_lists[0][2][:min(len(user_lists[0][2]), 50)])
        #print("weighted_user_lists[0] sr: ", weighted_user_lists[0][0][:min(len(user_lists[0][0]), 50)])
        #print("weighted_user_lists[0] pop: ", weighted_user_lists[0][1][:min(len(user_lists[0][2]), 50)])
        #print("weighted_user_lists[0] new: ", weighted_user_lists[0][2][:min(len(user_lists[0][2]), 50)])
        #print("user_lists[0] new: ", user_lists[0][2])
        #print("len(user_lists)", len(user_lists))
        #print("len(user_lists[0])", len(user_lists[0]))
        #print("len(weighted_user_lists)", len(weighted_user_lists))
        #print("len(weighted_user_lists[0])", len(weighted_user_lists[0]))
        #print("unweighted_user_lists[0][0]", self.actual_user_scores.value[np.repeat(0, len(user_lists[0][0])), user_lists[0][0]])
        #print("weights on user 0's RS list:\n", weighted_user_lists[0][0][:15])
        interactions, chosen_list = self.get_chosen_items(user_lists, weighted_user_lists)
        #print("\nitem selected:", interactions[0])
        self.chosen_lists.append(chosen_list)

        new_item_interactions = np.bincount(interactions)
        self.total_item_interactions[:len(new_item_interactions)] += new_item_interactions
        self.interactions_previous_iter = new_item_interactions

        if not self.repeat_interactions:
            self.actual_user_scores.set_item_scores_to_value(
                self.user_interactions, prev_interacted_scores
            )
            interactions_col = interactions.reshape((-1, 1))
            self.user_interactions = np.hstack([self.user_interactions, interactions_col])
        
        true_scores_from_interactions = self.scores_after_interaction[np.arange(self.scores_after_interaction.shape[0]), interactions]
        iteration_number = len(self.chosen_lists)
        for i in range(true_scores_from_interactions.shape[0]):
            if true_scores_from_interactions[i] > self.sharing_threshold[i]:
                connected_indices = np.where(self.social_network[i] == 1)[0]
                for j in connected_indices:
                    if interactions[i] not in [x[0] for x in self.shared_items[j]]:
                        if not self.repeat_interactions:
                            if interactions[i] not in self.user_interactions[j]:
                                self.shared_items[j].append((interactions[i], 1))
                        else:
                            self.shared_items[j].append((interactions[i], 1))
                    else:
                        if not self.repeat_interactions:
                            if interactions[i] not in self.user_interactions[j]:
                                t = self.shared_items[j][[x[0] for x in self.shared_items[j]].index(interactions[i])]
                                self.shared_items[j][[x[0] for x in self.shared_items[j]].index(interactions[i])] = (t[0], t[1]+1)
                                self.shared_items[j].sort(key=lambda x: x[1], reverse=True)
                        else:
                            t = self.shared_items[j][[x[0] for x in self.shared_items[j]].index(interactions[i])]
                            self.shared_items[j][[x[0] for x in self.shared_items[j]].index(interactions[i])] = (t[0], t[1]+1)
                            self.shared_items[j].sort(key=lambda x: x[1], reverse=True)
            self.sharing_threshold[i] = ((self.sharing_threshold[i]*(iteration_number - 1)) + true_scores_from_interactions[i]) / iteration_number
        
        # print("shared items:", self.shared_items)
        return interactions
    
    def attention_transform(self, recommended_item_scores):
        no_inf_rec_item_scores = np.copy(recommended_item_scores)
        no_inf_rec_item_scores[no_inf_rec_item_scores == float('-inf')] = 0.0
        if len(no_inf_rec_item_scores.shape) > 1:
            return super().attention_transform(no_inf_rec_item_scores)
        if self.attention_exp != 0:
            num_items = no_inf_rec_item_scores.shape[0]
            idxs = np.arange(num_items) + 1
            multiplier = np.power(idxs, self.attention_exp)
            return no_inf_rec_item_scores * multiplier
        else:
            return no_inf_rec_item_scores

    def get_chosen_items(self, ls, weighted_ls):
        interactions = []
        chosen_list = []
        for i in range(len(ls)):
            maximum_values = []
            maximum_indices = []
            for j in range(len(ls[i])):
                if len(ls[i][j]) == 0:
                    maximum_values.append(float('-inf'))
                    maximum_indices.append(-1)
                else:
                    maximum_values.append(max(weighted_ls[i][j]))
                    maximum_indices.append(max(enumerate(weighted_ls[i][j]), key=lambda x: x[1])[0])
            selected_ls = max(enumerate(maximum_values), key=lambda x: x[1])[0]
            interactions.append(ls[i][selected_ls][maximum_indices[selected_ls]])
            chosen_list.append(selected_ls)
            _l = [x[0] for x in self.shared_items[i]]
            if ls[i][selected_ls][maximum_indices[selected_ls]] in _l:
                self.shared_items[i].pop(_l.index(ls[i][selected_ls][maximum_indices[selected_ls]]))
        return np.array(interactions, dtype=int), np.array(chosen_list, dtype=int)
