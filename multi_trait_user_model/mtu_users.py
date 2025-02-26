import numpy as np
from trecs.components import Users, ActualUserScores
from trecs.random import Generator
import matplotlib.pyplot as plt

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
    def __init__(self, num_users, true_scores, social_network, traits, seed, *args, **kwargs):
        self.iteration_number = 0
        self.traits = traits
        self.feed_preference = np.tile(traits, (num_users, 1))
        self.total_sum_rand = 1.0
        if -1 in traits:
            self.total_sum_rand = 1 - traits[traits != -1].sum()
            rng = Generator(seed)
            random_traits = rng.dirichlet(np.ones((traits == -1).sum())*50, size=num_users)*self.total_sum_rand

            # Just making RSs more powerful
            #if traits[0] == -1:
            #    a = 0.01
            #    p_random_traits = np.copy(random_traits)
            #    random_traits[:,0] += (self.total_sum_rand - random_traits[:,0])*a
            #    ra = (1 - random_traits[:,0]) / (1 - p_random_traits[:,0])
            #    random_traits[:,1:] = np.array([random_traits[:,1:][i] * ra[i] for i in range(ra.shape[0])])
            
            self.feed_preference[:, np.all(self.feed_preference == -1, axis=0)] = random_traits
        self.feed_preference[:, 0] += 1 - self.feed_preference.sum(axis=1)
        self.mean_utility_per_feed = np.copy(self.feed_preference)
        self.total_item_interactions = np.array([], dtype=int)
        self.interactions_previous_iter = np.array([], dtype=int)
        self.new_feed = np.array([], dtype=int)
        self.social_network = social_network
        self.shared_items = np.empty(num_users, dtype=object)
        self.shared_items[...] = [[] for _ in range(self.shared_items.shape[0])]
        self.scores_after_interaction = true_scores
        self.chosen_lists = []
        self.rank_chosen_items = []
        self.sharing_threshold = np.zeros(num_users)
        super().__init__(*args, **kwargs)
    
    def score_new_items(self, items):
        new_items_interactions = np.zeros(items.shape[1], dtype=int)
        self.new_feed = np.hstack((np.arange(self.new_feed.shape[0], self.new_feed.shape[0]+items.shape[1])[::-1], self.new_feed))
        self.total_item_interactions = np.hstack((self.total_item_interactions, new_items_interactions))
        self.interactions_previous_iter = np.hstack((self.interactions_previous_iter, new_items_interactions))
    
    def make_rs_feed_plot(self, list_to_plot, plotting_iter_gap):
        if self.iteration_number == 9 or self.iteration_number == 19 or self.iteration_number == 99:
            plt.pcolormesh(list_to_plot[:, :200], vmin=0, vmax=0.7)
            plt.show()

    def get_user_feedback(self, items_shown):
        if not self.repeat_interactions:
            prev_interacted_scores = self.actual_user_scores.get_item_scores(self.user_interactions)
            self.actual_user_scores.set_item_scores_to_value(self.user_interactions, float("-inf"))

        random_feed = np.array([self.rng.permutation(self.total_item_interactions.shape[0]) for _ in range(self.user_vector.shape[0])])
        if not self.repeat_interactions:
            random_feed = np.array([row[~np.isin(row, self.user_interactions[i])] for i, row in enumerate(random_feed)])
        random_feed = random_feed.tolist()

        popular_items_tiebreak = np.zeros(
            self.total_item_interactions.shape, dtype=[("rank", "u4"), ("random", "f8")]
        )
        popular_items_tiebreak["rank"] = self.total_item_interactions
        popular_items_tiebreak["random"] = self.rng.random(self.total_item_interactions.shape)
        popular_feed = np.argsort(popular_items_tiebreak, order=["rank", "random"])[::-1]
        resized_popular_feed = np.tile(popular_feed, (self.user_vector.shape[0], 1))
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
        resized_trending_feed = np.tile(trending_feed, (self.user_vector.shape[0], 1))
        if not self.repeat_interactions:
            resized_trending_feed = np.array([row[~np.isin(row, self.user_interactions[i])] for i, row in enumerate(resized_trending_feed)])
        resized_trending_feed = resized_trending_feed.tolist()

        resized_new_feed = np.tile(self.new_feed, (self.user_vector.shape[0], 1)).tolist()

        ideal_feed = np.argsort(self.scores_after_interaction[:,:self.total_item_interactions.shape[0]])[:, ::-1]

        user_lists = [[items_shown[i],
                       random_feed[i],
                       resized_popular_feed[i],
                       resized_trending_feed[i],
                       resized_new_feed[i],
                       [x[0] for x in self.shared_items[i]],
                       ideal_feed[i]
                       ] for i in range(self.user_vector.shape[0])]
        if -1 in self.traits:
            g = self.mean_utility_per_feed[:, np.where(self.traits == -1)[0]]
            self.feed_preference[:, np.where(self.traits == -1)[0]] = np.array([row / row.sum() for row in g]) * self.total_sum_rand
            float_error = 1 - self.feed_preference.sum(axis=1)
            self.feed_preference[:, self.traits.tolist().index(-1)] += float_error

        weighted_user_lists = [[self.attention_transform(self.actual_user_scores.value[np.repeat(i, len(user_lists[i][j])), user_lists[i][j]]) * self.feed_preference[i][j] for j in range(len(user_lists[i]))] for i in range(len(user_lists))]

        #plot_user = np.array([[i] * items_shown.shape[1] for i in range(items_shown.shape[0])])
        #rs_index = 0
        #self.make_rs_feed_plot(self.scores_after_interaction[plot_user, items_shown], plotting_iter_gap=10)

        interactions, chosen_list, rank_chosen_item = self.get_chosen_items(user_lists, weighted_user_lists)
        self.chosen_lists.append(chosen_list)
        self.rank_chosen_items.append(rank_chosen_item)
        self.iteration_number += 1
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
            self.sharing_threshold[i] = ((self.sharing_threshold[i]*(self.iteration_number - 1)) + true_scores_from_interactions[i]) / self.iteration_number
        
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

    def get_chosen_items(self, user_lists, weighted_ls):
        interactions = []
        chosen_list = []
        rank_chosen_item = []
        #a = 1/10
        #w = []
        for u in range(len(user_lists)):
            maximum_values = []
            maximum_indices = []
            for l in range(len(user_lists[u])):
                if len(user_lists[u][l]) == 0:
                    maximum_values.append(float('-inf'))
                    maximum_indices.append(-1)
                else:
                    maximum_values.append(max(weighted_ls[u][l]))
                    maximum_indices.append(max(enumerate(weighted_ls[u][l]), key=lambda x: x[1])[0])
            selected_ls = max(enumerate(maximum_values), key=lambda x: x[1])[0]
            selected_item = user_lists[u][selected_ls][maximum_indices[selected_ls]]

            if self.traits[selected_ls] == -1:
                aa = 0.1 if self.iteration_number < 10 else 1
                alph = 0.1*aa
                self.mean_utility_per_feed[u][selected_ls] += alph*(self.scores_after_interaction[u, selected_item] - self.mean_utility_per_feed[u][selected_ls]) 

            interactions.append(selected_item)
            chosen_list.append(selected_ls)
            rank_chosen_item.append(maximum_indices[selected_ls])
            _l = [x[0] for x in self.shared_items[u]]
            if selected_item in _l:
                self.shared_items[u].pop(_l.index(selected_item))
            #w.append(max(weighted_ls[i][selected_ls]))
        #w = np.array(w)
        #tru = self.scores_after_interaction[np.arange(self.scores_after_interaction.shape[0]), interactions]
        #print("max(tru - w)", max(tru - w))
        return np.array(interactions, dtype=int), np.array(chosen_list, dtype=int), np.array(rank_chosen_item, dtype=int)
