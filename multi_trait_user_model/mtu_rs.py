import numpy as np
from trecs.models import ContentFiltering, BaseRecommender
from scipy.optimize import nnls
import scipy.sparse as sp
from sklearn.decomposition import PCA
import umap
from sklearn import preprocessing

"""
RSs taken from algo_confounding
"""
class TRECSChaneyContent(ContentFiltering):
    """
    Chaney ContentFiltering model which uses NNLS solver
    It is actually identical to ContentFiltering and doesn't act
    like the RS utilized in Chaney et al. 2018
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_internal_state(self, interactions):
        # update cumulative interactions
        num_new_items = self.items_hat.shape[1] - self.cumulative_interactions.shape[1] # how many new items were added to the system?
        if num_new_items > 0:
            self.cumulative_interactions = np.hstack([self.cumulative_interactions, np.zeros((self.num_users, num_new_items))]) # add new items to cumulative interactions
        self.cumulative_interactions[self.users.user_vector, interactions] += 1

    def train(self):
        if hasattr(self, 'cumulative_interactions') and self.cumulative_interactions.sum() > 0: # if there are interactions present:
            items_to_train = self.cumulative_interactions.shape[1] # can't train representations for new items before interactions have happened!
            for i in range(self.num_users):
                item_attr = self.items_hat.value[:, :items_to_train].T
                self.users_hat.value[i, :] = nnls(item_attr, self.cumulative_interactions[i, :])[0] # solve for Content Filtering representation using nnls solver
            num_new_items = self.items_hat.shape[1] - self.cumulative_interactions.shape[1] # how many new items were added to the system?

        else:
            self.cumulative_interactions = np.zeros((self.users_hat.shape[0], self.items_hat.shape[1]))
        super().train()

class IdealRecommender(ContentFiltering):
    """
    With the Ideal Recommender, we make the *strong assumption* that the true scores are provided
    to the recommender system through a custom scoring function, which always returns the true
    underlying user-item scores. Therefore, this class is pretty much an empty skeleton; the only
    modification is that we don't update any internal state of the recommender at each time step.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_internal_state(self, interactions):
        # do not change users_hat!
        pass

class RandomRecommender(ContentFiltering):
    """
    Random recommender - randomly update user representation at every step
    """
    def _update_internal_state(self, interactions):
        self.items_hat.value[:, :] = self.random_state.random(self.items_hat.shape)
        self.users_hat.value[:, :] = self.random_state.random(self.users_hat.shape)

    def process_new_items(self, new_items):
        """
        Generate random attributes for new items.
        """
        num_items = new_items.shape[1]
        num_attr = self.items_hat.value.shape[0]
        item_representation = self.random_state.random((num_attr, num_items))
        return item_representation

"""
Custom RSs
"""

class ChaneyContent(ContentFiltering):
    def __init__(self, item_rep_for_threshold, num_attributes, *args, **kwargs):
        if item_rep_for_threshold.shape[0] > num_attributes:
            item_rep_for_threshold = self.downscaling_attrs(item_rep_for_threshold, num_attributes)
        possible_thresholds = np.sort(item_rep_for_threshold.flatten())[::-1]
        total_items = item_rep_for_threshold.shape[1]
        for thresh in possible_thresholds[total_items:]:
            num_connected = (item_rep_for_threshold >= thresh).any(axis=0).sum()
            if num_connected >= total_items:
                self.item_attrs_binary_threshold = thresh
                break
        super().__init__(*args, **{'num_attributes': num_attributes, **kwargs})
    
    def process_new_items(self, new_items):
        if new_items.shape[0] > self.predicted_item_attributes.shape[0]:
            new_items = self.downscaling_attrs(new_items, self.predicted_item_attributes.shape[0])
        binarized_new_items = (new_items >= self.item_attrs_binary_threshold).astype(float)
        empty_interactions = sp.csr_matrix((self.num_users, binarized_new_items.shape[1]), dtype=int)
        self.all_interactions = sp.hstack([self.all_interactions, empty_interactions])
        return binarized_new_items

    def downscaling_attrs(self, item_attrs, M):
        k = M / np.gcd(item_attrs.shape[0], M)
        extended_attrs = np.repeat(item_attrs, k, axis=0)
        item_attrs = extended_attrs.reshape(-1, extended_attrs.shape[1], extended_attrs.shape[0] // M).mean(axis=2)
        return item_attrs

class ContentFilteringReduced(ContentFiltering):
    def __init__(self, item_rep_for_threshold, num_attributes, dimred="pca", *args, **kwargs):
        self.dimred = dimred
        if self.dimred == "pca":
            self.dimred_model = PCA(n_components=num_attributes) # probar t-SNE, UMAP
            self.dimred_model.fit(item_rep_for_threshold.T)
        elif self.dimred == "umap":
            self.dimred_model = umap.UMAP(n_components=num_attributes)
            self.dimred_model.fit(item_rep_for_threshold.T)
        super().__init__(*args, **{'num_attributes': num_attributes, **kwargs})
    
    def process_new_items(self, new_items):
        #print("new_items", new_items.T)
        if self.dimred == "pca":
            reduced_dimension_items = self.dimred_model.transform(new_items.T).T
        elif self.dimred == "umap":
            reduced_dimension_items = self.dimred_model.transform(new_items.T).T
        #print("reduced_dimension_items", reduced_dimension_items.T)
        empty_interactions = sp.csr_matrix((self.num_users, reduced_dimension_items.shape[1]), dtype=int)
        self.all_interactions = sp.hstack([self.all_interactions, empty_interactions])
        return reduced_dimension_items
 