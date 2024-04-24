import numpy as np
from trecs.models import ContentFiltering
from scipy.optimize import nnls

"""
RSs taken from algo_confounding
"""
class ChaneyContent(ContentFiltering):
    """
    Chaney ContentFiltering model which uses NNLS solver
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