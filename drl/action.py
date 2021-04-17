import math
import numpy as np

class ActionSelector:
    """
    Abstract class with converts scores to the actions
    """
    def __call__ (self, scores):
        raise NotImplementedError

class GreedySelector (ActionSelector):
    """
    Selects actions using argmax
    """
    def __call__ (self, scores: np.ndarray):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)

class EpsilonGreedySelector (ActionSelector):
    def __init__ (self, selector: ActionSelector=GreedySelector(), epsilon: float=1.0):
        self.selector = selector
        self.epsilon = epsilon

    def __call__ (self, scores: np.ndarray):
        assert isinstance(scores, np.ndarray)
        
        batch_size, n_actions = scores.shape
        actions = self.selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        return actions


class ProbabilityActionSelector (ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__ (self, probs: np.ndarray):
        assert isinstance(prob, np.ndarray)

        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        
        return np.array(actions)
