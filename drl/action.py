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
        return np.argmax(scores)

class EpsilonGreedySelector (ActionSelector):
    def __init__ (self, selector: ActionSelector=GreedySelector(), epsilon: float=1.0):
        self.selector = selector
        self.epsilon = epsilon

    def __call__ (self, scores: np.ndarray):
        assert isinstance(scores, np.ndarray)
        
        if np.random.random() < self.epsilon:
            action = np.random.randint(scores.size)
        else:
            action = self.selector(scores)

        return action


class ProbabilityActionSelector (ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__ (self, prob: np.ndarray):
        assert isinstance(prob, np.ndarray)
        action = np.random.choice(len(prob), p=prob)
        
        return action

class EpsilonTracker:
    def __init__ (self, selector: GreedySelector, eps_start: float=1.0, eps_end: float=0.01, decay_steps: int=1000, lin: bool=True):
        self.selector = selector
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_steps = decay_steps
        self.lin = lin
        self.decay_eps(0)

    def decay_eps (self, step: int):
        if self.lin:
            decay_ratio = step / self.decay_steps
        else:
            step = max(step, 1)
            decay_ratio = math.log(step, self.decay_steps)
        eps = self.eps_start - (self.eps_start - self.eps_end) * decay_ratio
        eps = max(eps, self.eps_end)

        self.selector.epsilon = eps
