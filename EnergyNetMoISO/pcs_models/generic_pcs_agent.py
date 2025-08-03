from abc import ABC, abstractmethod

class GenericPCSAgent(ABC):

    @abstractmethod
    def predict(self, obs, deterministic=True, **kwargs):
        """
        Predict the action for the given observation.
        
        Args:
            obs: The observation from the environment
            deterministic: Whether to use deterministic action selection
            **kwargs: Additional keyword arguments
            
        Returns:
            tuple: (action, state) where state can be None for stateless agents
        """
        pass