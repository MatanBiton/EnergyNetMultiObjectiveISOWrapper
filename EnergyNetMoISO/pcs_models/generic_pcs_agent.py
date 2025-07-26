from abc import ABC, abstractmethod

class GenericPCSAgent(ABC):

    @abstractmethod
    def predict(self,obs, deterministice, **kwargs):
        pass