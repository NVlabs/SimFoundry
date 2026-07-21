from abc import ABC, abstractmethod


class InferenceClient(ABC):
    @abstractmethod
    def __init__(self, args) -> None:
        """
        Initializes the client.
        """
        pass

    @abstractmethod
    def infer(self, obs, instruction) -> dict:
        """
        Does inference on observation and returns the final processed
        dictionary used to do inference.
        """

        pass

    @abstractmethod
    def reset(self):
        """
        Resets the client to start a new episode.
        """
        pass

