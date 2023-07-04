from abc import ABCMeta, abstractmethod, ABC
from typing import Any
from gym import Env


class Environment(Env, ABC):

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self, mode: str = "human") -> Any:
        pass

    @property
    @abstractmethod
    def observation(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def current_player_num(self) -> int:
        pass

    @property
    @abstractmethod
    def n_players(self) -> int:
        pass

    @property
    @abstractmethod
    def players(self) -> []:
        pass

    # @abstractmethod
    # @property
    # def current_player(self):
    #     pass

    # @abstractmethod
    # @property
    # def legal_actions(self):
    #     pass

    # @abstractmethod
    # def rules_move(self):
    #     pass
