from abc import ABC, abstractmethod
from typing import List


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, smiles: List[str]) -> List[str]:
        pass
