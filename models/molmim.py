import ast
from typing import List

import requests
from tqdm import tqdm

from .base import BaseGenerator


class MOLMIM(BaseGenerator):

    invoke_url = "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate"

    def __init__(self, api_key, algorithm="none", num_molecules=10, minimize=False, min_similarity=0.7, particles=30, iterations=10):
        self.api_key = api_key
        self.algorithm = algorithm
        self.num_molecules = num_molecules
        self.minimize = minimize
        self.min_similarity = min_similarity
        self.particles = particles
        self.iterations = iterations
        self._set_headers()

    def _set_headers(self):
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def _sample_from_smiles(self, smi: str) -> List[str]:
        payload = {
            "algorithm": self.algorithm,
            "num_molecules": self.num_molecules,
            "minimize": self.minimize,
            "min_similarity": self.min_similarity,
            "particles": self.particles,
            "iterations": self.iterations,
            "smi": smi,
        }

        session = requests.Session()
        response = session.post(self.invoke_url, headers=self.headers, json=payload)
        response.raise_for_status()
        response_body = response.json()

        mols = []
        samples = ast.literal_eval(response_body["molecules"])
        for sample in samples:
            mols.append(sample["sample"])
        return mols

    def generate(self, smiles: List[str]) -> List[str]:
        data = []
        for smi in tqdm(smiles, desc=f"Generating SMILES with {self.__class__.__name__}"):
            data.extend(self._sample_from_smiles(smi))
        return data
