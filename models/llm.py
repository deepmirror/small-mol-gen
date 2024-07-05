from typing import List

import anthropic
from rdkit import Chem
from tqdm import tqdm

from .base import BaseGenerator


class Claude(BaseGenerator):

    N_PER_SMILES = 10

    def __init__(self, api_key: str, scaffold_hop: bool, model="claude-3-5-sonnet-20240620", max_tokens=1024):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.scaffold_hop = scaffold_hop

    def _sample_from_smiles(self, smi: str, scaffold_hop: bool) -> List[str]:
        sampled = []
        prompt_template = (
            (
                f"Generate me {self.N_PER_SMILES} molecules represented as *valid* SMILES strings that are similar to this SMILES string: {smi}",
                "I have generated {count} valid SMILES string representing similar molecules. How do you want me to return them to you?",
                "Return the SMILES strings as a sequence of strings in your next reply and no other characters! Like this SMILES,SMILES,SMILES,...",
            )
            if not scaffold_hop
            else (
                f"Generate me {self.N_PER_SMILES} molecules represented as *valid* SMILES strings that are similar to this SMILES string: {smi}. Try to integrate novel scaffolds.",
                "I have generated {count} valid SMILES string representing similar molecules with novel scaffolds. How do you want me to return them to you?",
                "Return the SMILES strings as a sequence of strings in your next reply and no other characters! Like this SMILES,SMILES,SMILES,...",
            )
        )

        user_prompt, assistant_response, user_confirmation = prompt_template

        while len(sampled) < self.N_PER_SMILES:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response.format(count=self.N_PER_SMILES)},
                    {"role": "user", "content": user_confirmation},
                ],
            )

            strings = message.content[0].text.split(",")
            for string in strings:
                try:
                    mol = Chem.MolFromSmiles(string)
                    if mol:
                        sampled.append(Chem.MolToSmiles(mol))
                except:
                    continue
        return sampled

    def generate(self, smiles: List[str]) -> List[str]:
        data = []
        for smi in tqdm(smiles, desc=f"Generating SMILES with {self.__class__.__name__}"):
            data.extend(self._sample_from_smiles(smi, self.scaffold_hop))

        return data
