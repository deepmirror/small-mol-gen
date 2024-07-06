import gzip
import os
import random
import shutil
from typing import List

import requests
from crem.crem import grow_mol, mutate_mol
from rdkit import Chem
from tqdm import tqdm

from .base import BaseGenerator


class CREM(BaseGenerator):
    def __init__(self, db_name="replacements02_sa2.db"):
        self.data_dir = "./data"
        self.db_name = os.path.join(self.data_dir, db_name)
        self.local_gz_path = f"{self.db_name}.gz"
        self.url = "https://www.dropbox.com/scl/fi/tezfk6odkqog1q4b3tip3/replacements02_sa2.db.gz?dl=1&rlkey=iryzf7irfrjpi44cf7dag8kzf"

        self._download_and_extract_db()

    def _download_and_extract_db(self):
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        # Check if the file already exists
        if not os.path.exists(self.db_name):
            print(f"Downloading {self.db_name}...")
            response = requests.get(self.url, stream=True)
            total_size = int(response.headers.get("content-length", 0))

            with open(self.local_gz_path, "wb") as file, tqdm(
                desc=self.local_gz_path,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            print(f"File downloaded and saved as {self.local_gz_path}")

            # Unzip the file
            with gzip.open(self.local_gz_path, "rb") as f_in:
                with open(self.db_name, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"File unzipped and saved to {self.data_dir}")

            # Clean up the downloaded .gz file
            os.remove(self.local_gz_path)
            print(f"Cleaned up the gz file: {self.local_gz_path}")
        else:
            print(f"{self.db_name} already exists in {self.data_dir}")

    def _sample_from_smiles(self, smi: str) -> List[str]:
        m = Chem.MolFromSmiles(smi)
        random_number = random.random()

        if random_number < 0.33333:
            mols = list(mutate_mol(m, db_name=self.db_name))
        elif random_number < 0.66666:
            mols = list(grow_mol(m, db_name=self.db_name))
        else:
            mutated = random.choice(list(mutate_mol(m, db_name=self.db_name)))
            m = Chem.MolFromSmiles(mutated)
            mols = list(grow_mol(m, db_name=self.db_name))

        return random.choices(mols, k=10)

    def generate(self, smiles: List[str]) -> List[str]:
        data = []
        for smi in tqdm(smiles, desc=f"Generating SMILES with {self.__class__.__name__}"):
            data.extend(self._sample_from_smiles(smi))
        return data
