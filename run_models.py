import os

import dotenv
import pandas as pd
from tqdm import tqdm

from models import CREM, MOLMIM, Claude, Reinvent

# Load environment variables
dotenv.load_dotenv()

# Ensure necessary API keys are set
assert os.getenv("ANTHROPIC_API_KEY") is not None, "Please set the ANTHROPIC_API_KEY environment variable"
assert os.getenv("MOLMIM_API_KEY") is not None, "Please set the MOLMIM_API_KEY environment variable"

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Define dataset file paths
DATASET_FILE = {
    "a2a": "data/adenosineA2A.csv",
    "aryl": "data/Aryl piperazine.csv",
    "sirt2": "data/SIRT2.csv",
}

# Initialize models with respective API keys
molmim = MOLMIM(api_key=os.getenv("MOLMIM_API_KEY"))
claude = Claude(api_key=os.getenv("ANTHROPIC_API_KEY"), scaffold_hop=False)
claude_scaffold = Claude(api_key=os.getenv("ANTHROPIC_API_KEY"), scaffold_hop=True)
reinvent = Reinvent(config_filename="models/reinvent/config.toml")
crem = CREM()

# Store models in a dictionary for iteration
MODELS = {"molmim": molmim, "claude": claude, "claude_scaffold": claude_scaffold, "reinvent": reinvent, "crem": crem}

# Process each dataset with each model
for dataset_name, path in tqdm(DATASET_FILE.items(), desc="Datasets"):
    df = pd.read_csv(path)
    smiles = df["Smiles"].tolist()
    for model_name, model in tqdm(MODELS.items(), desc=f"Models for {dataset_name}"):
        output_file = os.path.join("output", f"{dataset_name}_{model_name}.csv")
        if not os.path.exists(output_file):
            generated_smiles = model.generate(smiles)
            output_df = pd.DataFrame({"Smiles": generated_smiles})
            output_df["Dataset"] = dataset_name
            output_df["Model"] = model_name
            output_df.to_csv(output_file, index=False)
