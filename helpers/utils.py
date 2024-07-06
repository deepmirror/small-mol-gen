import glob
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap
import useful_rdkit_utils as uru
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import BulkTanimotoSimilarity

FORMATTED_NAMES = {
    "original": "Original",
    "molmim": "MolMIM",
    "claude": "Claude",
    "claude_scaffold": "Claude Scaffold",
    "crem": "CREM",
    "reinvent": "Reinvent",
}

PALETTE = {
    "original": "#4B444A",
    "molmim": "#5359CC",
    "claude": "#DA6AF7",
    "claude_scaffold": "#AA6BE0",
    "crem": "#FBAC3B",
    "reinvent": "#F94156",
}


def plot_boxenplot(df: pd.DataFrame, x: str, y: str, hue: str, title: str, order: List[str]) -> None:
    hue_values = df[hue].unique()
    custom_palette = {hue_value: PALETTE[hue_value] for hue_value in hue_values if hue_value in PALETTE}
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxenplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=custom_palette, hue_order=order)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    ax.set_title(title)
    plt.tight_layout()


def canonicalize_and_validate_smiles(smiles: str) -> str:
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule:
            return Chem.MolToSmiles(molecule)
    except:
        pass
    return np.nan


def smiles_to_fp(smiles_series: pd.Series) -> List[AllChem.GetMorganFingerprintAsBitVect]:
    fingerprints = []
    for smile in smiles_series:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fingerprints.append(fp)
    return fingerprints


def calculate_tanimoto_similarities(reference_smiles: str, original_smiles_list: List[str]) -> List[float]:
    ref_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reference_smiles), radius=2, nBits=1024)
    fps_list = smiles_to_fp(pd.Series(original_smiles_list))
    return BulkTanimotoSimilarity(ref_fp, fps_list)


def make_umap(smiles_series: pd.Series, source_labels: List[str], n_neighbors: int, dataset: str, results_dir: str) -> None:
    save_path = os.path.join(results_dir, f"{dataset}_umap_data.csv")
    if not os.path.exists(save_path):
        fps = smiles_to_fp(smiles_series)
        if not fps:
            return
        fps_array = np.array(fps)
        umap_reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=20)
        umap_results = umap_reducer.fit_transform(fps_array)
        umap_df = pd.DataFrame(umap_results, columns=["UMAP1", "UMAP2"])
        umap_df["Source"] = [label for label, fp in zip(source_labels, fps) if fp is not None]
        umap_df.to_csv(os.path.join(results_dir, f"{dataset}_umap_data.csv"), index=False)


def plot_only_one_method_umap(umap_df: pd.DataFrame, grid_axs: plt.Axes, only_plot: str) -> None:
    df = umap_df[umap_df["Source"].isin(["original", only_plot])]
    sns.scatterplot(
        data=df,
        x="UMAP1",
        y="UMAP2",
        hue="Source",
        edgecolor=None,
        s=15,
        palette=PALETTE,
        alpha=0.3,
        ax=grid_axs,
    )


def plot_umap(dataset_name: str, results_dir: str) -> None:
    umap_datasets = glob.glob(os.path.join(results_dir, "*.csv"))
    fig, axs = plt.subplots(2, 3, figsize=(10, 7.5))
    axs = axs.flatten()
    subplot_idx = 0

    for dataset_path in umap_datasets:
        umap_df = pd.read_csv(dataset_path)
        if os.path.basename(dataset_path).split("_")[0] != dataset_name:
            continue

        for method in umap_df["Source"].unique():
            plot_only_one_method_umap(umap_df, grid_axs=axs[subplot_idx], only_plot=method)
            axs[subplot_idx].legend().remove()
            axs[subplot_idx].set_title(FORMATTED_NAMES.get(method, method), fontsize=12)
            subplot_idx += 1

    plt.tight_layout()


def extract_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def extract_scaffold_skeleton(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
    return Chem.MolToSmiles(scaffold)


def calculate_novelty_rate(original: set, generated: set) -> float:
    novel_scaffolds = generated - original
    return len(novel_scaffolds) / len(generated) if generated else 0


def compute_novelty_rates(df: pd.DataFrame, groupby: str = "Dataset", algorithm: str = "Model") -> pd.DataFrame:
    scaffold_novelty_rate_list = []
    skeleton_novelty_rate_list = []
    dataset_list = []
    method_list = []

    for dataset, dataset_df in df.groupby(groupby):
        original_df = dataset_df[dataset_df[algorithm] == "original"]
        original_scaffolds = set(original_df["Smiles"].apply(extract_scaffold))
        original_skeletons = set(original_df["Smiles"].apply(extract_scaffold_skeleton))

        for method in dataset_df[algorithm].unique():
            if method == "original":
                continue

            generated_df = dataset_df[dataset_df[algorithm] == method]
            generated_scaffolds = set(generated_df["Smiles"].apply(extract_scaffold))
            generated_skeletons = set(generated_df["Smiles"].apply(extract_scaffold_skeleton))

            scaffold_novelty_rate_list.append(calculate_novelty_rate(original_scaffolds, generated_scaffolds))
            skeleton_novelty_rate_list.append(calculate_novelty_rate(original_skeletons, generated_skeletons))

            dataset_list.append(dataset)
            method_list.append(method)

    return pd.DataFrame(
        {
            algorithm: method_list,
            "Dataset": dataset_list,
            "Scaffold novelty rate": scaffold_novelty_rate_list,
            "Skeleton novelty rate": skeleton_novelty_rate_list,
        }
    )


def process_rare_rings(df: pd.DataFrame, smiles_column: str) -> pd.DataFrame:
    ring_system_lookup = uru.RingSystemLookup.default()
    df["ring_systems"] = df[smiles_column].apply(ring_system_lookup.process_smiles)
    df[["min_ring", "min_freq"]] = df["ring_systems"].apply(uru.get_min_ring_frequency).to_list()
    df["rare_ring"] = df["min_freq"] < 100

    return df
