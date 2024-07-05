# Small-Mol-Gen Documentation

## Introduction

Small-Mol-Gen is a project designed for the generation of small molecules using different AI and machine learning techniques.

## Installation

To set up the Small-Mol-Gen environment, follow these steps:

1. **Create a Conda Environment**

   Create a new Conda environment named `small-mol-gen` with Python 3.10:

   ```bash
   conda create -n small-mol-gen python==3.10
   ```

2. **Activate the Conda Environment**

   Activate the newly created Conda environment:

   ```bash
   conda activate small-mol-gen
   ```

3. **Install Required Packages**

   Install the required packages listed in the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install REINVENT Package**

   Install the patched version of the REINVENT package from GitHub to ensure compatibility with macOS:

   ```bash
   pip install --no-deps git+https://github.com/rg314/REINVENT4.git@macos-bugfix-torch#egg=reinvent
   ```

## API Keys

To utilize the full functionality of Small-Mol-Gen, you will need to obtain API keys from the following services:

1. **Anthropic API Key**

   Get your API key from Anthropic by following the instructions in their [API Getting Started Guide](https://docs.anthropic.com/en/api/getting-started).

2. **MOLMIN API Key**

   Obtain your API key from MOLMIN. (Further details will need to be provided for the specific URL or process to obtain this key.)

## Usage

### Running Analysis

To generate results and perform analysis, run the `analysis.ipynb` notebook. This script will guide you through the process and use the generative models stored in the `models/` folder.

### Generative Models

All generative models used in this project are located in the `models/` folder. These models are essential for the generation of small molecules and are utilized by the scripts and notebooks provided.

## Platform Compatibility

This project has been tested only on macOS. If you encounter any issues running it on other operating systems, please let us know by reporting them in the issue tracker or community forums.

## Troubleshooting

- Ensure that you have the correct versions of all dependencies as specified.
- Verify that your API keys are correctly set up and have the necessary permissions.
- If you encounter any issues, refer to the project's issue tracker or community forums for support.
