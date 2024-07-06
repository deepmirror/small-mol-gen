# Small-Mol-Gen

## Introduction

Small-Mol-Gen is for the generation of small molecules using different AI and machine learning techniques accompanying our blog post.

## Installation

To set up the Small-Mol-Gen environment, follow these steps below. Please note that we ran the code based on macOS. If you're having any issues configuring, please reach out.

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

A few of the models in the repo call external models and you will need to obtain API keys from the following services:

1. **Anthropic API Key**

   Get your API key from Anthropic by following the instructions in their [API Getting Started Guide](https://docs.anthropic.com/en/api/getting-started).

2. **molmim API Key**

   Obtain your API key from molmim. It can be found [here](https://build.nvidia.com/nvidia/molmim-generate).

Both API keys should be added to the `.env` file:

```plaintext
ANTHROPIC_API_KEY=your_anthropic_api_key
MOLMIM_API_KEY=your_molmim_api_key
```

## Usage

### Running the Models

Before starting on the main notebook, you need to run the [`run_models.py`](https://github.com/deepmirror/small-mol-gen/blob/main/run_models.py#L1) script. This step is crucial for generating the initial output files, which may take up to an hour. However, we have included the output files for each model, so you don't need to wait.

To run the script, execute the following command:

```bash
python run_models.py
```

This script will guide you through the process and use the generative models stored in the `models/` folder.

Once this has been run (if you want to run it again, you'll need to delete the `output` folder). Otherwise, you should be good to go with the analysis interactive notebook, i.e., [`analysis.ipynb`](https://github.com/deepmirror/small-mol-gen/blob/main/analysis.ipynb#L1).

### Running Analysis

To generate results and perform analysis, run the `analysis.ipynb` notebook.

### Generative Models

All generative models used in this project are located in the `models/` folder. Therefore, you can just take them if you want to use them in your own project or do other testing.

## Platform Compatibility

This project has been tested only on macOS. If you encounter any issues running it on other operating systems, please let us know by reporting them in the issue tracker or community forums.

## Troubleshooting

- Ensure that you have the correct versions of all dependencies as specified.
- Verify that your API keys are correctly set up and have the necessary permissions.
- If you encounter any issues, please reach out.
