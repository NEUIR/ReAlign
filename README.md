# ReAlign

Training and evaluation of the multimodal document retriever ReAlign.

## Environment

From the project root, create and activate a Conda environment (Python 3.10):

```bash
cd realign
conda create -n realign python=3.10 -y
conda activate realign
```

Install dependencies and the editable package:

```bash
pip install -r requirements.txt
pip install -e .
```

## Data and Model Paths

`config/dir_config.sh` is the configuration file for model and data directories. Please download the required assets and set the paths according to the instructions in that file.

All absolute paths for data and checkpoints are centralized in [`config/dir_config.sh`](config/dir_config.sh). Updating a path only requires editing that single file.

## Training

Create the log directory if it does not exist:

```bash
mkdir -p log
```

Phi-3 Vision training example:

```bash
bash train_kl_phi3v.sh > log/realign-phi3v.log 2>&1
```

Qwen training example (enable as needed):

```bash
bash train_kl_qwen.sh > log/realign-qwen.log 2>&1
```

## Evaluation

The second argument of each evaluation script is a comma-separated list of GPU IDs. The examples below use four GPUs; adjust to match your hardware (e.g. use `0` for a single GPU).

```bash
bash sh/eval.sh realign-phi3v 0,1,2,3
bash sh/eval_qwen.sh realign-qwen 0,1,2,3
```
