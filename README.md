# LM Peel

A deeper understanding of language model behaviors, especially as related to numeric generation.
This work corresponds to the IPDPS 2025 Workshop HPAI4S paper, "Is In-Context Learning Feasible for HPC Performance Autotuning?" by Thomas Randall, Akhilesh Bondapalli, Rong Ge and Prasanna Balaprakash.

## Setup

### Python Environment

Our Python environment is based on Python 3.11.9, with all dependent modules documented in `requirements.txt`.
We do not rigorously check for forward/backward compatibility of these dependencies, but do not anticipate our usage of these packages to be subject to near-term deprecation.

### Models

We utilize the HuggingFace Meta-LLaMa 3.1 8B-Instruct model as the LLM in our experiments.
To download and install this model for your own use, please consult the [HuggingFace website](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

### Artifacts

To assist in replication and extension of our work, we provide dataset artifacts via Zenodo [https://doi.org/10.5281/zenodo.14878735](https://doi.org/10.5281/zenodo.14878735).
These artifacts include the preprocessed LLM generations, response editing, and numeric generation associated with the paper experiments.
They are not strictly required to replicate the work, but can greatly accelerate the process.

## Repository Organization

### Preprocessing

We provide the script `dataset_to_llm_format.py` which we utilized to prepare our CSV dataset for utilization with the LLM.
If you are using the provided artifacts, the given datasets are already converted for you as documented in the `documenting.txt` file.
If you are attempting to utilize a new dataset, the script's usage is: `python3 dataset_to_llm_format.py --in <YOUR_ORIGINAL_CSV> --out <YOUR_NEW_CSV>`.
For additional arguments and options, run `python3 dataset_to_llm_format.py --help`.

### Main Experiments

Our main results are all driven from the python script `drive_syr2k_icl.py`.
Due to the complexity of operating this script, we also provide the meta-driver `icl_scan.sh`, which composes and executes all variations of the python script necessary to reproduce our results.

**PLEASE REFER TO THE `CacheNotice.txt` PRIOR TO RUNNING THE BASH SCRIPT.**

If you wish to run the python script for your own purposes, refer to `python3 drive_syr2k_icl.py --help` for a complete listing of available options.
If you also intend to run the supporting analyses, you should capture all text output produced during the `icl_scan.sh` execution, ie: `./icl_scan.sh | tee my_log.txt`.

The XGBoost results are separately driven by the script `predict_with_xgboost.py`, expected usage is `python3 predict_with_xgboost.py`.

### Supporting Analyses

Two forms of analyses presented in the paper are based on the overall results spanning multiple experiments.
As such, they are not driven by a single call to `drive_syr2k_icl.py` and are instead run on the logs spanning all experiments captured by `icl_scan.sh`.

* `clt_parse.py`: Extracts and analyzes the central-limit-theorem data presented in the paper. Expected usage: `python3 clt_parse.py <PREVIOUSLY_COLLECTED_TEXT_LOG>`
* `haystack_posthoc.py`: Extracts and analyzes the "needles in a haystack" data presented in the paper. Expected usage: `python3 haystack_posthoc.py <PREVIOUSLY COLLECTED_TEXT_LOG>`

### Supporting Modules

The following python modules are provided as means to separate code from other components, and are local dependencies for other scripts within the repository:
* `interactive_text_editor.py`: Provides a text-editing interface that facilitates quickly trimming LLM outputs to the relevant portions for this work.
* `peeled_huggingface.py`: Provides an object interface that wraps HuggingFace's transformers API to permit our experimental post-hoc logit inspections.
* `pickle_cache.py`: Defines a simple protocol for saving and retrieving LLM responses and generated distributions based on LLM logits and tokens, utilized to reduce the computational burden and accelerate replication efforts.
* `timerdict.py`: Defines a simple timestamping class used to track execution times of other scripts.
