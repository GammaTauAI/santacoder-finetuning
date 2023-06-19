# Fine-tuning SantaCoder for TypeScript Type Inference

In this repository we are finetuning Santacoder for TypeScript Type Inference (Fill-In-The-Middle).
Our approach is to split on every type annotation and to evaluate loss on the inferred type.
We do this using tree-sitter to parse the code and to extract the type annotations.
Furthermore, we remove all the type annotations from the suffix of the FIM in order to simulate
the same scenario in training as we have in inference ([Santacoder Inference Server](https://github.com/GammaTauAI/sanatacoder-server)),
as we are filling type-holes left to right.
This approach is named Fill-In-The-Type (FIT) and will be described in a paper soon.

We are currently fine-tuning a model, which can be found [here](https://huggingface.co/gammatau/santacoder-ts-fim).

The model is trained on the [TS Training](https://huggingface.co/datasets/nuprl/ts-training) dataset,
which is a subset of The Stack dataset, containing only TypeScript code.

### Usage

To start fine-tuning, dependencies need to be installed first:

```bash
pip install -r requirements.txt
```

Before starting the fine-tuning process, the `run_type_annotation_fim.sh` script needs
to be configured to point to the correct data and model paths.
_We will release the full dataset when we can de-anonymize. We provide instructions for generating the training dataset, see the `../datasets` directory._

Then, the fine-tuning can be started by utilizing the `run_type_annotation_fim.sh` script:

```bash
./run_type_annotation_fim.sh <num_gpus> <batch_size> <num_workers>
```

For example, to start fine-tuning on 4 GPUs with a batch size of 32 and 64 cpu workers, run:

```bash
./run_type_annotation_fim.sh 4 32 64
```

#### Fine-tuning for the SantaCoder-TS Model

To fine-tune for the SantaCoder-TS model used for ablation, utilize the `./run_simple.sh` instead,
following the same steps as above.

# Original README from the SantaCoder-Finetuning repo

Fine-tune [SantaCoder](https://huggingface.co/bigcode/santacoder) on Code and Text Generation datasets. For example on new programming languages from [The Stack](https://huggingface.co/datasets/bigcode/the-stack) dataset, or on a code-to-text dataset like [GitHub-Jupyter](https://huggingface.co/datasets/codeparrot/github-jupyter-code-to-text). SantaCoder is a 1B parameters model pre-trained on Python, Java & JavaScript, we suggest fine-tuning on programming languages close to them, otherwise, the model might not converge well.

## Setup & Fine-Tuning with The Stack

We provide code to fine-tune the pre-trained [SantaCoder](https://huggingface.co/bigcode/santacoder) model on code/text datasets such as [The Stack](https://huggingface.co/bigcode/the-stack) dataset. Check this [repository](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main/finetuning) for fine-tuning models on other code tasks such as code classification.

- You can use this [**Google Colab**](https://colab.research.google.com/drive/1UMjeXHwOldpLnWjdm1499o2IYy0RgeTw?usp=sharing) by @mrm8488 for the fine-tuning.
- To train on a local machine, you can use the `train.py` script by following the steps below. It allows you to launch training using the command line on multiple GPUs.

1. To begin with, we should clone the repository locally, install all the required packages and log into HuggingFace Hub and Weight & Biases.

First, you can clone this repo with:

```
git clone https://github.com/bigcode/santacoder-finetuning.git
cd santacoder-finetuning
```

Second, install the required packages. The packages are listed in the `requirements.txt` file and can be installed with

```
pip install -r requirements.txt
```

Third, make sure you are logged to HuggingFace Hub and Weights & Biases

```
huggingface-cli login
wandb login
```

2. Next, take a look at the `train.py` script to get an understanding of how it works. In short, the script does the following:

   - Load the given dataset
   - Load the model with given hyperparameters
   - Pre-process the dataset to input into the model
   - Run training
   - Run evaluation

3. The following examples show how you can launch fine-tuning for The Stack dataset.
   Here we will run the script on the _Ruby_ subset of the dataset for demonstration purposes. Note that:

- Gradient Checkpointing is enabled by default and the caching mechanism is disabled to save memory. If you want to disable them call `no_gradient_checkpointing` argument. Note that mixed precision is disabled with the `no_fp16` flag due to some issues we noticed when using it, you can enable it by removing that argument. However, a better choice would be to use bf16 mixed precision, if it's supported on your hardware (e.g A100), it's enabled with the `bf16` flag and can be more stable in training.
- If the model still doesn't fit in your memory use `batch_size` 1 and reduce `seq_length` to 1024 for example.
- If you want to use [streaming](https://huggingface.co/docs/datasets/stream) and avoid downloading the entire dataset, add the flag `streaming`.
- If you want to train your model with Fill-In-The-Middle ([FIM](https://arxiv.org/abs/2207.14255)), use a tokenizer that includes FIM tokens, like SantaCoder's and specify the FIM rate arguments `fim_rate` and `fim_spm_rate` (by default they are 0, for SantaCoder we use 0.5 for both).

```bash
python train.py \
        --model_path="bigcode/santacoder" \
        --dataset_name="bigcode/the-stack-dedup" \
        --subset="data/shell" \
        --data_column "content" \
        --split="train" \
        --seq_length 2048 \
        --max_steps 30000 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --num_warmup_steps 500 \
        --eval_freq 3000 \
        --save_freq 3000 \
        --log_freq 1 \
        --num_workers="$(nproc)" \
	--no_fp16
```

To launch the training on multiple GPUs use the following command (we just add python -m torch.distributed.launch \--nproc_per_node number_of_gpus):

```bash
python -m torch.distributed.launch \
        --nproc_per_node number_of_gpus train.py \
        --model_path="bigcode/santacoder" \
        --dataset_name="bigcode/the-stack-dedup" \
        --subset="data/shell" \
        --data_column "content" \
        --split="train" \
        --seq_length 2048 \
        --max_steps 30000 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --num_warmup_steps 500 \
        --eval_freq 3000 \
        --save_freq 3000 \
        --log_freq 1 \
        --num_workers="$(nproc)" \
	--no_fp16
```

Note: The checkpoints saved from this training command will have argument `use_cache` in the file `config.json` as `False`, for fast inference you should change it to `True` like in this [commit](https://huggingface.co/arjunguha/santacoder-lua/commit/e57b3c39fd29e36ba86970e49618448f5d3d5529) or add it each time you're loading the model.

If you want to fine-tune on other text datasets, you just need to change `data_column` argument to the name of the column containing the code/text you want to fine-tune on.

For example, We fine-tuned the model on the [GitHub-Jupyter](https://huggingface.co/datasets/codeparrot/github-jupyter-code-to-text) dataset on 4 A100 using the following command:

```bash
python -m torch.distributed.launch \
        --nproc_per_node 4 train.py \
        --model_path="bigcode/santacoder" \
        --dataset_name="codeparrot/github-jupyter-code-to-text" \
        --data_column "content" \
        --split="train" \
        --seq_length 2048 \
        --max_steps 1000 \
        --batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-5 \
        --num_warmup_steps 100 \
        --eval_freq 100 \
        --save_freq 100 \
        --log_freq 1 \
        --num_workers="$(nproc)" \
        --no_fp16
```

The resulting model can be found [here](https://huggingface.co/loubnabnl/santacoder-code-to-text) with an associated [space](https://huggingface.co/spaces/loubnabnl/santa-explains-code).

**Can I use another Model**: Yes! you can use other CLM models on the hub such as GPT2, CodeParrot, CodeGen, InCoder... Just make sure to change the `seq_length` and `eos_token_id` arguments.

## How to upload my trained checkpoint

To upload your trained checkpoint, you have to create a new model repository on the 🤗 model hub, from this page: https://huggingface.co/new

> You can also follow the more in-depth instructions [here](https://huggingface.co/transformers/model_sharing.html) if needed.

Having created your model repository on the hub, you should clone it locally:

```bash
git lfs install

git clone https://huggingface.co/username/your-model-name
```

Then and add the following files that fully define a SantaCoder checkpoint into the repository. You should have added the following files.

- `tokenizer_config.json`
- `tokenizer.json`
- `config.json`
- `pytorch_model.bin`
- modeling files (see below)

Note: As previously stated, the checkpoints saved from this training with gradient checkpointing and no caching command will have argument `use_cache` in the file `config.json` as `False`, for fast inference you should change it to `True` like in this [commit](https://huggingface.co/arjunguha/santacoder-lua/commit/e57b3c39fd29e36ba86970e49618448f5d3d5529).

You can get the tokenizer files by cloning the [model repo](https://huggingface.co/bigcode/santacoder/tree/main) and copying them to your directory. Santacoder currently has a custom [modeling file](https://huggingface.co/bigcode/santacoder/blob/main/modeling_gpt2_mq.py) + config file on the hub, but they will be included with the saved checkpoints if you used the `transformers` branch in `requirements.txt`.

Having added the above files, you should run the following to push files to your model repository.

```
git add . && git commit -m "Add model files" && git push
```

The next **important** step is to create the model card. For people to use your fine-tuned
model it is important to understand:

- What kind of model is it?
- What is your model useful for?
- What data was your model trained on?
- How well does your model perform?

All these questions should be answered in a model card which is the first thing people see when
visiting your model on the hub under `https://huggingface.co/{your_username}/{your_modelname}`.

Don't hesitate to also create a Gradio Demo for your model to showcase its capabilities 🚀. You can find more information on how to do that [here](https://huggingface.co/docs/hub/spaces-sdks-gradio).

## Acknowledgments

This is inspired by the [Wave2vec fine-tuning week](https://github.com/huggingface/transformers/edit/main/examples/research_projects/wav2vec2/) by [Hugging Face](https://huggingface.co/).
