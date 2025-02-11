from huggingface_hub import hf_hub_download


required_files = [
    "config.json",
    "generation_config.json",
    "model-00001-of-00002.safetensors",  # recommend using the *.safetensors files for safety reasons.
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
]

local_dir = None  # directory if you want to download the files to a specific location

for filename in required_files:
    hf_hub_download(repo_id="google/flan-t5-xl", filename=filename, local_dir=local_dir)
