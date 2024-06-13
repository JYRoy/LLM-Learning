from huggingface_hub import snapshot_download

snapshot_download(repo_id="openai-community/gpt2", local_dir="./ckpt")