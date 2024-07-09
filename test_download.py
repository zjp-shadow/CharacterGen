from huggingface_hub import hf_hub_download, list_repo_files

repo_id = "zjpshadow/CharacterGen"
all_files = list_repo_files(repo_id, revision="main")

for file in all_files:
    if file.startswith("2D_Stage") or file.startswith("3D_Stage"):
        hf_hub_download(repo_id, file, local_dir=".")