import os


def create_folder_structure():
    folders = [
        "data/external", "data/interim", "data/processed", "data/raw",
        "envs",
        "docs",
        "models",
        "notebooks",
        "references",
        "reports/figures",
        "src",
        "src/modeling"
    ]

    for folder in folders:
        os.makedirs(os.path.join(folder), exist_ok=True)

    files = ["Makefile", "requirements.txt",
             "envs/.dev_env", "envs/.prod_env", "envs/.dev2_env",
             "envs/dev_req.txt", "envs/prod_req.txt", "envs/dev2_req.txt",
             "src/__init__.py", "src/config.py", "src/dataset.py",
             "src/features.py", "src/modeling/__init__.py",
             "src/modeling/predict.py", "src/modeling/train.py",
             "src/plots.py"]

    for file in files:
        open(os.path.join(file), 'a').close()


create_folder_structure()