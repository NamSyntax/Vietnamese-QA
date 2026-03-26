from datasets import load_dataset

def get_dataset(config: dict):
    """
    Loads dataset from Hugging Face Datasets based on configuration name.
    Defaults to 'taidng/UIT-ViQuAD2.0' if not customized.
    """
    dataset_name = config["data"].get("dataset_name", "taidng/UIT-ViQuAD2.0")
    dataset = load_dataset(dataset_name)
    return dataset
