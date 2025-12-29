"""Module to handle downloading datasets from Kaggle."""
import kaggle

def download_dataset(dataset_name: str, download_path: str) -> None:
    """
    Download a dataset from Kaggle.

    Args:
        dataset_name (str): The name of the dataset on Kaggle (e.g., 'username/dataset-name').
        download_path (str): The local path where the dataset should be downloaded.
    """
    try:
        kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        print(f"Dataset '{dataset_name}' downloaded successfully to '{download_path}'.")
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")

