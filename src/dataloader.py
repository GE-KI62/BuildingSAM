import torch
import glob
import os 
import numpy as np
from src.processor import Samprocessor
import src.utils as utils
import rasterio as rio
from torch.utils.data import Dataset


class DatasetSegmentation(Dataset):
    """
    Dataset to process the images and masks

    Arguments:
        folder_path (str): The path of the folder containing the images
        processor (obj): Samprocessor class that helps pre processing the image, and prompt 
    
    Return:
        (dict): Dictionnary with 4 keys (image, original_size, boxes, ground_truth_mask)
            image: image pre processed to 1024x1024 size
            original_size: Original size of the image before pre processing
            boxes: bouding box after adapting the coordinates of the pre processed image
            ground_truth_mask: Ground truth mask
    """

    def __init__(self, img_paths, msk_paths, processor: Samprocessor, transform = None):
        super().__init__()
        self.img_paths = img_paths
        self.mask_paths = msk_paths
        self.processor = processor
        self.transform = transform


    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index: int) -> list:
        image_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        meta = None

        with rio.open(image_path, "r") as img:
            with rio.open(mask_path, "r") as mask:
                image = img.read()
                gt = mask.read()
                meta = img.meta.copy()

                if np.argmin(image.shape) == 0:
                    image = np.moveaxis(image, 0, 2)
                    if gt.ndim == 3:
                        gt = np.moveaxis(gt, 0, 2)

                if self.transform is not None:
                    transformed = self.transform(image=image, mask=gt)
                    image = transformed["image"].astype(np.uint8)
                    gt = transformed["mask"]

                gt = gt.squeeze()
                box = utils.get_bounding_box(gt)
                inputs = self.processor(image, image.shape[:-1], box)
                inputs["ground_truth_mask"] = torch.from_numpy(gt).to(torch.bool)
                return (self.img_paths[index], inputs, meta)

    
def collate_fn(batch: torch.utils.data) -> list:
    """
    Used to get a list of dict as output when using a dataloader

    Arguments:
        batch: The batched dataset
    
    Return:
        (list): list of batched dataset so a list(dict)
    """
    return batch