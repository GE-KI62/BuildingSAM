import torch
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from src.segment_anything import build_sam_vit_b
from src.lora import LoRA_sam
import glob
import random
import rasterio as rio
from modules.config import Config

def dummy(image):
	return image, dummy
	
def hf(image):
	return np.flip(image, axis=2), hf

def vf(image):
	return np.flip(image, axis=1), vf

def rr(image, k=random.randint(1, 3)):
	rotated_image = np.rot90(image, k=k, axes=(1, 2))
	return rotated_image, lambda x: rr(x, k=-k)

FN_MAPPING = {
	"HorizontalFlip": hf,
	"VerticalFlip": vf,
	"RandomRotate90": rr
}

def infer(inference_config, model_config):
	os.makedirs(inference_config.output_dir, exist_ok=True)
	device = "cuda" if model_config.device == "cuda" and torch.cuda.is_available() else "cpu"
	transforms = [dummy]

	if inference_config.augmentations:
		transforms.extend([FN_MAPPING[x] for x in inference_config.augmentations])

	
	imgs = glob.glob(os.path.join(inference_config.image_dir, "*" + inference_config.image_format))
	print(f"Found {len(imgs)} images for processing in {inference_config.image_dir}...")
	sam = build_sam_vit_b(checkpoint=model_config.load_pth_from)
	sam_lora = LoRA_sam(sam, model_config.rank)
	sam_lora.load_lora_parameters(model_config.load_checkpoint_from)
	model = sam_lora.sam
	sam.to(device)

	with torch.no_grad():
		for path in tqdm(imgs):
			with rio.open(path, "r") as f: 
				image = f.read()
				mask = f.read_masks(1)
				meta = f.meta.copy() 
				meta.update({ "count": 1, "compress": 'lzw' })
			
				pred = np.zeros((1, f.height, f.width))
	
				for t in transforms:
					transformed, undo = t(image[:3])  
					data = torch.Tensor(transformed.copy()).to(device) 
 
					inputs = [{
						"image": data,
						"original_size": (f.height, f.width),
						"boxes": torch.Tensor([[[0, 0, f.height, f.width]]]).to(device),
						"ground_truth_mask": torch.ones_like(data)
					}]
		
					outputs = model(batched_input=inputs, multimask_output=False)   
					output = outputs[0][inference_config.output_type].to("cpu")[0].detach().numpy().astype(np.float32)
					output, _ = undo(output)
					pred += output

				pred /= len(transforms) 
				pred *= 255
				with rio.open(os.path.join(inference_config.output_dir, Path(path).name), "w", **meta) as result:
					result.write(pred)
					result.write_mask(mask)

	print("Success")



if __name__ == "__main__":
	config = Config.from_yaml("config2.yaml")
	infer(config.inference, config.model)
