import torch
import rasterio as rio
import glob
import os
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from pathlib import Path

import src.utils as utils
from modules.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b
from src.lora import LoRA_sam

FN_MAPPING = {
	"RandomSizedCrop": A.RandomSizedCrop,
	"HorizontalFlip": A.HorizontalFlip,
	"VerticalFlip": A.VerticalFlip,
	"RandomRotate90": A.RandomRotate90,
	"Transpose": A.Transpose,
	"RandomBrightnessContrast": A.RandomBrightnessContrast,
	"CoarseDropout": A.CoarseDropout
}

def train(train_config, model_config):
	sam = build_sam_vit_b(checkpoint=model_config.load_pth_from)
	sam_lora = LoRA_sam(sam, model_config.rank)

	num_epochs = train_config.num_epochs
	device = "cuda" if model_config.device == "cuda" and torch.cuda.is_available() else "cpu"
	model_name = train_config.model_dir.name
	transform = None

	if train_config.augmentations:
		transform = A.Compose([
			FN_MAPPING[a.name](**{k: v for k, v in a.__dict__.items() if k != "name"}) for a in train_config.augmentations
		])
		

	if model_config.load_checkpoint_from:
		sam_lora.load_lora_parameters(model_config.load_checkpoint_from)
		print(f"LORA parameters loaded from {model_config.load_checkpoint_from}")

	model = sam_lora.sam
	processor = Samprocessor(model)


	train_image_paths = glob.glob(os.path.join(train_config.train_dataset.image_dir, "*" + train_config.image_extension))
	train_target_paths = glob.glob(os.path.join(train_config.train_dataset.target_dir, "*" + train_config.image_extension))
	print(f"Found {len(train_image_paths)} (input) | {len(train_target_paths)} (target) | training images")

	if train_config.val_dataset:
		val_image_paths = glob.glob(os.path.join(train_config.val_dataset.image_dir, "*" + train_config.image_extension))
		val_target_paths = glob.glob(os.path.join(train_config.val_dataset.target_dir, "*" + train_config.image_extension))
		print(f"Found {len(val_image_paths)} (input) | {len(val_target_paths)} (target) | validation images")

	train_ds = DatasetSegmentation(train_image_paths, train_target_paths, processor, transform=transform)
	train_dataloader = DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True, collate_fn=collate_fn)

	val_dataloader = None

	if train_config.val_dataset:
		val_ds = DatasetSegmentation(val_image_paths, val_target_paths, processor)
		val_dataloader = DataLoader(val_ds, batch_size=train_config.batch_size, shuffle=True, collate_fn=collate_fn)


	vis_config = train_config.visualization
	if vis_config:
		os.makedirs(vis_config.output_dir, exist_ok=True)

	optimizer = Adam(model.image_encoder.parameters(), lr=train_config.learning_rate, weight_decay=0)
	seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

	model.to(device)
	os.makedirs(train_config.model_dir, exist_ok=True)

	writer = None
	if train_config.tensorboard_log_dir:
		writer = SummaryWriter(os.path.join(train_config.tensorboard_log_dir, model_name))
	
	for epoch in range(1, num_epochs + 1):
		print(f'EPOCH: {epoch}')

		model.train()
		train_losses = []

		print("[Training]")	
		for stack in tqdm(train_dataloader):
			filepaths, inputs, meta = zip(*stack)
			outputs = model(batched_input=inputs,
							multimask_output=False)
			stk_gt, stk_out = utils.stacking_batch(inputs, outputs)

			stk_out = stk_out.squeeze(1)
			stk_gt = stk_gt.unsqueeze(1) 
			loss = seg_loss(stk_out, stk_gt.float().to(device))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			train_losses.append(loss.item())

			if vis_config and epoch % vis_config.save_each_n_epoch == 0:
				output_dir = os.path.join(vis_config.output_dir, str(epoch), "train_dataset")
				os.makedirs(output_dir, exist_ok=True)
				for i in range(len(filepaths)):
					m = meta[i].copy()
					m.update({
						"count": 1,
						"compression": "lzw"
					})
					mask = outputs[i][vis_config.output_type].to("cpu")[0].detach().numpy() * 255
					with rio.open(os.path.join(output_dir, Path(filepaths[i]).name), "w", **m) as f:
						f.write(mask)

		print(f'Mean loss training: {mean(train_losses):.3f}')

		if writer:
			writer.add_scalar("Loss/train", mean(train_losses), epoch)

		if val_dataloader:
			val_loss = []
	
			with torch.no_grad():
				model.eval()
				print("[Validation]")	
				for stack in tqdm(val_dataloader):
					filepaths, inputs, meta = zip(*stack)
					outputs = model(batched_input=inputs,
					  		multimask_output=False)

					stk_gt, stk_out = utils.stacking_batch(inputs, outputs)
					stk_out = stk_out.squeeze(1)
					stk_gt = stk_gt.unsqueeze(1) 
					loss = seg_loss(stk_out, stk_gt.float().to(device))

					val_loss.append(loss.item())

					if vis_config and epoch % vis_config.save_each_n_epoch == 0:
						output_dir = os.path.join(vis_config.output_dir, str(epoch), "val_dataset")
						os.makedirs(output_dir, exist_ok=True)
						for i in range(len(filepaths)):
							m = meta[i].copy()
							m.update({
								"count": 1,
								"compression": "lzw"
							})
							mask = outputs[i][vis_config.output_type].to("cpu")[0].detach().numpy() * 255
							with rio.open(os.path.join(output_dir, Path(filepaths[i]).name), "w", **m) as f:
								f.write(mask)

				print(f'Mean loss validation: {mean(val_loss):.3f}') 
				if writer:
					writer.add_scalar("Loss/val", mean(val_loss), epoch)

		if epoch % train_config.save_each_n_epoch == 0:
			print("Saving model checkpoint...")
			filename = os.path.join(train_config.model_dir, f"{epoch}_{model_name}.safetensors")
			sam_lora.save_lora_parameters(filename)
			print("Success")

		if writer:
			writer.flush()

from modules.config import Config

if __name__ == "__main__":
	config = Config.from_yaml("./myConfig2.yaml", sub=["training", "model"])
	train(config.training, config.model)