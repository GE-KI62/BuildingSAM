import numpy as np
import rasterio as rio
import os
from tqdm import tqdm
from pathlib import Path
from rasterio.merge import merge as rio_merge
import glob
from rasterio.windows import Window
from modules.config import Config

class AverageMerger:
	def __init__(self, width, height):
		self.sum = np.zeros((height, width), dtype=np.float64)
		self.count = np.zeros((height, width), dtype=np.uint32)
	
	def merge(self, merged_data, new_data, merged_mask, new_mask, coff, roff, **kwargs):
		valid_pixels = ~new_mask
		if np.any(valid_pixels):
			rows, cols = np.where(valid_pixels.squeeze())
			rmin, rmax = roff + rows.min(), roff + rows.max() + 1
			cmin, cmax = coff + cols.min(), coff + cols.max() + 1

			self.sum[rmin:rmax, cmin:cmax] += new_data.squeeze()[rows.min():rows.max() + 1, cols.min():cols.max() + 1] 
			self.count[rmin:rmax, cmin:cmax] += 1

class Postprocessor:
	def __init__(self, input_dir: str, image_extension: str = ".tif"):
		self.input_dir = Path(input_dir)
		self.image_extension = "." + image_extension.lstrip().lower()

	def create_alpha_sources(self, img_paths):
		mem_files = []
		alpha_sources = []
		print("Creating alpha masks...")
		for path in tqdm(img_paths):
			with rio.open(path) as src:
				alpha_profile  = src.profile.copy()
				alpha_profile.update(count=1)

				memfile =  rio.MemoryFile()
				ds = memfile.open(**alpha_profile)
				mask = src.dataset_mask()
				ds.write(np.expand_dims(mask, 0))
				alpha_sources.append(ds)
				mem_files.append(memfile)
				#memfile.close()
		return alpha_sources, mem_files

	def clip(self, src):
		mask = src.dataset_mask()
		rows, cols = np.where(mask > 0)
		window = Window.from_slices((rows.min(), rows.max() + 1), (cols.min(), cols.max() + 1))
		clipped = src.read(window=window)
		clipped_mask = src.dataset_mask(window=window)
		transform = src.window_transform(window)
		meta = src.meta.copy()
		meta.update({
			"transform": transform,
			"width": window.width,
			"height": window.height
		})

		return clipped, clipped_mask, meta


	def __call__(self, output_path: str = "./", output_type="masks", threshold=127):
		img_paths = glob.glob(os.path.join(self.input_dir, f"*{self.image_extension}"))
		unique_stems = set(["_".join(Path(x).name.split("_")[:-2]) for x in img_paths])
		os.makedirs(output_path, exist_ok=True)

		with rio.open(img_paths[0], "r") as f:
			profile = f.profile.copy()

		for unique in tqdm(unique_stems):
			img_paths = glob.glob(os.path.join(self.input_dir, f"{unique}*{self.image_extension}"))
			print(f"Merging {unique}...({len(img_paths)} images)")

			alpha_sources, mem_files = self.create_alpha_sources(img_paths)
			mask, transform = rio_merge(alpha_sources, method="max")
			print("Finished merging masks...")

			height, width = mask.shape[-2:]
			merger = AverageMerger(width=width, height=height)

			[a.close() for a in alpha_sources]
			[m.close() for m in mem_files]

			mosaic, transform = rio_merge(img_paths, method=merger.merge)
			print("Finished merging images...")

			with np.errstate(divide="ignore", invalid="ignore"):
				avg = np.where(
					merger.count > 0,
					merger.sum / merger.count,
					np.nan
				)

			profile.update({
					"width": mosaic.shape[2],
					"height": mosaic.shape[1],
					"transform": transform,
					"count": 1,
					"compression": "LZW",
					"dtype": rio.uint8
				})

			out_filename = os.path.join(output_path, f"{unique}{self.image_extension}")
			memfile = rio.MemoryFile()
			dst = memfile.open(**profile)
			avg = np.nan_to_num(avg, nan=0, posinf=255, neginf=0)
			dst.write(np.expand_dims(avg, 0))
			dst.write_mask(mask.squeeze())

			print(f"Clipping {unique} by mask...")

			clipped, clipped_mask, meta = self.clip(dst)
			clipped[clipped < threshold] = 0

			if output_type == "masks":
				clipped[clipped != 0] = 255

			with rio.open(out_filename, "w", **meta) as f:
				f.write(clipped)
				f.write_mask(clipped_mask)

			dst.close()
			memfile.close()

def postprocess(config):
	postprocessor = Postprocessor(config.image_dir, config.image_extension)
	postprocessor(config.output_dir, config.output_type, config.threshold)


if __name__ == "__main__":
	config = Config.from_yaml("config2.yaml")
	postprocess(config.postprocessor)