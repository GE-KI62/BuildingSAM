import os
import numpy as np
import rasterio as rio
from tqdm import tqdm
from pathlib import Path
from itertools import product
import glob
from modules.config import Config
from rasterio import windows

DTYPE = {
	rio.uint8: 2**8 - 1,
	rio.uint16: 2**16 - 1,
	rio.uint32: 2**32 - 1
}


def join_make(*args):
	path = os.path.join(*args)
	os.makedirs(path, exist_ok=True)
	return path

def easy_write(filename, meta, image, mask):
	with rio.open(filename, "w", **meta) as f:
		f.write(image)
		f.write_mask(mask)

#tiles the window of an image by using the specified width and height + overlap in both directions (x, y)
def get_tiles(meta, transform, width=256, height=256, step_size=256):
	ncols, nrows = meta['width'], meta['height']
	step_size = max(step_size, 1) 

	steps = product(range(0, ncols, step_size), range(0, nrows, step_size))
	#big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
	for col_off, row_off in steps:
		#window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
		window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height)
		win_transform = windows.transform(window, transform)
		yield window, win_transform



class Preprocessor:
	def __init__(self, buffer_size=0, step_size=1024, resample_size=0.3, keep_empty=False, image_format=".tif"):
		self.buffer_size = buffer_size
		self.dest_width = 1024
		self.dest_height = 1024 
		self.step_size = step_size
		self.resample_size = resample_size
		self.keep_empty = keep_empty
		self.image_format = "." + image_format.lstrip(".").lower()

	def _buffer_image(self, file):
		meta = file.meta.copy()

		x_res, y_res = file.res

		out_shape = (file.count, *file.shape)

		if round(x_res, 2) != self.resample_size or round(y_res, 2) != self.resample_size:
			out_shape = (file.count, int(file.height * y_res / self.resample_size), int(file.width * x_res / self.resample_size))

		data = file.read(out_shape=out_shape)
		premask = file.dataset_mask(out_shape=out_shape[1:])

		width = data.shape[-1]
		height = data.shape[-2]

		scale_x = file.width / data.shape[-1] 
		scale_y = file.height / data.shape[-2] 

		new_transform = file.transform * file.transform.scale(scale_x, scale_y)
		#meta.update(transform=new_transform)

		width, height = int(width + self.buffer_size * 2), int(height + self.buffer_size * 2)
		img = np.zeros((file.count, height, width))

		if self.buffer_size > 0:
			buf_slice_w = slice(self.buffer_size, -self.buffer_size)
			buf_slice_h = slice(self.buffer_size, -self.buffer_size)
		else:
			buf_slice_w = slice(0, width)
			buf_slice_h = slice(0, height)

		img[:, buf_slice_h, buf_slice_w] = data
		
		if "dtype" in meta and meta["dtype"] in DTYPE:
			img = img * 255 / DTYPE[meta["dtype"]]

		mask = np.zeros((height, width), dtype=np.uint8)
		mask[buf_slice_h, buf_slice_w] = premask

		new_transform = rio.Affine(
			new_transform.a,
			new_transform.b,
			new_transform.c - (self.buffer_size * new_transform.a),
			new_transform.d,
			new_transform.e,
			new_transform.f - (self.buffer_size * new_transform.e)
		)

		meta.update({
			"width": width,
			"height": height,
			"transform": new_transform
		})

		return img, mask, meta, Path(file.name).stem


	#crops a list of images to the specified width and height. Additionally resampling and channel limiting can be done.
	def crop(self, buffered, dir_name):
		image, mask, meta, img_name = buffered
		for window, win_transform in tqdm(list(get_tiles(meta, meta["transform"], self.dest_width, self.dest_height, self.step_size))):	
			tile_meta = meta.copy()
			tile_meta.update({
				"width" : window.width,
				"height" : window.height,
				"transform": win_transform,
				"dtype" : rio.uint8,
				"compress": "LZW"
			})

			tile = np.zeros((image.shape[0], window.height, window.width), dtype=np.uint8)
			w = image[:, window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width]
			tile[:, :w.shape[1], : w.shape[2]] = w

			mask_tile = np.zeros((window.height, window.width), dtype=np.uint8)
			m = mask[window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width]
			mask_tile[:m.shape[0], :m.shape[1]] = m

			if not self.keep_empty and np.all(mask_tile == 0): continue  

			image_path = join_make(self.out_path, "crop", dir_name)
			image_filename = os.path.join(image_path, f"{img_name}_{window.row_off}_{window.col_off}.{self.image_format}")
			
			easy_write(image_filename, tile_meta, tile, mask_tile)
				

	def _buffer(self, img_path) :
		buffered = None
		with rio.open(img_path, "r+") as file:
			buffered = self._buffer_image(file)
		return buffered 


	def __call__(self, img_paths, out_path, dir_name): 
		self.out_path = out_path
		os.makedirs(self.out_path, exist_ok=True)

		for path in img_paths:
			print(f"Processing image '{Path(path).name}'")

			buffered = self._buffer(path)
			image, mask, meta, _ = buffered

			if self.buffer_size > 0:
				buffer_path = join_make(self.out_path, "buffer", dir_name)
				filename = os.path.join(buffer_path, Path(path).name)
				easy_write(filename, meta, image, mask)
			
			self.crop(buffered, dir_name)


def process(config):
	search_folder = os.path.join(config.image_dir, "*" + config.image_format)
	img_paths = glob.glob(search_folder)
	print(f"Found {len(img_paths)} image(s) for preprocessing...")

	processor = Preprocessor(config.buffer_size, config.step_size, config.resample_size, config.keep_empty, config.image_format)
	processor(img_paths, config.output_dir, "images")

	if config.target_dir:	
		target_dir = os.path.join(config.target_dir, "*" + config.image_format)
		masks_paths = glob.glob(target_dir)
		print(f"Found {len(masks_paths)} target(s) for preprocessing...")
		processor(masks_paths, config.output_dir, "targets")
	


if __name__ == "__main__":
	config = Config.from_yaml("./config2.yaml")
	process(config.preprocessor)	
	