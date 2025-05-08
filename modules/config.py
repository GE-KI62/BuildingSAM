from pydantic import BaseModel, ValidationError, model_validator, ConfigDict, field_validator, Field, DirectoryPath, FilePath, confloat, validator, computed_field
from enum import Enum
from typing import Optional, Literal, Union, Annotated
import yaml
from pathlib import Path


class OutputType(str, Enum):
    PROBS = "probs"
    MASKS = "masks"

class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"

class TestTimeAugmentationType(str, Enum):
    HORIZONTAL_FLIP = "HorizontalFlip"
    VERTICAL_FLIP = "VerticalFlip"
    RANDOM_ROTATE_90 = "RandomRotate90"


class TrainTimeAugmentationType(BaseModel):
    name: Literal["HorizontalFlip", "VerticalFlip", "RandomRotate90", "Transpose", "RandomBrightnessContrast"]
    p: confloat(ge=0.0, le=1.0)

def validate_tupel(cls, v):
    if len(v) != 2:
        raise ValueError("Must contain exactly 2 values")
    if v[0] > v[1]:
        raise ValueError(f"First value has to be smaller than last value: {v[0]} > {v[1]}")
    return v

class RandomSizedCrop(TrainTimeAugmentationType):
    name: Literal["RandomSizedCrop"]
    min_max_height: list[int]
    size: list[int]

    _validate_min_max_height = validator("min_max_height", allow_reuse=True)(validate_tupel)
    _validate_size = validator("size", allow_reuse=True)(validate_tupel)

class CoarseDropout(TrainTimeAugmentationType):
    name: Literal["CoarseDropout"]
    num_holes_range: list[int]
    hole_height_range: list[int]
    hole_width_range: list[int]

    _validate_num_holes_range = validator("num_holes_range", allow_reuse=True)(validate_tupel)
    _validate_hole_height_range = validator("hole_height_range", allow_reuse=True)(validate_tupel)
    _validate_hole_width_range = validator("hole_width_range", allow_reuse=True)(validate_tupel)


TrainAugmentation = Annotated[Union[
    TrainTimeAugmentationType,
    RandomSizedCrop,
    CoarseDropout
], Field(discriminator="name")]

class ProtoConfig(BaseModel):
    image_extension: Optional[str] = ".tif"

    @field_validator("image_extension")
    def validate_image_extension(cls, v: str) -> str:
        return "." + v.lstrip(".") if v else ".tif"

class BaseConfig(ProtoConfig):
    image_dir: DirectoryPath = Field(..., path_type="dir", description="Input directory containing images")
    output_dir: Path = Field(..., path_type="dir", description="Output directory for processed files")

    @field_validator("output_dir")
    def validate_paths(cls, v: str) -> Path:
        try:
            v = Path(v)
            if not v.is_file():
                return v
            else:
                raise ValueError(f"Invalid directory path: {v}")
        except (TypeError, OSError) as e:
            raise ValueError(f"Invalid argument: {v}") from e

class PreprocessorConfig(BaseConfig):
    target_dir: Optional[DirectoryPath] = Field(default=None, path_type="dir", description="Optional directory (validated if provided)")
    buffer_size: Optional[int] = 0
    step_size: Optional[int] = 1024
    resample_size: Optional[float] = 0.3
    keep_empty: Optional[bool] = False

    @field_validator("target_dir")
    def validate_target_dir(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v

    @field_validator("buffer_size")
    def validate_buffer_size(cls, v: int) -> int:
        if v < 0:
            raise ValueError("buffer_size must be >= 0")
        return v

    @field_validator("step_size")
    def validate_step_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("step_size must be > 0")
        return v

    @field_validator("resample_size")
    def validate_resample_size(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("resample_size must be > 0")
        return v


class PostprocessorConfig(BaseConfig):
    threshold: Optional[int] = 0
    output_type: Optional[OutputType] = OutputType.MASKS

    @field_validator("threshold")
    def validate_threshold(cls, v: int) -> int:
        if not 0 <= v <= 255:
            raise ValueError("threshold must be in range 0 to 255")
        return v
    

class NetworkConfig(BaseModel):
    load_pth_from: FilePath = Field(..., path_type="file", description="Path to .pth model weights")
    rank: Optional[int] = 512
    load_checkpoint_from: Optional[FilePath] = Field(default=None, path_type="file", description="Path to .safetensors LoRA-adapter weights")
    device: Optional[DeviceType] = DeviceType.CPU


    @field_validator("rank")
    def validate_rank(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("rank must be > 0")
        return v


class VisualizationConfig(BaseModel):
    save_each_n_epoch: Optional[int] = 1
    output_dir: Path = None
    output_type: Optional[OutputType] = OutputType.MASKS

    @field_validator("save_each_n_epoch")
    def validate_save_each_n_epoch(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("save_each_n_epoch must be > 0")
        return v
    
    @field_validator("output_dir")
    def validate_path(cls, v: str) -> Path:
        try:
            v = Path(v)
            if not v.is_file():
                return v
            else:
                raise ValueError(f"Invalid directory path: {v}")
        except (TypeError, OSError) as e:
            raise ValueError(f"Invalid argument: {v}") from e

class DatasetConfig(ProtoConfig):
    image_dir: DirectoryPath
    target_dir: DirectoryPath

class TrainConfig(ProtoConfig):
    batch_size: Optional[int] = 1
    num_epochs: Optional[int] = 100
    learning_rate: Optional[float] = 1e-4
    save_each_n_epoch: Optional[int] = 1
    model_dir: Path 
    tensorboard_log_dir: Optional[Path] = None
    augmentations: Optional[list[TrainAugmentation]] = None
    train_dataset: Optional[DatasetConfig] = Field(None, alias="trainDataset")
    val_dataset: Optional[DatasetConfig] = Field(None, alias="valDataset")
    visualization: Optional[VisualizationConfig] = None

    @field_validator("batch_size", "num_epochs", "save_each_n_epoch")
    def validate_positive_ints(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Value must be > 0")
        return v

    @field_validator("learning_rate")
    def validate_learning_rate(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("learning_rate must be > 0")
        return v
    
    @field_validator("model_dir", "tensorboard_log_dir")
    def validate_paths(cls, v: str) -> Path:
        v = Path(v)
        if not v.is_file():
            return v
        else:
            raise ValueError(f"Invalid directory path: {v}")


class InferenceConfig(BaseConfig):
    output_type: Optional[OutputType] = OutputType.MASKS
    augmentations: Optional[list[TestTimeAugmentationType]] = None
    threshold: Optional[int] = 0

    @field_validator("threshold")
    def validate_threshold(cls, v: int) -> int:
        if not 0 <= v <= 255:
            raise ValueError("threshold must be in range 0 to 255")
        return v

class Config(ProtoConfig):
    preprocessor: Optional[PreprocessorConfig] = None
    model: Optional[NetworkConfig] = None
    training: Optional[TrainConfig] = None
    inference: Optional[InferenceConfig] = None
    postprocessor: Optional[PostprocessorConfig] = None

    @model_validator(mode="after")
    def link_parent(self):
        def update_child(child: BaseConfig):
            child.image_extension = self.image_extension

        if self.preprocessor:
            update_child(self.preprocessor)
        if self.training:
            update_child(self.training)
        if self.inference:
            update_child(self.inference)
        if self.postprocessor:
            update_child(self.postprocessor)
        return self

    @classmethod 
    def from_yaml(cls, config_path: str, sub=None) -> "Config":
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if sub is not None:
                    if not isinstance(sub, list): sub = [sub]
                    return cls(**dict((k, v) for k, v in config_data.items() if (not isinstance(v, dict) or k in sub)))
                return cls(**config_data)
        except FileNotFoundError:
            raise ValueError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")

if __name__ == "__main__":
    config = Config.from_yaml("./myConfig.yaml")
    print(config)