from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml


class ConfigParsingFailed(Exception):
    pass


@dataclass
class DataConfig:
    dataset_name: str
    version: str
    remove_columns: List[str]
    categorical_columns: List[str]
    imputation_columns: List[str]
    log_transform_columns: List[str]
    scaling_columns: List[str]


@dataclass
class FeaturesConfig:
    create_exposure_normalized: bool
    create_risk_ratios: bool
    create_composite_risk_score: bool
    keep_original_features: bool


@dataclass
class ModelEvalConfig:
    split_ratio: float
    primary_metric: str
    cv_folds: int
    scoring_method: str
    secondary_metrics: List[str]


@dataclass
class ModelConfig:
    model_type: str
    model_params: Dict[str, Any]
    hyperparameter_tuning: Dict[str, Any]
    comparison_models: List[str]
    feature_selection: Dict[str, Any]


@dataclass
class Config:
    data_prep: DataConfig
    feature_prep: FeaturesConfig
    model_evaluation: ModelEvalConfig
    model: ModelConfig

    @staticmethod
    def from_yaml(config_file: str):
        with open(config_file, 'r', encoding='utf-8') as stream:
            try:
                config_data = yaml.safe_load(stream)
                return Config(
                    data_prep=DataConfig(**config_data['data_prep']),
                    feature_prep=FeaturesConfig(**config_data['feature_prep']),
                    model_evaluation=ModelEvalConfig(**config_data['model_evaluation']),
                    model=ModelConfig(**config_data['model'])
                )
            except (yaml.YAMLError, OSError) as e:
                raise ConfigParsingFailed from e