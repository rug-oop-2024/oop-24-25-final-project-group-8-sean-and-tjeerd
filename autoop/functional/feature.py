
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    data_frame = dataset.read()

    features = []

    max_categorical_values = 0.1 * len(data_frame.columns[0])

    for column_name in data_frame.columns:
        column_data = data_frame[column_name]
        unique_values = len(set(column_data))
    
        if column_data.dtype == 'object' or unique_values < max_categorical_values:
            feature_type = 'categorical'
        else:
            feature_type = 'numerical'

        feature = Feature(type=feature_type, name=column_name, data=column_data)
        features.append(feature)
    
    return features
