from typing import Dict
from nightKnights.src.schemas.feature_schema import CandidateFeatures

class FeatureEngineeringInterface:
    def run(self, state: Dict) -> Dict:
        raise NotImplementedError

    def validate_inputs(self, state: Dict):
        raise NotImplementedError

    def generate_raw_metrics(self, state: Dict) -> Dict:
        raise NotImplementedError

    def generate_normalized_metrics(self, raw_metrics: Dict, state: Dict) -> Dict:
        raise NotImplementedError

    def validate_schema(self, candidate_features: CandidateFeatures):
        raise NotImplementedError

    def construct_candidate_features(self, raw_metrics: Dict, normalized_metrics: Dict, metadata: Dict) -> CandidateFeatures:
        raise NotImplementedError
