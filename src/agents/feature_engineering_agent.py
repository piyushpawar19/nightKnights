import logging
from typing import Dict, Any
from pydantic import ValidationError

from nightKnights.src.interfaces.feature_interface import FeatureEngineeringInterface
from nightKnights.src.preprocessing.feature_engineering import FeatureEngineering
from nightKnights.src.schemas.feature_schema import FeatureEngineeringRequest, CandidateFeatures

logger = logging.getLogger(__name__)

class FeatureEngineeringAgent(FeatureEngineeringInterface):
    def __init__(self):
        self.feature_engineering = FeatureEngineering()

    def validate_inputs(self, state: Dict):
        try:
            # Validate the incoming state against the request schema
            FeatureEngineeringRequest(parsed_jd=state.get("parsed_jd", {}),
                                      extracted_skills=state.get("extracted_skills", {}),
                                      candidate_profile=state.get("candidate_profile", {}))
            logger.info("FeatureEngineeringAgent: Inputs validated successfully.")
        except ValidationError as e:
            logger.error(f"FeatureEngineeringAgent: Input validation failed: {e.errors()}")
            raise ValueError(f"Invalid input for feature engineering: {e.errors()}")

    def generate_raw_metrics(self, state: Dict) -> Dict:
        logger.info("FeatureEngineeringAgent: Generating raw metrics.")
        return self.feature_engineering.generate_raw_metrics(state)

    def generate_normalized_metrics(self, raw_metrics: Dict, state: Dict) -> Dict:
        logger.info("FeatureEngineeringAgent: Generating normalized metrics.")
        return self.feature_engineering.generate_normalized_metrics(raw_metrics, state)

    def generate_metadata(self, raw_metrics: Dict) -> Dict:
        logger.info("FeatureEngineeringAgent: Generating metadata.")
        return self.feature_engineering.generate_metadata(raw_metrics)

    def construct_candidate_features(self, raw_metrics: Dict, normalized_metrics: Dict, metadata: Dict) -> CandidateFeatures:
        logger.info("FeatureEngineeringAgent: Constructing CandidateFeatures object.")
        return self.feature_engineering.construct_candidate_features(raw_metrics, normalized_metrics, metadata)

    def validate_schema(self, candidate_features: CandidateFeatures):
        try:
            candidate_features.model_dump_json()  # Attempt to serialize to JSON to trigger validation
            logger.info("FeatureEngineeringAgent: Output schema validated successfully.")
        except ValidationError as e:
            logger.error(f"FeatureEngineeringAgent: Output schema validation failed: {e.errors()}")
            raise ValueError(f"Invalid output candidate features schema: {e.errors()}")

    def run(self, state: Dict) -> Dict:
        logger.info("FeatureEngineeringAgent: Running feature engineering process.")
        self.validate_inputs(state)

        raw_metrics = self.generate_raw_metrics(state)
        normalized_metrics = self.generate_normalized_metrics(raw_metrics, state)
        metadata = self.generate_metadata(raw_metrics)
        
        candidate_features = self.construct_candidate_features(raw_metrics, normalized_metrics, metadata)
        self.validate_schema(candidate_features)

        state["candidate_features"] = candidate_features.model_dump()
        logger.info("FeatureEngineeringAgent: Feature engineering completed and state updated.")
        return state
