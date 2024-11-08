from pydantic import BaseModel


class MetricScoreModel(BaseModel):
    metric_name: str
    score: float
    segment_idx: int = 0
    patient_number: int = 0
