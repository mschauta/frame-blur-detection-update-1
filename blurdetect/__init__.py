"""Frame blur detection: inference code for the published weights (see METHOD.md)."""

from .model import BlurDetector, load_detector, model_input

__all__ = ["BlurDetector", "load_detector", "model_input"]
