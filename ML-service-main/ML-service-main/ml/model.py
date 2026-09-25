from dataclasses import dataclass
from pathlib import Path

import yaml
from transformers import pipeline

# load config file
# = Path(__file__).parent / "config.yaml"
#with open(config_path, "r") as file:
    #config = yaml.load(file, Loader=yaml.FullLoader)


@dataclass
class SentimentPrediction:
    """Класс, представляющий результат предсказания настроения."""

    label: str
    score: float


def load_model():
    """Загрузка предварительно обученной модели"""

    classifier = pipeline("text-classification",model="skandavivek2/spam-classifier",device=-1)

    def predict(text: str) -> SentimentPrediction:
        pred = classifier(text)
        pred_best_class = pred[0]

        return SentimentPrediction(
            label=pred_best_class["label"],
            score=pred_best_class["score"],
        )

    return predict