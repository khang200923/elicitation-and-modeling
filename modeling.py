from dataclasses import dataclass
import json
from typing import Union
from pydantic import BaseModel
from utils import parse, create, systemp, userp, assistantp
import prompts

@dataclass
class Model:
    """
    A class used to represent a Model with predictions.
    """
    predictions: dict[str, float]

    def __str__(self) -> str:
        """
        Returns a string representation of the model's predictions.
        """
        return "\n".join(f"{i+1}. {p[0]} ({p[1]*100}% chance)" for i, p in enumerate(self.predictions.items()))

    def add(self, feature: str, probability: float) -> None:
        """
        Adds a new feature to the model.
        """
        self.predictions[feature] = probability

    def remove(self, feature: str) -> None:
        """
        Removes a feature from the model.
        """
        self.predictions.pop(feature)

    def update(self, feature: str, likelihood_ratio: float) -> float:
        """
        Updates the probability of an existing feature using a likelihood ratio.
        """
        prior = self.predictions[feature]
        self.predictions[feature] = prior * likelihood_ratio / (prior * likelihood_ratio + 1 - prior)
        return self.predictions[feature]

    def predict(self, feature: str) -> float:
        """
        Returns the probability of a feature, whatever it is, using information from the model.
        """
        prompt = prompts.prediction(feature, str(self))
        response = create(
            model="gpt-4o-mini-2024-07-18",
            messages=[systemp(prompt)],
            max_tokens=10
        ).choices[0].message.content
        return float(response)

    def use(self, question: str, answer: str, extraction: str) -> list[float]:
        """
        Uses a question and answer to update the model.
        """
        prompt = prompts.update(question, answer, extraction, str(self))
        class Schema(BaseModel):
            likelihood_ratios: list[list[Union[str, bool, float]]]
        response = parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[systemp(prompt)],
            max_tokens=1000,
            response_format=Schema
        ).choices[0].message.content
        for feature, hypothesis, likelihood_ratio in json.loads(response)["likelihood_ratios"]:
            real_ratio = likelihood_ratio if hypothesis else 1 / likelihood_ratio
            self.update(feature, real_ratio)
        return json.loads(response)["likelihood_ratios"]
