import json
from pydantic import BaseModel
import core.prompts as prompts
import core.modeling as modeling
from core.utils import parse, create, systemp

def elicitation(purpose: str, model: modeling.Model) -> str:
    prompt = prompts.elicitation(purpose, str(model))
    class Schema(BaseModel):
        questions: list[str]
        best_question: str
    response = parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[systemp(prompt)],
        max_tokens=500,
        response_format=Schema
    ).choices[0].message.content
    return json.loads(response)["best_question"]

def expectation(question: str, model: modeling.Model) -> list[str]:
    prompt = prompts.expectation(question, str(model))
    class Schema(BaseModel):
        answers: list[str]
    response = parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[systemp(prompt)],
        max_tokens=1000,
        response_format=Schema
    ).choices[0].message.content
    return json.loads(response)["answers"]

def extraction(question: str, answer: str, expected: list[str], model: modeling.Model) -> str:
    prompt = prompts.extraction(question, answer, expected, str(model))
    response = create(
        model="gpt-4o-mini-2024-07-18",
        messages=[systemp(prompt)],
        max_tokens=1000
    ).choices[0].message.content
    return response
