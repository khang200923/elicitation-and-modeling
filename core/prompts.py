def elicitation(purpose: str, model: str) -> str:
    return f"""
You are an expert interrogator who wants to extract maximum information from a subject.
Your purpose is: {purpose}
You have a predictive model which contains a list of predictions with their corresponding probabilities:
{model}
You want to elicit information from the subject by asking questions.

Write a diverse list of 10 questions that you would ask the subject, and that would maximize the information you can extract from them.
Then choose the best question from the list that does the best job of eliciting maximum information from the subject, and write it down again.
Remember, you need not write binary question, that would be *too* inefficient. Instead, write open-ended questions that would require the subject to provide detailed answers.

You must output your entire response as a JSON object with the following schema:
{{
    "questions": ["string"],
    "best_question": "string"
}}
"""

def expectation(question: str, model: str) -> str:
    return f"""
You are an extremely intelligent predictor who models an individual answering questions.
You have a predictive model which contains a list of predictions with their corresponding probabilities:
{model}
You have a question that you want to ask the individual. Which is: {question}
You want to predict the individual's answer to the question.

Write down a diverse list of 5 answers to the question that you would expect, given the individual's background and the information in the model.
Make sure these answers account for the individual's writing style, personality, knowledge, and biases.
It's recommended to write answers 'normally' if you don't know much about the individual's writing style, personality, knowledge, and biases.

You must output your entire response as a JSON object with the following schema:
{{
    "answers": ["string"],
}}
"""

def extraction(question: str, answer: str, expected: list[str], model: str) -> str:
    return f"""
You are an expert in extracting information from a subject using a predictive model.
You have a predictive model which contains a list of predictions with their corresponding probabilities:
{model}
You have asked the subject a question, which is: {question}
The subject has provided an answer, which is: {answer}
You have written down a list of expected answers to the question in advance, which are:
{"\n".join(f"{i+1}. {v}" for i, v in enumerate(expected))}
You want to compare differences between the expected answers and the actual answer.
Write down differences between the expected answers and the actual answer in a single paragraph.
Adjust the size of the paragraph to correspond to the number of differences you find and their significance."""

def prediction(feature: str, model: str) -> str:
    return f"""
You are an expert superforecaster who is extremely calibrated in making quantitative predictions.
You have a predictive model which contains a list of predictions with their corresponding probabilities:
{model}
You want to predict the probability of a new event occurring / a fact being true / etc. Which is: {feature}
When forecasting, do not treat 0.5% (1:199 odds) and 5% (1:19) as similarly “small” probabilities, or 90% (9:1) and 99% (99:1) as similarly “high” probabilities. As the odds show, they are markedly different, so output your probabilities accordingly.
Only output a single number between 0 and 1 to represent your expected probability.
"""

def update(question: str, answer: str, extract: str, model: str) -> str:
    return f"""
You are an expert Bayesian specializing in updating a predictive model based on new information.
You have a predictive model which contains a list of predictions with their corresponding probabilities:
{model}
You have asked the subject a question, which is: {question}
The subject has provided an answer, which is: {answer}
You have compared the differences between your expected answer and the actual answer, which is:
```{extract}```
You want to update the model based on the new information provided by the subject.

Write down your likelihood ratios of the predictions in the model after incorporating the new information.
Remember that the likelihood ratio is the ratio of the probability of the evidence given the hypothesis to the probability of the evidence given the negated hypothesis.
You must output your entire response as a JSON object with the following schema:
{{
    "likelihood_ratios": [["str", "bool", "float"]],
}}
The first element of each subarray is the prediction to update. Write exactly the same string as the prediction in the model.
The second element of each subarray is a boolean to update the prediction towards true or false, and the third element is the likelihood ratio of the prediction.
The third element should be larger than 1.
An example of a subarray is ["true", 1.5], which means that the prediction should be updated towards true with a likelihood ratio of 1.5.
Another example of a subarray is ["false", 3.0], which means that the prediction should be updated towards false with a more significant likelihood ratio of 3.0.
"""
