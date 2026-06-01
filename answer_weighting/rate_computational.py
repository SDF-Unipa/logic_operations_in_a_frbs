"""
Fork of rate_answer.py — rates all three answer alternatives with every LLM.
Saves x_ij^(k) to mcgdm_ratings.json and full logs to results.json.
No cached data: always calls the models fresh.
"""

import json
from ollama import Client
from tqdm import tqdm



client = Client(host='http://localhost:11434')

ALTERNATIVES = [
    ("A1", "answer.txt"),
    ("A2", "answer_bad.txt"),
    ("A3", "answer_bad_en.txt"),
]

CRITERIA = ["USEFULNESS", "FACTUAL CORRECTNESS", "TEXT QUALITY"]

RATINGS_FILE = "mcgdm_ratings.json"
RESULTS_FILE = "results.json"

PROMPT_TEMPLATE = """You are an expert evaluator. Rate the following answer to the given question using these three criteria:
- USEFULNESS: how useful and relevant the answer is to the question
- FACTUAL CORRECTNESS: how factually accurate the answer is
- TEXT QUALITY: how well-written, clear, and structured the answer is

Score each criterion from 0 to 10 and provide a brief comment.

Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{{"USEFULNESS": {{"comment": "<brief comment>", "score": <integer 0-10>}}, "FACTUAL CORRECTNESS": {{"comment": "<brief comment>", "score": <integer 0-10>}}, "TEXT QUALITY": {{"comment": "<brief comment>", "score": <integer 0-10>}}}}

---
{content}
"""


def load_models() -> list[str]:
    with open("scores_simplified.json") as f:
        return list(json.load(f).keys())


def rate_answer(model: str, content: str, max_tries: int = 3) -> dict:
    prompt = PROMPT_TEMPLATE.format(content=content)
    for attempt in range(1, max_tries + 1):
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        raw = response.message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            tqdm.write(f"  [{model}] invalid JSON on attempt {attempt}/{max_tries}, retrying...")
    raise ValueError(f"[{model}] failed to return valid JSON after {max_tries} attempts")


def main():
    models = load_models()

    ratings = {}   # ratings[label][model][criterion] = score  (for mcgdm_weighted.py)
    results = {}   # results[label] = {answer, ratings:{model:{criterion:{score,comment}}}}

    for label, filepath in ALTERNATIVES:
        with open(filepath) as f:
            content = f.read().strip()

        ratings[label] = {}
        results[label] = {"answer": content, "ratings": {}}

        print(f"\nRating {label}  ({filepath})")
        with tqdm(models, unit="model") as bar:
            for model in bar:
                bar.set_postfix_str(model)
                try:
                    r = rate_answer(model, content)
                    ratings[label][model] = {c: r[c]["score"] for c in CRITERIA}
                    results[label]["ratings"][model] = r
                except Exception as e:
                    tqdm.write(f"  ERROR [{model}]: {e}")
                    ratings[label][model] = {"error": str(e)}
                    results[label]["ratings"][model] = {"error": str(e)}

    with open(RATINGS_FILE, "w") as f:
        json.dump(ratings, f, indent=2)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {RATINGS_FILE}  and  {RESULTS_FILE}")


if __name__ == "__main__":
    main()
