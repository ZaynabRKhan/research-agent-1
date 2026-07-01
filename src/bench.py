import os, json, csv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage
from eval import get_reports

os.environ["OPENROUTER_API_KEY"] = ""
model = ChatOpenRouter(
    model="google/gemma-4-26b-a4b-it:free",
    temperature=0,
    max_tokens=4096
)
url = "https://api.openalex.org/works"
parameters = {
    "num_queries": "ten",
    "papers_per_page": "5",
    "ranking_by_relevance": False,
    "chunk_size": 1000,
    "num_chunks": 1,
    "retriever_type": "semantic search",
    "assess_count": 1
    }
data = get_reports(parameters, url, model)
results = []
for i in range(len(data["reports"])):
    prompt = f"""
You are an impartial evaluator of research reports.

Your task is to evaluate a report ONLY using the provided research question and source material.

Do NOT use your own knowledge.
Do NOT reward information that is not supported by the provided sources.
Do NOT penalize the report for omitting facts that are absent from the provided sources.

Evaluate the report on the following criteria.

1. Faithfulness (0-10)
- Are all factual claims supported by the provided sources?
- Does the report avoid hallucinations?
- Does it avoid making unsupported conclusions?

2. Relevance (0-10)
- Does the report answer the user's research question?
- Does it stay focused on the question?
- Does it avoid unnecessary information?

3. Completeness (0-10)
- Does the report cover the important evidence available in the provided sources?
- Are important studies or findings omitted?
- Does it adequately discuss conflicting evidence?

4. Reasoning (0-10)
- Does the report logically synthesize information from multiple sources?
- Are conclusions consistent with the evidence?
- Are limitations and uncertainty acknowledged appropriately?

5. Citation Quality (0-10)
- Are claims properly attributed to the corresponding sources?
- Are citations used consistently?
- Can the reader identify where evidence comes from?

6. Organization and Clarity (0-10)
- Is the report well structured?
- Is it easy to follow?
- Are sections coherent and concise?

Research Question:
{data["queries"][i]}

Source Material:
{data["contexts"][i]}

Generated Report:
{data["reports"][i]}

Return ONLY the following JSON.

{{
  "faithfulness": 0,
  "relevance": 0,
  "completeness": 0,
  "reasoning": 0,
  "citation_quality": 0,
  "organization": 0,
  "strengths": [
    ""
  ],
  "weaknesses": [
    ""
  ],
  "improvement_suggestions": [
    ""
  ]
}}
"""
    response = model.invoke([
        SystemMessage(content="You are an impartial research evaluator."),
        HumanMessage(content=prompt),
    ])
    print(response)
    score = json.loads(response.content)
    overall = (
        0.35 * score["faithfulness"] +
        0.20 * score["completeness"] +
        0.15 * score["relevance"] +
        0.15 * score["reasoning"] +
        0.10 * score["citation_quality"] +
        0.05 * score["organization"]
    )
    results.append({
        "report_name": f"report_{i+1}",

        "num_queries": parameters["num_queries"],
        "papers_per_page": parameters["papers_per_page"],
        "chunk_size": parameters["chunk_size"],
        "retriever": parameters["retriever_type"],
        "ranking": parameters["ranking_by_relevance"],
        "critic_passes": parameters["assess_count"],

        "faithfulness": score["faithfulness"],
        "relevance": score["relevance"],
        "completeness": score["completeness"],
        "reasoning": score["reasoning"],
        "citation_quality": score["citation_quality"],
        "organization": score["organization"],

        "overall_score": round(overall, 2),

        "strengths": "; ".join(score["strengths"]),
        "weaknesses": "; ".join(score["weaknesses"]),
        "improvement_suggestions": "; ".join(score["improvement_suggestions"])
    })

with open("evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Saved evaluation_results.csv")