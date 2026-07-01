import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openrouter import ChatOpenRouter
from Researcher import scraper_agent
benchmark = [
    # "Does coffee cause blindness?",
    "Is marine life more diverse than terrestial life?",
    "Can LLMs replace coders?",
    "What is the significance of the discovery of CRISPR?",
    "What causes schizophrenia?",
    "Why do whales sing?",
    "Does intermittent fasting increase lifespan?",
    # "How do bees use the Earth's magnetic field?",
    # "What is the cumulative effect of global warming up till now?",
    # "What are the side effects of insulin?",
    # "How does social media affect humans?",
    # "Why do birds migrate?",
    # "How does fast food affect our mental health?",
    # "When can humans land on Mars?",
    # "What is the next step in tech after LLM agents?",
    # "Can AI detect genetic diseases from DNA?",
    # "What is the evidence of black holes?",
    # "How do we know the dinosaurs were covered in feathers?",
    # "What are the implications of bulding AI datacentres on the environment?",
    # "How many people did Alexander the Great kill?",
    # "Which country has the best education system in the world?",
]
print(len(benchmark))

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
OPEN_ALEX_API_KEY = "bhxxpyMQdDYpnQkDjHn4cc"

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

def get_reports(params=parameters, url=url, model=model):
    reports = []
    contexts = []
    n=0
    for query in benchmark:
        result = scraper_agent.invoke({
            "url": url,
            "params": {},
            "query": query,
            "parameters": params,
            "context": "",
            "pdf_links": [],
            "extracted_data": [],
            "vector_database": None,
            "model": model,
            "assessment": {},
            "error_log": "",
            "status": "ready"
        })
        n+=1
        name = f"../data/report-{query.replace("?","")}.txt"
        print(f"Completed report for '{query}'")
        print("*"*40)
        with open(name, "w", encoding="utf-8") as f:
            f.write(result["extracted_data"])
        reports.append(result["extracted_data"])
        contexts.append(result["context"])
    return {
        "reports": reports,
        "queries": benchmark,
        "contexts": contexts
    }