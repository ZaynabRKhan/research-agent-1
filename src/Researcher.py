import os, time, requests, json, pymupdf, re
from typing import Optional
from typing_extensions import TypedDict
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from pydantic import BaseModel, Field
from openrouter_embeddings import OpenRouterEmbeddings
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from openrouter.errors import TooManyRequestsResponseError

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["OPENROUTER_API_KEY"] = "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"
OPEN_ALEX_API_KEY = "bhxxpyMQdDYpnQkDjHn4cc"
# llm = HuggingFaceEndpoint(
#     repo_id = "meta-llama/Llama-3.1-8B-Instruct",
#     temperature = 0,
#     max_new_tokens = 2048
# )
# model = ChatHuggingFace(llm=llm)
model = ChatOpenRouter(
    model="google/gemma-4-26b-a4b-it:free",
    temperature=0,
    max_tokens=4096
)

# STATE
def merge_lists(existing: list, new: list) -> list:
    return existing + new

class ResearcherState(TypedDict):
    url: str
    params: dict
    query: str
    queries: list
    pdf_links: list
    extracted_data: str
    context: str
    parameters: dict
    vector_database: Optional[Chroma]
    model: ChatOpenRouter
    assessment: dict
    error_log: str
    status: str

class QueryList(BaseModel):
    """A list of queries"""
    queries: list = Field(description="A list of queries")

class ReportAssessment(BaseModel):
    """An assessment of the report"""
    decision: str = Field(description="APPROVE or REVISE")
    score: int = Field(description="An overall score of the report")
    assessment: str = Field(description="A general assessment of the report based on the user question.")
    unsupported_claims: list = Field(description="A list of unsupported claims made in the report, if any otherwise empty")
    contradictions: list = Field(description="A list of contradictory claims made in the report, if any otherwise empty")
    revision_instructions: list = Field(description="Instructions to make the revised report if decision is REVISE")

def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return None
    words = [None] * (max(max(pos) for pos in inverted_index.values()) + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words)

# NODES
def get_links(state: ResearcherState) -> dict:
    prompt = f"""
Given a user's question, generate {state["parameters"]["num_queries"]} search queries that are likely to retrieve relevant academic papers from research databases such as arXiv, Semantic Scholar, or Google Scholar.

The search queries should:
- Focus on the core concepts behind the user's question.
- Use terminology commonly found in academic literature.
- Avoid detours of the topic.
- Include related technical terms, synonyms, or subtopics when appropriate.
- Be concise and suitable for direct use as search queries.

Return only a JSON ARRAY of {state["parameters"]["num_queries"]} search queries and nothing else.

User Query:
{state["query"]}
"""
    try:
        structured_model = model.with_structured_output(QueryList, method="json_schema")
        response = structured_model.invoke([
            SystemMessage(content="You are a research query generation agent."),
            HumanMessage(content=prompt),
        ])
        print(response)
        # query_list = json.loads(response)
        query_list = response.queries
        # print(type(query_list))
        print("LLM response:\n",query_list)
    except Exception as error:
        # print("Trying to get queries again.")
        raise RuntimeError(error)
    try:
        paper_links = []
        # query_list_formatted = [q.replace(' ','+') for q in query_list]
        for search_string in query_list:
            print("quering string:", search_string)
            params = {
                "search.semantic": f"{search_string}",
                "api_key": OPEN_ALEX_API_KEY,
                "per_page": state["parameters"]["papers_per_page"]
            }
            try:
                paper = requests.get(state["url"], params=params, timeout=90)
                paper = json.loads(paper.text)
            except Exception as e:
                print("Type:", type(e))
                print(f"Error: {e}")
                continue
            paper_linksq = [
                {
                    "title": x["title"],
                    "doi": x["doi"],
                    "url": x["primary_location"]["pdf_url"],
                    "abstract": reconstruct_abstract(x["abstract_inverted_index"])
                } for x in paper["results"] #if x["primary_location"]["pdf_url"] is not None
            ]
            paper_links.extend(paper_linksq)

        print("Total papers extracted:", len(paper_links))
        if len(paper_links) == 0:
            print("API returned 0 pdfs.")
            return {
                "status": "get_links_error",
                "error_log": "AP returned 0 pdfs"
            }
        return {
            "model": model,
            "queries": query_list,
            "pdf_links": paper_links,
            "status": "get_links_done",
        }
    except requests.RequestException as error:
        print(error)
        raise RuntimeError(f"Fetch failed for {url}: {str(error)}")
    except Exception as error:
        raise RuntimeError(error)
        # return {
        #     "status": "get_links_error",
        #     "error_log": [f"Fetch failed for {url}: {str(error)}"],
        # }
    
def parse_links(state: ResearcherState) -> dict:
    pdf_links = state["pdf_links"]
    print("count of data", len(pdf_links))
    global pdfs
    pdfs = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*"
    }

    for link in pdf_links:
        if link["url"] is not None:
            try:
                res = requests.get(link["url"], headers=headers, allow_redirects=True, timeout=30)
                content_type = res.headers.get("Content-Type", "")
                if content_type== "application/pdf":
                    print("valid pdf")
                    pdfs.append({"title": link["title"], "doi": link["doi"], "content": res.content})
                elif link["abstract"] is not None:
                    print("used abstract")
                    pdfs.append({"title": link["title"], "doi": link["doi"], "content": link["abstract"]})
                else:
                    print(f"{link["title"]} is not a valid pdf.")
            except requests.exceptions.ReadTimeout as e:
                print(f"Timeout downloading {link["title"]}")
                continue
            except Exception as e:
                print(f"Error downloading {link["title"]}: {e}")
                continue
    # with ThreadPoolExecutor(max_workers=5) as exe:
    #     exe.map(get_concurrent_requests, pdf_links)
    # print("got concurrent pdfs")
    if len(pdfs) == 0:
        raise RuntimeError("There are zero PDFs that can be parsed.")
        # return {
        #     "error_log": "There are zero PDFs that can be parsed.",
        #     "status": "parse_links_error"
        # }
    return {
        "extracted_data": pdfs,
        "status": "parse_links_done"
    }

def parse_pdfs(state: ResearcherState) -> dict:
    valid_count = 0
    abstract_count = 0
    all_chunks = []
    abs = ''
    pdfs = state.get("extracted_data")
    for pdf in pdfs: 
        try:
            if type(pdf["content"]) == str:
                abstract_count += 1
                text = pdf["content"]
                abs += text
            else:
                valid_count += 1
                doc = pymupdf.Document(stream=pdf["content"])
                text = ''
                for page in doc:
                    text += page.get_text()
            lc_doc = Document(
                page_content=text,
                metadata={
                    "title": pdf["title"],
                    "doi": pdf["doi"]
                }
                )
            chunk_size = state["parameters"]["chunk_size"]
            chunk_overlap = round(chunk_size * 0.2, 2)
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(documents=[lc_doc])
            all_chunks.extend(chunks)
        except Exception as e:
            # print(f"ERROR: Could not parse pdf bytes for {pdf["title"]}:", e)
            raise RuntimeError(f"Could not parse pdf bytes for {pdf["title"]}: {str(e)}")
        #     return {
        #     "status": "parse_pdfs_error",
        #     "error_log": [f"Could not parse pdf bytes for {pdf["title"]}: {str(e)}"],
        # }
    print("Streamed all pdfs")
    print("valid pdf count:", valid_count, "Abstract pdf count:", abstract_count)
    print("chunked all documents")
    print("total chunks:", len(all_chunks))
    try:
        # embeddings = HuggingFaceEndpointEmbeddings(model="BAAI/bge-base-en-v1.5")
        embeddings = OpenRouterEmbeddings(os.environ.get("OPENROUTER_API_KEY"), "nvidia/llama-nemotron-embed-vl-1b-v2:free")
        # embeddings = embed_model.embed_documents()
        vector_db = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            collection_name="ReportCreator"
        )
        print("Vector DB created with name: ReportCreator")
        return {
            "vector_database": vector_db, 
            "status": "parse_pdfs_done"
            }
    except Exception as e:
        # print("Couldn't create the vector DB:", e)
        raise RuntimeError(f"Couldn't create the vector DB: {str(e)}")
        # return {
        #     "status": "parse_pdfs_error",
        #     "error_log": [f"Couldn't create the vector DB: {str(e)}"],
        # }

def query_model(state: ResearcherState) -> dict:
    vector_db = state["vector_database"]
    model = state["model"]
    query = state["query"]
    queries = state["queries"]
    queries.append(query)
    # print("got the state data")
    docs = []
    for q in queries:
         doc = vector_db.similarity_search(q, k=state["parameters"]["num_chunks"])
        #  ret = vector_db.as_retriever(search_kwargs = {"k":3})
        #  doc = ret.invoke(q)
         docs.extend(doc)
    context_parts = []
    for i, doc in enumerate(docs):
        title = doc.metadata.get("title", f"Source {i+1}")
        context_parts.append(
            f"""
SOURCE TITLE: {title},

CONTENT:
{doc.page_content}
"""
        )
        context = "\n\n".join(context_parts)
    prompt = f"""
You are an expert research analyst.

Your task is to generate a research report using ONLY the information provided in the source material based ONLY on the user query.

IMPORTANT RULES:

1. Use only the information contained in the provided sources.
2. Do NOT use prior knowledge.
3. Do NOT make assumptions.
4. Do NOT infer facts that are not explicitly supported by the sources.
5. Do NOT fabricate citations, statistics, methodologies, or conclusions.
6. If the evidence is incomplete, clearly state the limitations.
7. If the sources disagree, explicitly describe the disagreement.
8. If there is insufficient evidence to answer part of the question, say so.
9. Never claim certainty beyond what the sources support.

USER QUESTION:
{query}

SOURCE MATERIAL:
{context}

Generate a report with the following sections:

# Executive Summary

Provide a concise summary of the main findings.

# Mention the report title which is just the user question.

# Full Report

Answer the user's question elaborately using the given context.

# Sources

List all the sources used to answer this question.

For each source include:

* Source title
* Main findings
* Methodology (if available)
* Limitations (if available)

# Conclusion

Provide a conclusion strictly supported by the evidence.

REPORT LENGTH REQUIREMENTS:

* If only a small amount of evidence is available, produce a concise report.
* If substantial evidence is available, produce a detailed report.
* Let the quantity and quality of evidence determine the report length naturally.
* Do not add filler text to increase report size.

CITATION REQUIREMENTS:

For every factual claim, cite the supporting source title in parentheses.

Example:
"Several studies reported improved visual outcomes (Source A, Source B)."

Output only the final report.

"""
    print(len(prompt))
    try:
        response = model.invoke([
            SystemMessage(content="You are an expert research analyst."),
            HumanMessage(content=prompt),
        ])
        print("Got LLM response")
        print(type(response.content))
        return {
            "extracted_data": response.content,
            "context": context,
            "status": "query_model_done"
        }
    except TooManyRequestsResponseError as e:
        print(e)
        print(repr(e))
        raise
    except Exception as e:
        print(repr(e))
        if hasattr(e, "response_data"):
            print(e.response_data)
        raise

def assess_report(state: ResearcherState) -> dict:
    report = state["extracted_data"]
    context = state["context"]
    model = state["model"]
    query = state["query"]
    prompt = f"""
You are a research critic.

Evaluate the generated report using ONLY the provided source material.

Do not use outside knowledge.

Inputs:

USER QUESTION:
{query}

SOURCE MATERIAL:
{context}

GENERATED REPORT:
{report}

Evaluate the report on the following criteria:

1. **Relevance**

* Does the report answer the user's question?
* If not, explain why.

2. **Completeness**

* Has the report omitted any important evidence or major findings from the sources?


3. **Reasoning**

* Does the report avoid conclusions that go beyond the available evidence? 
Only report a contradiction if two or more sources make opposing claims about the SAME research question or outcome.
Do not consider two studies discussing different diseases, populations, or endpoints to be contradictory.

Finally, return a JSON object in the following format:
{{
  "decision": "APPROVE" | "REVISE",
  "score": 0-10,
  "assessment": "",
  "unsupported_claims": [],
  "contradictions": [],
  "revision_instructions": []
}}

Return only the JSON object. 

Do NOT explain your reasoning.

Do NOT think step-by-step.

Do NOT revise your answer while writing.

Never include intermediate thoughts.

Never include any text outside the JSON.

If a field is empty, return an empty list.
"""
    try:
        structured_model = model.with_structured_output(ReportAssessment, method="json_schema")
        response = structured_model.invoke([
            SystemMessage(content="You are a research critic."),
            HumanMessage(content=prompt)
        ])
        print("got critic reponse")
        # print(response)
        # print(type(response))

        # match = re.search(r"\{.*\}", response.content, re.DOTALL)
        # if match:
        #     print(match.group())
        #     # print(type(match.group()))
        #     response = json.loads(match.group())
        return {
            "assessment": response,
            "status": "assess_report_done"
        }
    except Exception as e:
        raise RuntimeError(f"Couldn't get the report assessment: {e}")
        # return {
        #     "status": "assess_report_error",
        #     "error_log": f"Couldn't get the report assessment: {e}"
        # }

def revise_report(state: ResearcherState) -> dict:
    print('revising the report')
    report = state["extracted_data"]
    assessment = state["assessment"]
    context = state["context"]
    query = state["query"]
    prompt = f"""
You are a research report revision agent.

Your task is to revise an existing research report using feedback from a research critic.

You are given:

USER QUESTION:
{query}

SOURCE MATERIAL:
{context}

ORIGINAL REPORT:
{report}

CRITIC ASSESSMENT:
{assessment}

Your objective is to improve the report while remaining completely faithful to the provided source material.

Requirements:

1. Use ONLY the provided source material.
2. Do NOT use outside knowledge.
3. Remove or correct unsupported claims identified by the critic.
4. Incorporate important missing evidence when it is supported by the source material.
5. Resolve contradictions or overstatements identified by the critic.
6. Ensure the report directly answers the user's question.
7. Clearly distinguish direct evidence from indirect evidence.
8. Preserve factual accuracy over writing style.
9. If the available evidence is insufficient to answer the question, explicitly state this.
10. Do not invent citations, statistics, or conclusions.

The revised report should contain:

* Executive Summary
* Research Question
* Evidence from Sources
* Cross-Source Analysis
* Limitations of Available Evidence
* Conclusion

Produce a revised report that addresses the critic's feedback while remaining fully supported by the supplied source material.

Output only the revised report.
"""
    try:
        response = model.invoke([
            SystemMessage(content="You are a research report revision agent."),
            HumanMessage(content=prompt)
        ])
        print("Got editor response")
        return {
            "extracted_data": response.content,
            "status": "revise_report_done"
        }
    except Exception as e:
        raise RuntimeError(f"Could not get editor response: {e}")
        # return {
        #     "error_log": f"Could not get editor response: {e}",
        #     "status": "revise_report_error" 
        # }

def route_after_eval(state: ResearcherState):
    if state["assessment"].decision == "REVISE":
        return "edited"
    return END

graph = StateGraph(ResearcherState)
graph.add_node("get", get_links)
graph.add_node("parse", parse_links)
graph.add_node("pdfs", parse_pdfs)
graph.add_node("report", query_model)
graph.add_node("eval", assess_report)
graph.add_node("edited", revise_report)
graph.add_edge(START, "get")
graph.add_edge("get", "parse")
graph.add_edge("parse", "pdfs")
graph.add_edge("pdfs", "report")
graph.add_edge("report", "eval")
graph.add_conditional_edges("eval", route_after_eval, ) #{END:END, "edited":"edited"}
graph.add_edge("edited", END)

scraper_agent = graph.compile()

if __name__ == "__main__":
    start = time.time()
    query = "can coffee cause blindness?"
    parameters = {
        "num_queries": "ten",
        "papers_per_page": "5",
        "ranking_by_relevance": False,
        "chunk_size": 1000,
        "num_chunks": 2,
        "retriever_type": "semantic search",
        "assess_count": 1
    }
    # url = f'http://export.arxiv.org/api/query'
    url = "https://api.openalex.org/works"
    result = scraper_agent.invoke({
        "url": url,
        "params": {},
        "query": query,
        "pdf_links": [],
        "parameters": parameters,
        "extracted_data": [],
        "vector_database": None,
        "model": model,
        "assessment": {},
        "error_log": "",
        "status": "ready"
    })
    print("*"*40)
    print("Final output of agent: received.")
    with open("/report2.txt", "w") as f:
        f.write(result["extracted_data"])
    print("Assessment:", result["assessment"])
    print("Time taken:", time.time()-start)