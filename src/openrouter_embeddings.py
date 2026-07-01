from langchain_core.embeddings import Embeddings
import requests, json
# os.environ["OPENROUTER_API_KEY"] = "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"
class OpenRouterEmbeddings(Embeddings):
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
    def embed_documents(self, texts):
        response = requests.post(
            url="https://openrouter.ai/api/v1/embeddings",
              headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
              data=json.dumps({
                "model": self.model,
                "input": texts
                })
        )
        return [item["embedding"] for item in response.json()["data"]]
    def embed_query(self, text):
        return self.embed_documents([text])[0]
    
# embedmodel = OpenRouterEmbeddings(os.environ.get('OPENROUTER_API_KEY', 'default_key'), "nvidia/llama-nemotron-embed-vl-1b-v2:free")
# print(embedmodel.embed_documents(["hello", "hola"]))