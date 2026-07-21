from dotenv import load_dotenv
import os
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

result = client.search(
    "Artificial Intelligence",
    max_results=2
)

print(result)