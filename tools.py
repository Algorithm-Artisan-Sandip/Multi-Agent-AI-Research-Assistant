from langchain.tools import tool
from tavily import TavilyClient
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import os
import requests

# Load Environment Variables
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in .env file")

tavily = TavilyClient(api_key=TAVILY_API_KEY)


# Search Tool
# Tool name exposed to the LLM = web_search
# Python function name = web_search
@tool("web_search")
def web_search(query: str) -> str:
    """
    Search the web using Tavily and return detailed search results.
    """

    try:
        results = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
            include_raw_content=True
        )

        output = []

        if results.get("answer"):
            output.append(f"AI Summary:\n{results['answer']}\n")

        for i, result in enumerate(results.get("results", []), start=1):

            output.append(
                f"""
==============================
Result {i}

Title:
{result.get("title","N/A")}

URL:
{result.get("url","N/A")}

Content:
{result.get("content","No content available")}
"""
            )

        if len(output) == 0:
            return "No search results found."

        return "\n".join(output)

    except Exception as e:
        return f"Tavily Search Error: {str(e)}"


# URL Scraper
@tool
def scrape_url(url: str) -> str:
    """
    Scrape a webpage and return cleaned text.
    """

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/138.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        for tag in soup(
            [
                "script",
                "style",
                "header",
                "footer",
                "nav",
                "aside",
                "noscript"
            ]
        ):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        text = " ".join(text.split())

        if len(text) > 5000:
            text = text[:5000]

        return text

    except Exception as e:
        return f"Scraping Error: {str(e)}"