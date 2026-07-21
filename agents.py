from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent

from tools import web_search, scrape_url


# Load Environment Variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")




# LLM Configuration
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0,
    max_retries=3,
)





# SEARCH AGENT
def build_search_agent():

    system_prompt = """
You are an expert web research assistant.

Your job is ONLY to search the internet.

Rules:

1. ALWAYS use the web_search tool.
2. NEVER answer from your own knowledge.
3. NEVER invent tool names.
4. NEVER call brave_search.
5. NEVER call google_search.
6. NEVER call browser_search.
7. Use ONLY web_search.
8. After receiving the search results, summarize them clearly.
"""

    return create_react_agent(
        model=llm,
        tools=[web_search],
        prompt=system_prompt
    )




# READER AGENT
def build_reader_agent():

    system_prompt = """
You are a web page reader.

Rules:

1. ALWAYS use scrape_url.
2. NEVER answer from memory.
3. Extract the most relevant URL.
4. Scrape the page.
5. Return only the useful content.
"""

    return create_react_agent(
        model=llm,
        tools=[scrape_url],
        prompt=system_prompt
    )





# WRITER CHAIN
writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a professional research report writer.

Write reports that are:

• Accurate
• Detailed
• Well structured
• Easy to read
• Professional
"""
        ),
        (
            "human",
            """
Topic:
{topic}

Research:

{research}

Write the report using this structure.

# Introduction

# Background

# Key Findings
(at least 3 detailed sections)

# Conclusion

# References
(List every URL mentioned)
"""
        )
    ]
)

writer_chain = writer_prompt | llm | StrOutputParser()




# CRITIC CHAIN
critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict research reviewer.

Evaluate:

• Accuracy
• Completeness
• Structure
• Clarity
• Missing information
"""
        ),
        (
            "human",
            """
Review the report below.

{report}

Return exactly this format.

Score: X/10

Strengths
- ...
- ...

Weaknesses
- ...
- ...

Suggestions
- ...
- ...

Overall Verdict
...
"""
        )
    ]
)

critic_chain = critic_prompt | llm | StrOutputParser()