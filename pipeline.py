from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)

import traceback


def run_research_pipeline(topic: str) -> dict:

    state = {}

    print("\n" + "=" * 70)
    print("RESEARCH PIPELINE STARTED")
    print("=" * 70)

    try:
        # STEP 1 : SEARCH
        print("\n[STEP 1] Search Agent")

        search_agent = build_search_agent()

        search_result = search_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Find recent, reliable and detailed information about {topic}"
                    )
                ]
            }
        )

        state["search_results"] = search_result["messages"][-1].content

        print("\nSearch completed successfully.\n")



        # STEP 2 : SCRAPE
        print("[STEP 2] Reader Agent")

        reader_agent = build_reader_agent()

        reader_result = reader_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"""
Use the search results below.

Identify the most relevant URL.

Scrape that webpage.

Search Results:

{state["search_results"]}
"""
                    )
                ]
            }
        )

        state["scraped_content"] = reader_result["messages"][-1].content

        print("\nScraping completed successfully.\n")



        # STEP 3 : WRITER
        print("[STEP 3] Writer")

        research = f"""
SEARCH RESULTS

{state["search_results"]}


SCRAPED CONTENT

{state["scraped_content"]}
"""

        state["report"] = writer_chain.invoke(
            {
                "topic": topic,
                "research": research,
            }
        )

        print("\nReport generated successfully.\n")



        # STEP 4 : CRITIC
        print("[STEP 4] Critic")

        state["feedback"] = critic_chain.invoke(
            {
                "report": state["report"]
            }
        )

        print("\nCritic completed successfully.\n")


        # SAVE REPORT
        filename = topic.replace(" ", "_") + "_report.txt"

        with open(filename, "w", encoding="utf-8") as f:

            f.write("=" * 80 + "\n")
            f.write("RESEARCH REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(state["report"])

            f.write("\n\n")

            f.write("=" * 80 + "\n")
            f.write("CRITIC REVIEW\n")
            f.write("=" * 80 + "\n\n")

            f.write(state["feedback"])

        print(f"Report saved as: {filename}")

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)

        return state

    except Exception as e:

        print("\nPipeline Failed!\n")

        traceback.print_exc()

        state["error"] = str(e)

        return state


if __name__ == "__main__":

    topic = input("Enter a research topic: ").strip()

    if not topic:

        print("Topic cannot be empty.")

    else:

        result = run_research_pipeline(topic)

        if "report" in result:

            print("\n" + "=" * 80)
            print("FINAL REPORT")
            print("=" * 80)
            print(result["report"])

            print("\n" + "=" * 80)
            print("CRITIC REVIEW")
            print("=" * 80)
            print(result["feedback"])