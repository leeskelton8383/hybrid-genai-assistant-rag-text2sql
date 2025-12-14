from agent.llm_router import route_with_llm
from agent.composer import compose_final_answer
from agent.actions import run_stats_tool, run_narrative_tool

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Apply FedEx corporate cert for all HTTPS libs
cert_path = os.getenv("REQUESTS_CA_BUNDLE")
if cert_path:
    os.environ["REQUESTS_CA_BUNDLE"] = cert_path
    os.environ["CURL_CA_BUNDLE"]     = cert_path
    os.environ["SSL_CERT_FILE"]      = cert_path

def main():
    #question = "How many rushing yards per game did the Eagles have in 2024?"
    #question = "In which game did the Eagles have the most rushing yards in 2024?"
    #question = "Was the 2020 season more disappointing than 2023?"
    #question = "Who was the leading rusher for the Eagles in 2022 and what narratives are available about their performance that season?"
    #question = "What were the key factors that influenced the Eagles' performance in the 2023 season compared to 2022?"
    #question = "How did the Eagles' offensive performance in 2023 compare to their performance in 2022, considering both statistics and narratives?"
    #question = "What were the main reasons for the Eagles' success in the 2024 season compared to previous years?"
    #question = "Who was the better QB for the Eagles, Carson Wentz or Jalen Hurts, and what do the narratives say about their playing styles?"

    #question = "How many rushing yards per game did the Eagles have in 2024?"problem with sql generation fixed in stats_texttosql_parse_and_query_03.ipynb
    #question = "Who has been the Eagles most challenging opponent in the last 5 years?"
    #question = "Since 2020, what is the win loss record of the Eagles vs Dallas Cowboys?"# issue with SQL, calls oppon column
    #question = "What were the Eagles total rushing yards and average rushing yards per game in the 2023 season?"
    #question = "In what game did AJ Brown have the highest receiving yards for the Eagles?"
    question = "Which Eagle had the highest receiving yards in a game and when?"


    # 🔁 Agent makes routing decision
    routing_decision = route_with_llm(question)
    route = routing_decision["route"]
    reason = routing_decision["reason"]
    
    print(f"\n🧠 User Question: {question}")
    print(f"\n🧠 Agent route: {route} — Reason: {reason}")

    # Tool execution
    stats_result = None
    narrative_context = None

    if route in ["stats", "both"]:
        stats_result = run_stats_tool(question)["result"]
        print("\n✅ Stats Tool Result:\n", stats_result)

    if route in ["narrative", "both"]:
        chunks = run_narrative_tool(question, top_k=5)
        narrative_context = "\n\n".join(f"[{c['doc_title']}]\n{c['text']}" for c in chunks)
        print("\n✅ Narrative Context:\n", narrative_context[:1000], "...\n")   
    


    # 🧠 Final answer composition
    final_answer = compose_final_answer(
        question=question,
        stats_result=stats_result,
        narrative_context=narrative_context
    )

    print("\n📝 Final Answer:\n")
    print(final_answer)

if __name__ == "__main__":
    main()







