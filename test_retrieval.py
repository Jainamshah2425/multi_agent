from qa_engine import QAEngine
import json

if __name__ == "__main__":
    qa_engine = QAEngine()
    query = "What is the current weather in London?"
    
    print(f"Testing answer_query with web search DISABLED for query: \"{query}\"")
    
    results_no_web = qa_engine.answer_query(query, use_web_search=False)
    
    print("\n--- Results (Web Search Disabled) ---")
    print(json.dumps(results_no_web, indent=2))
    
    if not results_no_web.get('web_context'):
        print("\n✅ Web context is empty as expected when web search is disabled.")
    else:
        print("\n❌ Web context is NOT empty, web search was not disabled.")

    print(f"\nTesting answer_query with web search ENABLED for query: \"{query}\"")
    
    results_with_web = qa_engine.answer_query(query, use_web_search=True)
    
    print("\n--- Results (Web Search Enabled) ---")
    print(json.dumps(results_with_web, indent=2))
    
    if results_with_web.get('web_context'):
        print("\n✅ Web context is NOT empty as expected when web search is enabled.")
    else:
        print("\n❌ Web context is empty, web search was not enabled.")