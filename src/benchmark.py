import json
import time
from langchain_core.messages import HumanMessage
from src.agent import npc_app
from src.evaluator import Judge
from src.database import get_vectorstore

def run_benchmark():
    print("🚀 Starting Aetheria-DS Full RAG Benchmark...")
    
    with open("benchmark_data.json", "r") as f:
        data = json.load(f)
    
    judge = Judge()
    vectorstore = get_vectorstore()
    
    results = []
    
    # Accumulators for averages
    metrics = {
        "latency": 0,
        "context_precision": 0, # Was the retrieval relevant?
        "answer_relevance": 0,  # Did we answer the question?
        "faithfulness": 0,      # Did we stick to the retrieved text? (Hallucination)
        "correctness": 0        # Did we match ground truth?
    }
    
    for i, entry in enumerate(data):
        player_input = entry["player_input"]
        expected_facts = entry["expected_lore_facts"]
        
        print(f"\n[{i+1}/{len(data)}] Input: {player_input}")
        
        # --- 1. Execution ---
        start_time = time.time()
        input_state = {
            "messages": [HumanMessage(content=player_input)],
            "relationship_score": 50
        }
        output = npc_app.invoke(input_state)
        ai_response = output["messages"][-1].content
        duration = time.time() - start_time
        
        # --- 2. Evaluation Setup ---
        # We must simulate the Agent's retrieval to judge Context Precision.
        # Note: Ideally, the Agent should return the context it used, but here we re-fetch.
        retrieved_docs = vectorstore.similarity_search(player_input, k=2) 
        context_text = "\n".join([d.page_content for d in retrieved_docs])
        
        # --- 3. Run Judges ---
        
        # Metric A: Context Precision (Retrieval Quality)
        eval_ctx = judge.evaluate_context_precision(player_input, context_text)
        
        # Metric B: Answer Relevance (Response Quality)
        eval_rel = judge.evaluate_answer_relevance(player_input, ai_response)
        
        # Metric C: Faithfulness (Hallucination Check)
        eval_faith = judge.evaluate_faithfulness(ai_response, context_text)
        
        # Metric D: Correctness (Ground Truth Check)
        eval_corr = judge.evaluate_correctness(ai_response, expected_facts)
        
        # --- 4. Logging ---
        result_entry = {
            "input": player_input,
            "response": ai_response,
            "metrics": {
                "latency": duration,
                "context_precision": eval_ctx["score"],
                "answer_relevance": eval_rel["score"],
                "faithfulness": eval_faith["score"],
                "correctness": eval_corr["score"]
            },
            "reasons": {
                "ctx": eval_ctx.get("reason"),
                "faith": eval_faith.get("reason"),
                "missing": eval_corr.get("missing_facts")
            }
        }
        results.append(result_entry)
        
        # Update totals
        metrics["context_precision"] += eval_ctx["score"]
        metrics["answer_relevance"] += eval_rel["score"]
        metrics["faithfulness"] += eval_faith["score"]
        metrics["correctness"] += eval_corr["score"]

        print(f"   > Faithfulness: {eval_faith['score']:.2f} | Precision: {eval_ctx['score']:.2f}")
        print(f"   > Correctness:  {eval_corr['score']:.2f} | Relevance: {eval_rel['score']:.2f}")

    # --- 5. Final Calculation ---
    n = len(data)
    summary = {
        "total_test_cases": n,
        "averages": {k: v/n for k, v in metrics.items()},
        "detailed_results": results
    }
    
    with open("benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\n" + "="*50)
    print("📊 BENCHMARK COMPLETE")
    print(f"Avg Faithfulness (No Hallucination): {summary['averages']['faithfulness']:.2f}")
    print(f"Avg Context Precision (Retrieval):   {summary['averages']['context_precision']:.2f}")
    print(f"Avg Answer Relevance (Coherence):    {summary['averages']['answer_relevance']:.2f}")
    print(f"Avg Correctness (Ground Truth):      {summary['averages']['correctness']:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()