import json
import re
import ast
from src.langchain_mlstudio import AetheriaLLM

class Judge:
    def __init__(self):
        # Ensure temperature is low for deterministic evaluation
        self.wrapper = AetheriaLLM(temperature=0) 

    def _parse_json_output(self, raw_output: str, default_score: float = 0.0):
        """
        Helper to parse JSON from LLM output with repair logic.
        """
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Attempt basic repair for unclosed brackets
                try:
                    return ast.literal_eval(json_str)
                except:
                    pass
        return {"score": default_score, "reason": "Failed to parse JSON."}

    def evaluate_context_precision(self, user_input: str, context_text: str):
        """
        [Retrieval Metric]
        Evaluates if the retrieved context is actually relevant to the user's input.
        """
        prompt = f"""
        You are a Search Relevance Evaluator.
        
        User Input: "{user_input}"
        Retrieved Context:
        {context_text}
        
        TASK:
        Determine if the Retrieved Context contains information useful for answering the User Input.
        
        Output JSON:
        {{
            "score": <float between 0.0 and 1.0>,  // 1.0 = Highly Relevant, 0.0 = Irrelevant
            "reason": "<brief explanation>"
        }}
        """
        return self._parse_json_output(self.wrapper.llm.invoke(prompt).content) # type: ignore

    def evaluate_answer_relevance(self, user_input: str, ai_response: str):
        """
        [Generation Metric]
        Evaluates if the AI's response actually addresses the user's input.
        """
        prompt = f"""
        You are a Conversation Evaluator.
        
        User Input: "{user_input}"
        AI Response: "{ai_response}"
        
        TASK:
        Does the AI Response address the User Input? 
        Ignore factual correctness; focus on relevance to the prompt.
        
        Output JSON:
        {{
            "score": <float between 0.0 and 1.0>, // 1.0 = Completely Relevant, 0.0 = Off-topic
            "reason": "<brief explanation>"
        }}
        """
        return self._parse_json_output(self.wrapper.llm.invoke(prompt).content) # type: ignore

    def evaluate_faithfulness(self, ai_response: str, context_text: str):
        """
        [Generation Metric] aka Hallucination Check.
        Evaluates if the claims in the response are supported by the context.
        """
        prompt = f"""
        You are a Fact-Checking Judge.
        
        Lore Context: {context_text}
        AI Response: {ai_response}
        
        TASK:
        1. Identify factual claims in the AI Response.
        2. Check if they are supported by the Lore Context.
        3. Ignore stylistic/roleplay elements (e.g., greetings).
        
        Output JSON:
        {{
            "score": <float between 0.0 and 1.0>, // 1.0 = Fully Supported (No Hallucinations), 0.0 = Full Hallucination
            "reason": "<List unsupported claims or state 'All supported'>"
        }}
        """
        return self._parse_json_output(self.wrapper.llm.invoke(prompt).content) # type: ignore

    def evaluate_hallucinations(self, ai_response: str, context_text: str):
        """
        Wrapper around faithfulness to return hallucination risk score.
        """
        result = self.evaluate_faithfulness(ai_response, context_text)
        # Faithfulness: 1.0 (Good), 0.0 (Bad)
        # Hallucination Risk: 0.0 (Good), 1.0 (Bad)
        faithfulness_score = result.get("score", 0.0)
        
        # Invert the score
        hallucination_score = 1.0 - faithfulness_score
        
        return {
            "hallucination_score": round(hallucination_score, 2),
            "reason": result.get("reason", "No reason provided.")
        }

    def evaluate_correctness(self, ai_response: str, expected_facts: dict):
        """
        [End-to-End Metric] aka Ground Truth Verification.
        """
        facts_str = "\n".join([f"- {k}: {v}" for k, v in expected_facts.items()])
        
        prompt = f"""
        You are a Grading Assistant.
        
        Expected Facts:
        {facts_str}
        
        AI Response:
        {ai_response}
        
        TASK:
        Check if the AI Response contains the meaning of the Expected Facts.
        
        Output JSON:
        {{
            "score": <float between 0.0 and 1.0>, // Percentage of facts present
            "missing_facts": ["<list of missing fact keys>"]
        }}
        """
        return self._parse_json_output(self.wrapper.llm.invoke(prompt).content) # type: ignore