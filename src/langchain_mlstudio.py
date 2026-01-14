import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

class AetheriaLLM:
    def __init__(self, temperature=0.7, local_model_name="llama-3.2-3b-instruct", local_base_url="http://localhost:1234/v1"):
        """
        Initializes the connection to the LLM.
        - Checks for GOOGLE_API_KEY to use Gemini (Cloud/Production).
        - Defaults to LM Studio (Localhost/Development) if no key is found.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if api_key:
            print("[CLOUD] Detected GOOGLE_API_KEY. Using Google Gemini (Cloud Mode).")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=temperature,
                google_api_key=api_key
            )
        else:
            print("[LOCAL] No API Key found. Using LM Studio (Local Mode).")
            self.llm = ChatOpenAI(
                base_url=local_base_url,
                api_key="lm-studio",
                model_name=local_model_name, # type: ignore
                temperature=temperature
            )

    def chat(self, prompt, system_prompt="You are a helpful assistant."):
        """
        Sends a message to the model and returns the response.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content

if __name__ == "__main__":
    # Example usage
    model = AetheriaLLM()
    
    print("Connecting to LLM...")
    try:
        user_input = "Tell me a short fun fact about space."
        response = model.chat(user_input)
        print(f"\nUser: {user_input}")
        print(f"AI: {response}")
    except Exception as e:
        print(f"\nError connecting to LLM: {e}")
        print("If using Local Mode: Make sure LM Studio is running on port 1234.")
        print("If using Cloud Mode: Check your GOOGLE_API_KEY.")
