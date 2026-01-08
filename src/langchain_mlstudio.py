from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class LMStudioWrapper:
    def __init__(self, model_name="llama-3.2-3b-instruct", base_url="http://localhost:1234/v1"):
        """
        Initializes the connection to LM Studio's local server.
        Default port for LM Studio is 1234.
        """
        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key="lm-studio",  # LM Studio doesn't usually require a real key
            model_name=model_name, # type: ignore
            temperature=0.7
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
    model = LMStudioWrapper()
    
    print("Connecting to LM Studio...")
    try:
        user_input = "Tell me a short fun fact about space."
        response = model.chat(user_input)
        print(f"\nUser: {user_input}")
        print(f"AI: {response}")
    except Exception as e:
        print(f"\nError connecting to LM Studio: {e}")
        print("Make sure LM Studio is running and the Local Server is started on port 1234.")
