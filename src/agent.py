from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage

from .database import get_vectorstore
from .langchain_mlstudio import AetheriaLLM


vectorstore = get_vectorstore() # vector for RAG
llm_wrapper = AetheriaLLM()
llm = llm_wrapper.llm

class NPCState(TypedDict):
    messages: Annotated[list, add_messages]
    relationship_score: int

def npc_logic(state: NPCState):
    user_msg = state['messages'][-1].content
    
    relevant_lore = vectorstore.similarity_search(user_msg, k=2)
    lore_context = "\n".join([d.page_content for d in relevant_lore])
    
    # Roleplay instructions
    prompt_content = f"""
    You are Elara, a mysterious wood-elf. 
    Current Relationship with Player: {state['relationship_score']}/100
    Lore Context: {lore_context}
    Stay in character. Do not mention you are an AI.
    """
    
    system_msg = SystemMessage(content=prompt_content)
    
    # prepending the roleplay instructions at the start of the chat history
    response = llm.invoke([system_msg] + state['messages'])
    return {"messages": [response]}

def analyze_sentiment(state: NPCState):
    user_msg = state['messages'][-1].content
    
    sentiment_prompt = f"""
    Analyze the sentiment of this player's message: "{user_msg}"
    Is it: 
    - Friendly/Respectful? (+5)
    - Neutral? (0)
    - Hostile/Insulting? (-10)
    
    Return ONLY the number.
    """
    
    response = llm.invoke(sentiment_prompt)
    
    try:
        score_change = int(response.content.strip()) # type: ignore #
    except:
        score_change = 0
        
    new_score = max(0, min(100, state['relationship_score'] + score_change))
    
    return {"relationship_score": new_score}

# Graph
builder = StateGraph(NPCState)

builder.add_node("sentiment_node", analyze_sentiment)
builder.add_node("elara_brain", npc_logic)

builder.add_edge(START, "sentiment_node")
builder.add_edge("sentiment_node", "elara_brain")
builder.add_edge("elara_brain", END)

npc_app = builder.compile()