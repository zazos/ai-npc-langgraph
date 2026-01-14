import streamlit as st

# Fix for SQLite version on Streamlit Cloud (must be before importing chromadb)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import uuid
from langchain_core.messages import HumanMessage, AIMessage
from src.agent import npc_app
from src.evaluator import Judge
from src.database import get_vectorstore

st.set_page_config(page_title="Aetheria: Elara")

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "relationship_score" not in st.session_state:
    st.session_state.relationship_score = 50
if "previous_relationship_score" not in st.session_state:
    st.session_state.previous_relationship_score = 50
if "last_audit" not in st.session_state:
    st.session_state.last_audit = None

metrics_placeholder = st.empty()
with metrics_placeholder.container():
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.last_audit:
            score = st.session_state.last_audit['hallucination_score']
            st.metric("Hallucination Risk", f"{score*100:.1f}%", delta=f"-{score*100:.1f}%", delta_color="inverse")
        else:
            st.metric("Hallucination Risk", "N/A")
    with col2:
        rel_score = st.session_state.relationship_score
        rel_delta = rel_score - st.session_state.previous_relationship_score
        st.metric("Relationship Score", f"{rel_score}/100", delta=rel_delta)

st.title("Elara - The Wood Elf")
st.markdown("*A mysterious encounter in the forests of Aetheria...*")

judge = Judge()
vectorstore = get_vectorstore()

for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    if role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message.content)
    else:
        with st.chat_message(role):
            st.markdown(message.content)

# input from user
if prompt := st.chat_input("Say something to Elara..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # adds user message to session state (history)
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.spinner("Elara is thinking..."):
        # Run the agent logic
        # LLM "remembers" the full conversation from the input state
        input_state = {
            "messages": st.session_state.messages,
            "relationship_score": st.session_state.relationship_score
        }
        output = npc_app.invoke(input_state) # type: ignore
        ai_response_msg = output["messages"][-1]
        
        # update relationship score based on sentiment analysis from the agent
        if "relationship_score" in output:
            st.session_state.previous_relationship_score = st.session_state.relationship_score
            st.session_state.relationship_score = output["relationship_score"]
        
        # Save score to message metadata for history display
        ai_response_msg.additional_kwargs["relationship_score"] = st.session_state.relationship_score
        
        # lore as context for evaluation to the Judge
        # search using the AI response to find lore relevant to the claims made
        relevant_lore = vectorstore.similarity_search(ai_response_msg.content, k=3)
        context = "\n".join([d.page_content for d in relevant_lore])
        
        # Lore Audit by the Judge
        audit = judge.evaluate_hallucinations(ai_response_msg.content, context)
        st.session_state.last_audit = audit
        
        # In a real LangGraph with checkpointer, we wouldn't need to manually append response to session state
        st.session_state.messages.append(ai_response_msg)

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(ai_response_msg.content)

        if audit["hallucination_score"] > 0.4:
            st.error(f"**[WARNING] High Hallucination Risk!** Score: {audit['hallucination_score']}")
        
        with st.expander("View Hallucination Audit Trace"):
            st.json(audit)
            
    # Update metrics immediately
    with metrics_placeholder.container():
        col1, col2 = st.columns(2)
        with col1:
            score = audit['hallucination_score']
            st.metric("Hallucination Risk", f"{score*100:.1f}%", delta=f"-{score*100:.1f}%", delta_color="inverse")
        with col2:
            rel_score = st.session_state.relationship_score
            rel_delta = rel_score - st.session_state.previous_relationship_score
            st.metric("Relationship Score", f"{rel_score}/100", delta=rel_delta)
    # st.rerun() # Removed to prevent clearing the chat response/badge immediately
