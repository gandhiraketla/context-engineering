#!/usr/bin/env python3
"""
Streamlit UI for RAG Context Pruning Demo
=========================================

Clean UI interface that uses the RAG pipeline from prunecontext_rag.py

To run:
streamlit run streamlit_demo.py
"""

import streamlit as st
import sys
import os
from typing import Dict, List

# Import the RAG pipeline from the other file
try:
    from prunecontext_rag import RAGDemo
except ImportError:
    st.error("Could not import RAGDemo from prunecontext_rag.py. Make sure the file exists in the same directory.")
    st.stop()

def initialize_rag_demo():
    """Initialize the RAG demo with caching"""
    if 'rag_demo' not in st.session_state:
        with st.spinner("🔄 Initializing RAG pipeline..."):
            try:
                st.session_state.rag_demo = RAGDemo()
                st.session_state.rag_demo.setup_vectorstore()
                st.success("✅ RAG pipeline initialized successfully!")
            except Exception as e:
                st.error(f"❌ Error initializing RAG pipeline: {str(e)}")
                st.info("Make sure your .env file contains: OPENAI_API_KEY, PINECONE_API_KEY, DEEPSEEK_API_KEY, LANGSMITH_API_KEY")
                st.stop()
    
    return st.session_state.rag_demo

def get_comparison_results(rag_demo, query: str) -> Dict:
    """Get results from all three RAG approaches"""
    with st.spinner("🔍 Processing your question..."):
        # Get structured results from backend
        results = rag_demo.run_comparison(query, return_results=True)
        
    return results

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="RAG Context Pruning Demo",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 RAG Context Pruning Demo")
    st.markdown("Compare two approaches: **Plain RAG** and **LLM-based Context Pruning**")
    
    # Initialize RAG demo
    rag_demo = initialize_rag_demo()
    
    # Input section
    st.header("💬 Ask a Question")
    
    # Sample questions
    st.markdown("**Sample questions you can try:**")
    sample_questions = [
        "Can tuition fees be deducted from taxes?",
        "What medical expenses are tax deductible?",
        "How do I claim travel expenses for business?",
        "What are the education tax benefits available?",
        "Can I deduct charitable contributions?"
    ]
    
    for i, sample in enumerate(sample_questions):
        if st.button(f"📋 {sample}", key=f"sample_{i}"):
            st.session_state.user_question = sample
    
    # Text input
    user_question = st.text_input(
        "Enter your question:",
        value=st.session_state.get('user_question', ''),
        placeholder="e.g., Can tuition fees be deducted from taxes?"
    )
    
    # Submit button
    if st.button("🔍 Get Answer", type="primary") and user_question.strip():
        
        # Store question in session state
        st.session_state.user_question = user_question
        
        # Get results
        results = get_comparison_results(rag_demo, user_question)
        
        # Store results in session state
        st.session_state.results = results
        st.session_state.current_question = user_question
    
    # Display results if available
    if 'results' in st.session_state and 'current_question' in st.session_state:
        results = st.session_state.results
        question = st.session_state.current_question
        
        st.header(f"📋 Results for: *{question}*")
        
        # Context Comparison Section
        st.subheader("📄 Context Comparison")
        
        # Original Context
        with st.expander("📄 Original Retrieved Context", expanded=False):
            st.text_area(
                "Full original context:",
                value=results['contexts']['original'],
                height=200,
                disabled=False,
                key="original_context_display"
            )
            st.info(f"**Length:** {len(results['contexts']['original'])} characters")
        
        # LLM Pruned Context
        with st.expander("📝 LLM Pruned Context", expanded=False):
            st.text_area(
                "LLM pruned context:",
                value=results['contexts']['pruned'],
                height=200,
                disabled=False,
                key="pruned_context_display"
            )
            st.info(f"**Length:** {len(results['contexts']['pruned'])} characters")
            reduction_pruned = ((len(results['contexts']['original']) - len(results['contexts']['pruned'])) / len(results['contexts']['original'])) * 100
            st.info(f"**Reduction:** {reduction_pruned:.1f}%")
        
        # Answers Comparison Section
        st.subheader("💡 Answers Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 Plain RAG")
            st.write(results['answers']['plain'])
        
        with col2:
            st.markdown("### 📝 LLM Pruning")
            st.write(results['answers']['pruned'])
        
        # LLM Judge Evaluation Section
        st.subheader("⚖️ LLM Judge Evaluation")
        
        if results.get('evaluation') and results['evaluation'] and results['evaluation'].get('success'):
            # Display the full evaluation
            with st.expander("📋 Full Judge Evaluation", expanded=True):
                st.text_area(
                    "LLM Judge Analysis:",
                    value=results['evaluation']['evaluation'],
                    height=400,
                    disabled=False
                )
            
            # Display scores if available
        #     if results['evaluation'].get('scores'):
        #         st.subheader("📊 Evaluation Scores")
                
        #         scores_data = results['evaluation']['scores']
                
        #         # Create columns for score display
        #         col1, col2 = st.columns(2)
                
        #         with col1:
        #             st.markdown("#### 🤖 Plain RAG")
        #             if 'plain' in scores_data:
        #                 plain_scores = scores_data['plain']
        #                 st.metric("Relevance", f"{plain_scores.get('relevance', 0)}/10")
        #                 st.metric("Accuracy", f"{plain_scores.get('accuracy', 0)}/10") 
        #                 st.metric("Completeness", f"{plain_scores.get('completeness', 0)}/10")
        #                 st.metric("Clarity", f"{plain_scores.get('clarity', 0)}/10")
        #                 st.metric("**Total**", f"**{plain_scores.get('total', 0)}/40**")
                
        #         with col2:
        #             st.markdown("#### 📝 LLM Pruning")
        #             if 'pruned' in scores_data:
        #                 pruned_scores = scores_data['pruned']
        #                 st.metric("Relevance", f"{pruned_scores.get('relevance', 0)}/10")
        #                 st.metric("Accuracy", f"{pruned_scores.get('accuracy', 0)}/10")
        #                 st.metric("Completeness", f"{pruned_scores.get('completeness', 0)}/10") 
        #                 st.metric("Clarity", f"{pruned_scores.get('clarity', 0)}/10")
        #                 st.metric("**Total**", f"**{pruned_scores.get('total', 0)}/40**")
        # elif results.get('evaluation') and results['evaluation'] and not results['evaluation'].get('success'):
        #     st.error("⚠️ LLM Judge evaluation failed.")
        #     st.write(results['evaluation']['evaluation'])
        # else:
        #     st.warning("⚠️ LLM Judge evaluation was not computed for this query.")
        
        # Summary Statistics
        st.subheader("📊 Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Original Context", f"{len(results['contexts']['original'])} chars")
        
        with col2:
            st.metric(
                "LLM Pruned", 
                f"{len(results['contexts']['pruned'])} chars",
                f"-{reduction_pruned:.1f}%"
            )
        
        with col3:
            st.metric("Question Length", f"{len(question)} chars")

    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This demo compares two RAG approaches:
        
        **🤖 Plain RAG**
        - Uses full retrieved context
        - No pruning applied
        
        **📝 LLM-based Pruning**
        - Uses LLM to prune context with specific instructions
        - Two-step process: prune → answer
        
        **⚖️ LLM-as-a-Judge Evaluation**
        - Expert LLM evaluates both approaches
        - Scores: Relevance, Accuracy, Completeness, Clarity
        - Provides ranking and detailed analysis
        
        All interactions are tracked in **LangSmith** for detailed analysis.
        """)
        
        st.header("🔧 Setup")
        st.markdown("""
        Required environment variables:
        - `OPENAI_API_KEY`
        - `PINECONE_API_KEY`
        - `DEEPSEEK_API_KEY`
        - `LANGSMITH_API_KEY`
        """)

if __name__ == "__main__":
    main()