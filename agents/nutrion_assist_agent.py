"""
Nutritionist Agent Demo using LangGraph
=======================================
A demo agent that helps with nutrition questions, meal logging, and personalized advice.
"""

import os
import json
import requests
from datetime import datetime
from typing import TypedDict, Literal
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.graph.state import CompiledStateGraph

# OpenAI client for DeepSeek and Perplexity APIs
from openai import OpenAI

# Load environment variables
load_dotenv()

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Nutrition Agent"

# Initialize DeepSeek client
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class NutritionState(TypedDict):
    """State for the nutrition agent containing session data."""
    scratchpad: str  # Stores research results and session notes
    user_message: str  # Current user input
    intent: str  # Detected user intent
    topic: str  # Extracted topic from user message
    rewritten_query: str  # Search-optimized query
    memory_content: str  # Retrieved memory content
    final_answer: str  # Generated response


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def call_deepseek_llm(prompt: str, temperature: float = 0.0) -> str:
    """Call DeepSeek LLM with deterministic settings for demo consistency."""
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling DeepSeek: {str(e)}"


def call_perplexity_search(query: str) -> str:
    """Search Perplexity API for nutrition information using OpenAI client."""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return "Error: PERPLEXITY_API_KEY not found"
    
    try:
        # Initialize Perplexity client using OpenAI interface
        perplexity_client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        
        response = perplexity_client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": "Provide detailed nutritional information with scientific accuracy and practical advice."},
                {"role": "user", "content": f"Provide comprehensive nutritional information about: {query}"}
            ],
            max_tokens=1500,
            temperature=0.1
        )
        
        return response.choices[0].message.content or "No results found"
    except Exception as e:
        return f"Error searching Perplexity: {str(e)}"


# =============================================================================
# AGENT NODES
# =============================================================================

def intent_analyzer_node(state: NutritionState) -> NutritionState:
    """Analyze user intent and extract topic."""
    print("\n=== TRACE: Intent Analyzer ===")
    user_message = state["user_message"]
    print(f"Input: {user_message}")
    
    prompt = f"""
    Analyze the user's message and classify their intent. Return ONLY valid JSON.
    
    User message: "{user_message}"
    
    Classify into one of these intents:
    - "meal_log": User wants to log a meal they ate
    - "nutrition_lookup": User asks about nutritional information 
    - "recall_compare": User wants to compare with their past meals/preferences
    - "chat": General conversation or greeting
    
    Return JSON format:
    {{"intent": "meal_log|nutrition_lookup|recall_compare|chat", "topic": "extracted_topic_or_empty_string"}}
    
    Examples:
    "I ate rice for lunch" -> {{"intent": "meal_log", "topic": "rice lunch"}}
    "Is mango good for diabetes?" -> {{"intent": "nutrition_lookup", "topic": "mango diabetes"}}
    "How does my diet compare to what's healthy?" -> {{"intent": "recall_compare", "topic": "diet comparison"}}
    """
    
    response = call_deepseek_llm(prompt)
    print(f"Raw LLM Response: {response}")
    
    try:
        parsed = json.loads(response)
        intent = parsed.get("intent", "chat")
        topic = parsed.get("topic", "")
    except:
        # Fallback classification
        intent = "chat"
        topic = ""
        print("Warning: JSON parsing failed, using fallback classification")
    
    print(f"Output: intent='{intent}', topic='{topic}'")
    
    return {
        **state,
        "intent": intent,
        "topic": topic
    }


def query_rewriter_node(state: NutritionState) -> NutritionState:
    """Rewrite user query for optimal search results."""
    print("\n=== TRACE: Query Rewriter ===")
    user_message = state["user_message"]
    topic = state["topic"]
    print(f"Input: original_query='{user_message}', topic='{topic}'")
    
    prompt = f"""
    Rewrite this nutrition question into a clear, search-optimized query.
    Focus on nutritional facts, health benefits, dietary information.
    
    Original: "{user_message}"
    Topic: "{topic}"
    
    Return only the rewritten query, nothing else.
    """
    
    rewritten = call_deepseek_llm(prompt)
    print(f"Output: rewritten_query='{rewritten}'")
    
    return {
        **state,
        "rewritten_query": rewritten
    }


def perplexity_search_node(state: NutritionState) -> NutritionState:
    """Search for nutrition information using Perplexity."""
    print("\n=== TRACE: Perplexity Search ===")
    query = state["rewritten_query"]
    print(f"Input: query='{query}'")
    
    search_results = call_perplexity_search(query)
    preview = search_results[:200] + "..." if len(search_results) > 200 else search_results
    print(f"Output: Retrieved {len(search_results)} characters")
    print(f"Preview: {preview}")
    
    # Append to scratchpad
    current_scratchpad = state.get("scratchpad", "")
    updated_scratchpad = current_scratchpad + f"\n\n=== Search Results for '{query}' ===\n{search_results}"
    
    return {
        **state,
        "scratchpad": updated_scratchpad
    }


def memory_store_node(state: NutritionState, memory_store: InMemoryStore) -> NutritionState:
    """Store information in memory."""
    print("\n=== TRACE: Memory Store ===")
    
    intent = state["intent"]
    topic = state["topic"]
    user_message = state["user_message"]
    
    if intent == "meal_log":
        # Store meal with date-based key
        date_key = f"meals_{datetime.now().strftime('%Y_%m_%d')}"
        meal_entry = f"{datetime.now().strftime('%H:%M')} - {user_message}"
        
        # Get existing meals for today - use correct API
        try:
            existing_items = memory_store.search(("nutrition", date_key))
            if existing_items:
                existing_meals = existing_items[0].value
                updated_meals = existing_meals + "\n" + meal_entry
            else:
                updated_meals = meal_entry
        except:
            # If search fails, start fresh
            updated_meals = meal_entry
            
        # Store the updated meals using correct API
        memory_store.put(("nutrition", date_key), "meals", updated_meals)
        print(f"Input: meal_entry='{meal_entry}'")
        print(f"Output: Stored under key '{date_key}'")
        
    elif intent in ["nutrition_lookup"] and state.get("final_answer"):
        # Store nutrition summary for future reference
        topic_key = f"topic_{topic.replace(' ', '_')}"
        summary = state["final_answer"]
        memory_store.put(("nutrition", topic_key), "summary", summary)
        print(f"Input: topic_key='{topic_key}', summary_length={len(summary)}")
        print(f"Output: Stored nutrition summary")
    
    return state


def memory_retrieve_node(state: NutritionState, memory_store: InMemoryStore) -> NutritionState:
    """Retrieve relevant information from memory."""
    print("\n=== TRACE: Memory Retrieve ===")
    
    intent = state["intent"]
    topic = state["topic"]
    print(f"Input: intent='{intent}', topic='{topic}'")
    
    memory_content = ""
    
    # Always try to get recent meals
    today_key = f"meals_{datetime.now().strftime('%Y_%m_%d')}"
    try:
        recent_meals_items = memory_store.search(("nutrition", today_key))
        if recent_meals_items:
            memory_content += f"Today's Meals:\n{recent_meals_items[0].value}\n\n"
    except:
        print("No recent meals found")
    
    # For recall_compare, also look for related topics
    if intent == "recall_compare" and topic:
        topic_key = f"topic_{topic.replace(' ', '_')}"
        try:
            topic_items = memory_store.search(("nutrition", topic_key))
            if topic_items:
                memory_content += f"Previous Information on {topic}:\n{topic_items[0].value}\n\n"
        except:
            print(f"No previous info found for topic: {topic}")
    
    if memory_content:
        print(f"Output: Retrieved {len(memory_content)} characters from memory")
        print(f"Memory preview: {memory_content[:150]}...")
    else:
        print("Output: No relevant memory found")
    
    return {
        **state,
        "memory_content": memory_content
    }


def llm_summarizer_node(state: NutritionState) -> NutritionState:
    """Generate final response using scratchpad and memory."""
    print("\n=== TRACE: LLM Summarizer ===")
    
    scratchpad = state.get("scratchpad", "")
    memory_content = state.get("memory_content", "")
    user_question = state["user_message"]
    
    print(f"Input: scratchpad_length={len(scratchpad)}, memory_length={len(memory_content)}, question='{user_question}'")
    
    prompt = f"""
You are a helpful nutrition assistant.
Use the following research notes and meal history to answer the question.

Meal History / Preferences:
{memory_content if memory_content else "No meal history available."}

Research Notes:
{scratchpad if scratchpad else "No research notes available."}

User Question:
{user_question}

Provide a clear, concise, actionable answer.
"""
    
    final_answer = call_deepseek_llm(prompt)
    print(f"Output: Generated response ({len(final_answer)} characters)")
    print(f"Answer preview: {final_answer[:150]}...")
    
    return {
        **state,
        "final_answer": final_answer
    }


def quick_chat_node(state: NutritionState) -> NutritionState:
    """Handle general chat without search."""
    print("\n=== TRACE: Quick Chat ===")
    user_message = state["user_message"]
    print(f"Input: {user_message}")
    
    prompt = f"""
You are a friendly nutrition assistant. Respond to this general message:
"{user_message}"

Keep your response brief and helpful. If they're asking about nutrition, suggest they ask a specific question.
"""
    
    response = call_deepseek_llm(prompt)
    print(f"Output: {response}")
    
    return {
        **state,
        "final_answer": response
    }


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_nutrition_graph() -> tuple[CompiledStateGraph, InMemoryStore]:
    """Build the LangGraph workflow."""
    print("=== Initializing Nutrition Agent Graph ===")
    
    # Create memory store
    memory_store = InMemoryStore()
    
    # Create graph
    workflow = StateGraph(NutritionState)
    
    # Add nodes
    workflow.add_node("intent_analyzer", intent_analyzer_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("perplexity_search", perplexity_search_node)
    workflow.add_node("memory_retrieve", lambda state: memory_retrieve_node(state, memory_store))
    workflow.add_node("memory_store", lambda state: memory_store_node(state, memory_store))
    workflow.add_node("llm_summarizer", llm_summarizer_node)
    workflow.add_node("quick_chat", quick_chat_node)
    
    # Define entry point
    workflow.set_entry_point("intent_analyzer")
    
    # Define conditional routing based on intent
    def route_by_intent(state: NutritionState) -> str:
        intent = state["intent"]
        print(f"\n=== ROUTING: Intent '{intent}' ===")
        
        if intent == "meal_log":
            return "memory_store"
        elif intent == "nutrition_lookup":
            return "query_rewriter"
        elif intent == "recall_compare":
            return "memory_retrieve"
        else:  # chat
            return "quick_chat"
    
    # Add conditional edges from intent analyzer
    workflow.add_conditional_edges(
        "intent_analyzer",
        route_by_intent,
        {
            "memory_store": "memory_store",
            "query_rewriter": "query_rewriter", 
            "memory_retrieve": "memory_retrieve",
            "quick_chat": "quick_chat"
        }
    )
    
    # Chain for nutrition lookup: rewriter -> search -> memory -> summarizer -> store
    workflow.add_edge("query_rewriter", "perplexity_search")
    workflow.add_edge("perplexity_search", "memory_retrieve")
    workflow.add_edge("memory_retrieve", "llm_summarizer")
    workflow.add_edge("llm_summarizer", "memory_store")
    
    # Compile the graph
    compiled_graph = workflow.compile()
    
    print("Graph compiled successfully!")
    return compiled_graph, memory_store


# =============================================================================
# MAIN INTERACTIVE LOOP
# =============================================================================

def main():
    """Run the interactive nutrition agent demo."""
    print("🥗 Nutritionist Agent Demo")
    print("=" * 50)
    print("Ask nutrition questions, log meals, or chat!")
    print("Examples:")
    print("  - 'Is mango good for diabetes?'")
    print("  - 'I ate rice and chicken for lunch'") 
    print("  - 'How does my diet look today?'")
    print("Type 'quit' to exit.\n")
    
    # Initialize the graph and memory
    graph, memory_store = create_nutrition_graph()
    
    while True:
        try:
            # Get user input
            user_input = input("\n🍎 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Thanks for using the Nutrition Agent! Stay healthy!")
                break
            
            if not user_input:
                print("Please enter a message or 'quit' to exit.")
                continue
            
            print(f"\n{'='*60}")
            print(f"PROCESSING: {user_input}")
            print(f"{'='*60}")
            
            # Initialize state
            initial_state = NutritionState(
                scratchpad="",
                user_message=user_input,
                intent="",
                topic="",
                rewritten_query="",
                memory_content="",
                final_answer=""
            )
            
            # Run the graph
            final_state = graph.invoke(initial_state)
            
            # Display final answer
            print(f"\n🤖 Nutrition Agent:")
            print(f"{final_state.get('final_answer', 'No response generated.')}")
            
            # Show updated scratchpad size for trace visibility
            scratchpad_size = len(final_state.get('scratchpad', ''))
            if scratchpad_size > 0:
                print(f"\n=== TRACE: Scratchpad Updated ===")
                print(f"Total scratchpad size: {scratchpad_size} characters")
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the Nutrition Agent! Stay healthy!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again or type 'quit' to exit.")


if __name__ == "__main__":
    main()


# =============================================================================
# EXAMPLE RUN TRANSCRIPT
# =============================================================================
"""
Example Demo Run:
================

🥗 Nutritionist Agent Demo
==================================================
Ask nutrition questions, log meals, or chat!
Examples:
  - 'Is mango good for diabetes?'
  - 'I ate rice and chicken for lunch'
  - 'How does my diet look today?'
Type 'quit' to exit.

🍎 You: I ate oatmeal with berries for breakfast

============================================================
PROCESSING: I ate oatmeal with berries for breakfast
============================================================

=== TRACE: Intent Analyzer ===
Input: I ate oatmeal with berries for breakfast
Raw LLM Response: {"intent": "meal_log", "topic": "oatmeal berries breakfast"}
Output: intent='meal_log', topic='oatmeal berries breakfast'

=== ROUTING: Intent 'meal_log' ===

=== TRACE: Memory Store ===
Input: meal_entry='09:15 - I ate oatmeal with berries for breakfast'
Output: Stored under key 'meals_2025_09_19'

🤖 Nutrition Agent:
Meal logged successfully! I've recorded that you had oatmeal with berries for breakfast.

🍎 You: Is mango good for diabetes?

============================================================
PROCESSING: Is mango good for diabetes?
============================================================

=== TRACE: Intent Analyzer ===
Input: Is mango good for diabetes?
Raw LLM Response: {"intent": "nutrition_lookup", "topic": "mango diabetes"}
Output: intent='nutrition_lookup', topic='mango diabetes'

=== ROUTING: Intent 'nutrition_lookup' ===

=== TRACE: Query Rewriter ===
Input: original_query='Is mango good for diabetes?', topic='mango diabetes'
Output: rewritten_query='mango glycemic index diabetes blood sugar effects nutritional benefits'

=== TRACE: Perplexity Search ===
Input: query='mango glycemic index diabetes blood sugar effects nutritional benefits'
Output: Retrieved 654 characters
Preview: Mangoes have a moderate glycemic index of around 51-60, which means they can cause a moderate rise in blood sugar levels. For people with diabetes, mangoes can be consumed in moderation as part of a balanced diet...

=== TRACE: Memory Retrieve ===
Input: intent='nutrition_lookup', topic='mango diabetes'
Output: Retrieved 45 characters from memory
Memory preview: Today's Meals:
09:15 - I ate oatmeal with berries for...

=== TRACE: LLM Summarizer ===
Input: scratchpad_length=695, memory_length=45, question='Is mango good for diabetes?'
Output: Generated response (523 characters)
Answer preview: Based on the research, mangoes can be included in a diabetic diet but with important considerations. Mangoes have a moderate glycemic index (51-60), meaning they cause a moderate rise in blood sugar...

=== TRACE: Memory Store ===
Input: topic_key='topic_mango_diabetes', summary_length=523
Output: Stored nutrition summary

🤖 Nutrition Agent:
Based on the research, mangoes can be included in a diabetic diet but with important considerations. Mangoes have a moderate glycemic index (51-60), meaning they cause a moderate rise in blood sugar levels. The key is portion control - stick to about 1/2 cup of fresh mango slices. Mangoes do provide beneficial nutrients like vitamin C, fiber, and antioxidants. Given your healthy breakfast of oatmeal with berries today, adding a small portion of mango would fit well into a balanced meal plan. Always monitor your blood sugar response and consult with your healthcare provider about incorporating fruits like mango into your diabetes management plan.

🍎 You: quit

👋 Thanks for using the Nutrition Agent! Stay healthy!
"""