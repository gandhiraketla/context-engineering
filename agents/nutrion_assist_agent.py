"""
Nutritionist Agent Demo using LangGraph + Zep Cloud
==================================================
A demo agent that helps with nutrition questions, meal logging, and personalized advice.
Uses Zep Cloud for intelligent memory and context management.
"""

import os
import json
import requests
from datetime import datetime
from typing import TypedDict, Literal
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# Zep Cloud client
from zep_cloud.client import Zep
from zep_cloud import Message

# OpenAI client for DeepSeek and Perplexity APIs
from openai import OpenAI

# Load environment variables
load_dotenv()

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Nutrition Agent Zep"

# Get user ID from environment
ZEP_USER_ID = os.getenv("ZEP_USER_ID", "demo_nutrition_user")
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Initialize Zep client (will use mock if real client unavailable)
zep_client = Zep(
    api_key=os.getenv("ZEP_API_KEY", "mock_key_for_testing")
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
    memory_content: str  # Retrieved memory content from Zep
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
                {"role": "system", "content": "Provide detailed, accurate information with scientific backing. Focus on current research, evidence-based insights, and practical applications. Prioritize credible sources and peer-reviewed studies"},
                {"role": "user", "content": query}
            ],
            max_tokens=1500,
            temperature=0.1
        )
        
        return response.choices[0].message.content or "No results found"
    except Exception as e:
        return f"Error searching Perplexity: {str(e)}"


def get_daily_thread_id() -> str:
    """Generate daily thread ID for the user."""
    date_str = datetime.now().strftime('%Y_%m_%d')
    return f"nutrition_{ZEP_USER_ID}_{date_str}"


# =============================================================================
# HEALTH DATA CLASSIFICATION
# =============================================================================

def health_data_classifier(user_message: str, topic: str) -> dict:
    """Classify what type of health data is being logged."""
    msg_lower = user_message.lower()
    
    # Meal/food logging
    if any(word in msg_lower for word in ['ate', 'eat', 'had', 'breakfast', 'lunch', 'dinner', 'snack', 'meal', 'food']):
        return {"type": "meal", "category": "nutrition"}
    
    # Weight tracking
    elif any(word in msg_lower for word in ['weight', 'weigh', 'kg', 'lb', 'pounds', 'kilos']):
        return {"type": "weight", "category": "biometric"}
    
    # Blood sugar/glucose
    elif any(word in msg_lower for word in ['glucose', 'blood sugar', 'bg', 'mg/dl', 'mmol']):
        return {"type": "glucose", "category": "biometric"}
    
    # Blood pressure
    elif any(word in msg_lower for word in ['blood pressure', 'bp', '/', 'systolic', 'diastolic']) and any(char.isdigit() for char in user_message):
        return {"type": "blood_pressure", "category": "biometric"}
    
    # Exercise/activity
    elif any(word in msg_lower for word in ['walked', 'ran', 'exercise', 'workout', 'gym', 'minutes', 'steps']):
        return {"type": "exercise", "category": "activity"}
    
    # Symptoms
    elif any(word in msg_lower for word in ['feeling', 'feel', 'tired', 'headache', 'pain', 'bloated', 'nausea', 'dizzy']):
        return {"type": "symptom", "category": "wellness"}
    
    # Default to general health log
    else:
        return {"type": "general", "category": "health"}


# =============================================================================
# ZEP INTEGRATION FUNCTIONS
# =============================================================================

def ensure_user_exists():
    """Ensure the user exists in Zep Cloud."""
    try:
        # Try to get user, create if doesn't exist
        try:
            zep_client.user.get(user_id=ZEP_USER_ID)
            print(f"=== ZEP: User {ZEP_USER_ID} exists ===")
        except:
            # Create user if doesn't exist
            zep_client.user.add(
                user_id=ZEP_USER_ID,
                first_name="Demo",
                last_name="User",
                email="demo@nutrition-agent.com"
            )
            print(f"=== ZEP: Created user {ZEP_USER_ID} ===")
    except Exception as e:
        print(f"Warning: Could not ensure user exists: {e}")


def ensure_thread_exists(thread_id: str):
    """Ensure the thread exists for today."""
    try:
        # Try to create thread (Zep will handle if it already exists)
        zep_client.thread.create(
            thread_id=thread_id,
            user_id=ZEP_USER_ID
        )
        print(f"=== ZEP: Thread {thread_id} ready ===")
    except Exception as e:
        # Thread likely already exists, which is fine
        print(f"=== ZEP: Thread {thread_id} exists or created ===")


# =============================================================================
# AGENT NODES
# =============================================================================

def intent_analyzer_node(state: NutritionState) -> NutritionState:
    """Analyze user intent and extract topic."""
    print("\n" + "="*50)
    print("🧠 STEP 1: ANALYZING USER INTENT")
    print("="*50)
    user_message = state["user_message"]
    print(f"👤 User Input: '{user_message}'")
    
    prompt = f"""
    Analyze the user's message and classify their intent. Return ONLY valid JSON.
    
    User message: "{user_message}"
    
    Classify into one of these intents:
    - "health_log": User wants to log health data (meals, weight, blood pressure, glucose, symptoms, etc.)
    - "nutrition_lookup": User asks about nutritional information (questions about food/health)
    - "recall_compare": User wants to analyze their past data, patterns, or get recommendations based on health history
    - "chat": General conversation or greeting
    
    Return JSON format:
    {{"intent": "health_log|nutrition_lookup|recall_compare|chat", "topic": "extracted_topic_or_empty_string"}}
    
    Examples:
    "I ate rice for lunch" -> {{"intent": "health_log", "topic": "meal rice lunch"}}
    "My weight is 70kg today" -> {{"intent": "health_log", "topic": "weight 70kg"}}
    "Blood sugar reading 110 mg/dL" -> {{"intent": "health_log", "topic": "glucose 110"}}
    "BP: 120/80" -> {{"intent": "health_log", "topic": "blood pressure 120/80"}}
    "Feeling tired after lunch" -> {{"intent": "health_log", "topic": "symptom fatigue lunch"}}
    "Walked 30 minutes today" -> {{"intent": "health_log", "topic": "exercise walking 30min"}}
    "Is mango good for diabetes?" -> {{"intent": "nutrition_lookup", "topic": "mango diabetes"}}
    "What should I eat for protein?" -> {{"intent": "nutrition_lookup", "topic": "protein sources"}}
    "Analyze my meals today" -> {{"intent": "recall_compare", "topic": "meal analysis today"}}
    "How is my weight trending?" -> {{"intent": "recall_compare", "topic": "weight trend analysis"}}
    "Any patterns between my glucose and meals?" -> {{"intent": "recall_compare", "topic": "glucose meal correlation"}}
    "Based on my readings, recommend dinner" -> {{"intent": "recall_compare", "topic": "dinner recommendation"}}
    """
    
    print("🤖 DeepSeek LLM: Analyzing intent...")
    response = call_deepseek_llm(prompt)
    
    try:
        parsed = json.loads(response)
        intent = parsed.get("intent", "chat")
        topic = parsed.get("topic", "")
    except:
        # Fallback classification
        intent = "chat"
        topic = ""
        print("⚠️  JSON parsing failed, using fallback")
    
    print(f"✅ Intent Detected: '{intent}'")
    print(f"📝 Topic Extracted: '{topic}'")
    
    return {
        **state,
        "intent": intent,
        "topic": topic
    }


def query_rewriter_node(state: NutritionState) -> NutritionState:
    """Rewrite user query for optimal search results."""
    print("\n" + "="*50)
    print("✏️  STEP 2: OPTIMIZING SEARCH QUERY")
    print("="*50)
    user_message = state["user_message"]
    topic = state["topic"]
    print(f"🔍 Original Question: '{user_message}'")
    print(f"📋 Topic: '{topic}'")
    
    prompt = f"""
    Rewrite this nutrition question into a clear, search-optimized query.
    Focus on nutritional facts, health benefits, dietary information.
    
    Original: "{user_message}"
    Topic: "{topic}"
    
    Return only the rewritten query, nothing else.
    """
    
    print("🤖 DeepSeek LLM: Optimizing query for search...")
    rewritten = call_deepseek_llm(prompt)
    print(f"✅ Optimized Query: '{rewritten}'")
    
    return {
        **state,
        "rewritten_query": rewritten
    }


def perplexity_search_node(state: NutritionState) -> NutritionState:
    """Search for nutrition information using Perplexity."""
    print("\n" + "="*50)
    print("🔬 STEP 3: RESEARCHING NUTRITION INFORMATION")
    print("="*50)
    query = state["rewritten_query"]
    print(f"🔍 Searching for: '{query}'")
    print("🌐 Perplexity AI: Gathering latest nutrition research...")
    
    search_results = call_perplexity_search(query)
    preview = search_results[:150] + "..." if len(search_results) > 150 else search_results
    print(f"✅ Research Complete: {len(search_results)} characters retrieved")
    print(f"📄 Preview: {preview}")
    
    # Append to scratchpad
    current_scratchpad = state.get("scratchpad", "")
    updated_scratchpad = current_scratchpad + f"\n\n=== Research Results ===\n{search_results}"
    
    return {
        **state,
        "scratchpad": updated_scratchpad
    }


def zep_memory_store_node(state: NutritionState) -> NutritionState:
    """Store information in Zep Cloud memory."""
    print("\n" + "="*50)
    print("💾 ZEP CLOUD: STORING INFORMATION")
    print("="*50)
    
    intent = state["intent"]
    topic = state["topic"]
    user_message = state["user_message"]
    thread_id = get_daily_thread_id()
    
    # Preserve the final_answer from previous steps
    final_answer = state.get("final_answer", "")
    
    # Ensure user and thread exist
    ensure_user_exists()
    ensure_thread_exists(thread_id)
    
    memory_content = state.get("memory_content", "")
    
    try:
        if intent == "meal_log":
            print(f"🍽️ Storing Meal: '{user_message}'")
            print(f"📅 Thread: {thread_id}")
            
            result = zep_client.thread.add_messages(
                thread_id=thread_id,
                messages=[Message(
                    role="user",
                    name="Demo User",
                    content=user_message,
                    metadata={
                        "type": "meal_log",
                        "timestamp": datetime.now().isoformat(),
                        "topic": topic
                    }
                )],
                return_context=True
            )
            
            memory_content = result.context if hasattr(result, 'context') else ""
            final_answer = f"✅ Meal logged successfully! Recorded: {user_message}"
            
            print(f"✅ Storage Complete: {len(memory_content)} chars of context retrieved")
            
        elif intent == "nutrition_lookup" and final_answer:
            print(f"🧠 Storing Nutrition Insights for: '{topic}'")
            key_insights = extract_nutrition_insights(final_answer, topic)
            
            zep_client.thread.add_messages(
                thread_id=thread_id,
                messages=[Message(
                    role="assistant",
                    name="Nutrition Agent",
                    content=f"Key insight: {key_insights}",
                    metadata={
                        "type": "nutrition_insight",
                        "topic": topic,
                        "timestamp": datetime.now().isoformat()
                    }
                )]
            )
            print(f"✅ Insights stored for future reference")
            
    except Exception as e:
        print(f"❌ Storage Error: {e}")
        if intent == "meal_log":
            final_answer = f"⚠️ Meal noted: {user_message} (storage temporarily unavailable)"
    
    return {
        **state,
        "memory_content": memory_content,
        "final_answer": final_answer  # Make sure to preserve the final_answer
    }


def zep_health_store_node(state: NutritionState) -> NutritionState:
    """Store health information in Zep Cloud memory."""
    print("\n" + "="*50)
    print("💾 ZEP CLOUD: STORING HEALTH DATA")
    print("="*50)
    
    intent = state["intent"]
    topic = state["topic"]
    user_message = state["user_message"]
    thread_id = get_daily_thread_id()
    
    # Preserve the final_answer from previous steps
    final_answer = state.get("final_answer", "")
    
    # Ensure user and thread exist
    ensure_user_exists()
    ensure_thread_exists(thread_id)
    
    memory_content = state.get("memory_content", "")
    
    try:
        if intent == "health_log":
            # Classify the type of health data
            health_data = health_data_classifier(user_message, topic)
            
            print(f"🏥 Storing {health_data['category'].title()}: {health_data['type']} data")
            print(f"📝 Entry: '{user_message}'")
            print(f"📅 Thread: {thread_id}")
            
            result = zep_client.thread.add_messages(
                thread_id=thread_id,
                messages=[Message(
                    role="user",
                    name="Demo User",
                    content=user_message,
                    metadata={
                        "type": "health_log",
                        "data_type": health_data["type"],
                        "category": health_data["category"],
                        "timestamp": datetime.now().isoformat(),
                        "topic": topic
                    }
                )],
                return_context=True
            )
            
            memory_content = result.context if hasattr(result, 'context') else ""
            final_answer = f"✅ {health_data['category'].title()} data logged: {user_message}"
            
            print(f"✅ Storage Complete: {len(memory_content)} chars of context retrieved")
            
        elif intent == "nutrition_lookup" and final_answer:
            print(f"🧠 Storing Nutrition Insights for: '{topic}'")
            key_insights = extract_nutrition_insights(final_answer, topic)
            
            zep_client.thread.add_messages(
                thread_id=thread_id,
                messages=[Message(
                    role="assistant",
                    name="Nutrition Agent",
                    content=f"Key insight: {key_insights}",
                    metadata={
                        "type": "nutrition_insight",
                        "topic": topic,
                        "timestamp": datetime.now().isoformat()
                    }
                )]
            )
            print(f"✅ Insights stored for future reference")
            
    except Exception as e:
        print(f"❌ Storage Error: {e}")
        if intent == "health_log":
            final_answer = f"⚠️ Health data noted: {user_message} (storage temporarily unavailable)"
    
    return {
        **state,
        "memory_content": memory_content,
        "final_answer": final_answer
    }


def zep_memory_retrieve_node(state: NutritionState) -> NutritionState:
    """Retrieve relevant information from Zep Cloud memory."""
    print("\n" + "="*50)
    print("🧠 ZEP CLOUD: RETRIEVING MEMORY")
    print("="*50)
    
    intent = state["intent"]
    topic = state["topic"]
    thread_id = get_daily_thread_id()
    
    print(f"🔍 Looking for: {topic}")
    print(f"📅 Thread: {thread_id}")
    
    # Ensure user and thread exist
    ensure_user_exists()
    ensure_thread_exists(thread_id)
    
    memory_content = ""
    
    try:
        print("🌐 Querying Zep's knowledge graph...")
        context = zep_client.thread.get_user_context(
            thread_id=thread_id,
            mode="basic"
        )
        
        memory_content = context.context if hasattr(context, 'context') else ""
        
        if memory_content:
            print(f"✅ Memory Retrieved: {len(memory_content)} characters")
            # Count facts and entities for demo
            fact_count = memory_content.count('- ') if '<FACTS>' in memory_content else 0
            entity_count = memory_content.count('Name:') if '<ENTITIES>' in memory_content else 0
            print(f"📊 Found: {fact_count} facts, {entity_count} entities")
        else:
            print("📭 No relevant memory found")
            
    except Exception as e:
        print(f"❌ Retrieval Error: {e}")
        memory_content = ""
    
    return {
        **state,
        "memory_content": memory_content
    }


def extract_nutrition_insights(final_answer: str, topic: str) -> str:
    """Extract key insights from research for long-term memory storage."""
    prompt = f"""
    Extract the 2-3 most important, actionable nutrition insights from this response about {topic}.
    Focus on facts that would be useful for future meal planning and nutrition advice.
    Keep it concise (under 200 words).
    
    Full response: {final_answer}
    
    Return only the key insights, nothing else.
    """
    
    try:
        insights = call_deepseek_llm(prompt)
        return insights
    except Exception as e:
        # Fallback: just take first 200 chars of final answer
        return final_answer[:200] + "..." if len(final_answer) > 200 else final_answer


def llm_summarizer_node(state: NutritionState) -> NutritionState:
    """Generate final response using scratchpad (temporary) and Zep memory (persistent)."""
    print("\n" + "="*50)
    print("🤖 STEP 4: GENERATING PERSONALIZED RESPONSE")
    print("="*50)
    
    scratchpad = state.get("scratchpad", "")  # Temporary research data
    memory_content = state.get("memory_content", "")  # User's persistent context
    user_question = state["user_message"]
    intent = state["intent"]
    
    print(f"💭 Question: '{user_question}'")
    print(f"📄 Research Data: {len(scratchpad)} characters")
    print(f"🧠 User Memory: {len(memory_content)} characters")
    print("🤖 DeepSeek LLM: Combining research + personal history...")
    
    # Different prompts for different intents
    if intent == "recall_compare":
        # Determine if asking about specific health data vs discussions
        is_health_data_query = any(word in user_question.lower() for word in ['eat', 'ate', 'meal', 'weight', 'glucose', 'sugar', 'pressure', 'bp', 'exercise', 'symptom', 'today', 'yesterday'])
        
        if is_health_data_query:
            prompt = f"""
You are a health assistant analyzing the user's health patterns and data.

HEALTH HISTORY (from Zep):
{memory_content if memory_content else "No health data available."}

User Question: {user_question}

Look at the FACTS section for health data including meals, weight, glucose readings, blood pressure, symptoms, and exercise. Group related data by type and time. Identify patterns and correlations between different health metrics. Provide insights about trends and relationships in their health data.
"""
        else:
            prompt = f"""
You are a health assistant recalling previous discussions and research.

CONVERSATION HISTORY (from Zep):
{memory_content if memory_content else "No previous discussions available."}

User Question: {user_question}

Look through the FACTS and ENTITIES for any mentions of the health or nutrition topic they're asking about. Search for previous research, discussions, or advice given about specific foods, health conditions, or wellness topics. If found, summarize what was discussed. If not found, clearly state no previous discussion exists about that topic.
"""
    else:
        prompt = f"""
You are a helpful nutrition assistant with access to the user's meal history and current research.

USER'S NUTRITION CONTEXT (from Zep memory - long-term patterns and preferences):
{memory_content if memory_content else "No previous meal history or nutrition context available."}

CURRENT RESEARCH (temporary - use for this response only):
{scratchpad if scratchpad else "No research notes available."}

User Question: {user_question}

Instructions:
- Use the user's meal history and preferences to make recommendations personal and relevant
- Use the current research to provide accurate, up-to-date nutrition information
- Keep the response clear, concise, and actionable
- Focus on how this information applies to the user's eating patterns

Provide a personalized nutrition response.
"""
    
    final_answer = call_deepseek_llm(prompt)
    print(f"✅ Response Generated: {len(final_answer)} characters")
    
    return {
        **state,
        "final_answer": final_answer
    }


def quick_chat_node(state: NutritionState) -> NutritionState:
    """Handle general chat without search."""
    print("\n" + "="*50)
    print("💬 QUICK CHAT RESPONSE")
    print("="*50)
    user_message = state["user_message"]
    print(f"👤 User: '{user_message}'")
    print("🤖 Generating friendly response...")
    
    prompt = f"""
You are a friendly nutrition assistant. Respond to this general message:
"{user_message}"

Keep your response brief and helpful. If they're asking about nutrition, suggest they ask a specific question.
"""
    
    response = call_deepseek_llm(prompt)
    print(f"✅ Chat Response Ready")
    
    return {
        **state,
        "final_answer": response
    }


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_nutrition_graph() -> CompiledStateGraph:
    """Build the LangGraph workflow with Zep Cloud integration."""
    print("=== Initializing Nutrition Agent Graph with Zep Cloud ===")
    
    # Create graph
    workflow = StateGraph(NutritionState)
    
    # Add nodes
    workflow.add_node("intent_analyzer", intent_analyzer_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("perplexity_search", perplexity_search_node)
    workflow.add_node("zep_memory_retrieve", zep_memory_retrieve_node)
    workflow.add_node("zep_health_store", zep_health_store_node)
    workflow.add_node("llm_summarizer", llm_summarizer_node)
    workflow.add_node("quick_chat", quick_chat_node)
    
    # Define entry point
    workflow.set_entry_point("intent_analyzer")
    
    # Define conditional routing based on intent
    def route_by_intent(state: NutritionState) -> str:
        intent = state["intent"]
        print(f"\n🎯 ROUTING: Detected '{intent}' → Choosing appropriate path")
        
        if intent == "health_log":
            print("📝 Path: Direct health data storage")
            return "zep_health_store"
        elif intent == "nutrition_lookup":
            print("🔬 Path: Research → Analysis → Storage")
            return "query_rewriter"
        elif intent == "recall_compare":
            print("🧠 Path: Memory retrieval → Analysis")
            return "zep_memory_retrieve"
        else:  # chat
            print("💬 Path: Quick chat response")
            return "quick_chat"
    
    # Add conditional edges from intent analyzer
    workflow.add_conditional_edges(
        "intent_analyzer",
        route_by_intent,
        {
            "zep_health_store": "zep_health_store",
            "query_rewriter": "query_rewriter", 
            "zep_memory_retrieve": "zep_memory_retrieve",
            "quick_chat": "quick_chat"
        }
    )
    
    # Chain for nutrition lookup: rewriter -> search -> memory -> summarizer -> store
    workflow.add_edge("query_rewriter", "perplexity_search")
    workflow.add_edge("perplexity_search", "zep_memory_retrieve")
    
    # Connect memory retrieve to summarizer for both recall_compare and nutrition_lookup
    workflow.add_edge("zep_memory_retrieve", "llm_summarizer")
    
    # For recall/compare: memory -> summarizer (ends here)
    # For nutrition_lookup: memory -> summarizer -> store
    def route_after_summarizer(state: NutritionState) -> str:
        intent = state["intent"]
        if intent == "nutrition_lookup":
            return "zep_health_store"
        else:
            return "END"
    
    # Add conditional edge after summarizer
    workflow.add_conditional_edges(
        "llm_summarizer",
        route_after_summarizer,
        {
            "zep_health_store": "zep_health_store",
            "END": END
        }
    )
    
    # Compile the graph
    compiled_graph = workflow.compile()
    
    print("Graph compiled successfully with Zep Cloud integration!")
    return compiled_graph


# =============================================================================
# MAIN INTERACTIVE LOOP
# =============================================================================

def main():
    """Run the interactive nutrition agent demo with Zep Cloud."""
    print("\n" + "🥗" * 20)
    print("    NUTRITIONIST AGENT DEMO")
    print("    Powered by Zep Cloud + LangGraph")
    print("🥗" * 20)
    print(f"\n👤 Demo User: {ZEP_USER_ID}")
    print("\n📋 What you can try:")
    print("  🍽️  Log meals: 'I ate rice and beans for lunch'")
    print("  🔬 Ask nutrition: 'Is avocado good for heart health?'") 
    print("  📊 Review diet: 'What did I eat today?'")
    print("  💬 General chat: 'Hello!'")
    print("\n⌨️  Type 'quit' to exit")
    print("="*60)
    
    # Initialize the graph
    print("\n🚀 Initializing Nutrition Agent...")
    graph = create_nutrition_graph()
    
    # Ensure user exists in Zep
    ensure_user_exists()
    print("✅ Agent ready for demo!\n")
    
    while True:
        try:
            # Get user input
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("\n🎉 Demo complete! Thanks for trying the Nutrition Agent!")
                break
            
            if not user_input:
                print("💡 Please enter a message or 'quit' to exit.")
                continue
            
            print(f"\n{'🔄 PROCESSING' + '='*48}")
            
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
            print("\n" + "🤖 NUTRITION AGENT RESPONSE" + "="*28)
            print(final_state.get('final_answer', 'No response generated.'))
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\n🎉 Demo interrupted! Thanks for trying the Nutrition Agent!")
            break
        except Exception as e:
            print(f"\n❌ Demo Error: {str(e)}")
            print("💡 Please try again or type 'quit' to exit.")


if __name__ == "__main__":
    main()
    graph = create_nutrition_graph()
    
    # Ensure user exists in Zep
    ensure_user_exists()
    
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
            
            # Show memory context if available
            memory_size = len(final_state.get('memory_content', ''))
            if memory_size > 0:
                print(f"\n=== TRACE: Used Zep Context ===")
                print(f"Context size: {memory_size} characters")
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the Nutrition Agent! Stay healthy!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again or type 'quit' to exit.")


if __name__ == "__main__":
    main()


# =============================================================================
# EXAMPLE ENVIRONMENT SETUP
# =============================================================================
"""
Create a .env file with:

ZEP_API_KEY=z_your_zep_cloud_api_key_here
ZEP_USER_ID=demo_nutrition_user
DEEPSEEK_API_KEY=your_deepseek_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key

Run with:
pip install zep-cloud langgraph openai python-dotenv
python nutrition_agent_zep.py
"""