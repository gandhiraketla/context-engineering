#!/usr/bin/env python3
"""
FinTech Customer Support Triage: Context Quarantine vs Non-Quarantined Agents

This script demonstrates the difference between:
1. Non-Quarantined Agent: Single agent with full access to all tools and conversation history
   - Risk: Context Clash (mixing unrelated info) and Context Distraction (losing focus)
2. Quarantined Multi-Agent Supervisor: Specialized agents with minimal, focused context
   - Benefit: Each agent sees only relevant information for their specific domain

The demo shows how context quarantine can improve response quality and reduce confusion
in complex multi-domain scenarios like FinTech customer support.
"""

import os
import sys
from typing import Dict, Any, TypedDict, Literal
from dotenv import load_dotenv
from langsmith import traceable

# LangChain imports
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage

# LangGraph imports
from langgraph.graph import StateGraph, END


# ===== ENVIRONMENT SETUP =====
def setup_environment():
    """Load environment and validate API keys."""
    print("🔧 Setting up environment...")
    
    # Load .env file
    load_dotenv()
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in environment variables!")
        print("Please create a .env file with: OPENAI_API_KEY=your_api_key_here")
        sys.exit(1)
    
    print(f"✅ OpenAI API key loaded (ends with: ...{api_key[-4:]})")
    
    # Setup LangSmith tracing (optional)
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "fintech-context-quarantine-demo"
        print(f"✅ LangSmith tracing enabled (key ends with: ...{langsmith_api_key[-4:]})")
        print("🔗 Traces will be available at: https://smith.langchain.com/")
    else:
        print("⚠️  LangSmith tracing disabled (LANGSMITH_API_KEY not found)")
        print("   Add LANGSMITH_API_KEY to .env file to enable detailed tracing")
    
    return api_key


# ===== DEMO TOOLS (HARD-CODED) =====
@traceable(name="billing_tool")
def billing_tool(customer_id: str) -> str:
    """Fake billing tool that returns transaction history and decline reasons."""
    fake_data = {
        "12345": {
            "transactions": ["$50.00 - Coffee Shop (Approved)", "$1,200.00 - Rent Payment (Declined)"],
            "decline_reason": "Insufficient funds - Account balance: $875.23",
            "additional_info": "Account opened 2019. Credit score: 720. Overdraft protection: Enabled."
        },
        "67890": {
            "transactions": ["$25.99 - Gas Station (Approved)", "$500.00 - ATM Withdrawal (Declined)"],
            "decline_reason": "Daily limit exceeded - Limit: $300.00",
            "additional_info": "Premium account. Credit score: 780. Multiple linked accounts."
        }
    }
    
    data = fake_data.get(customer_id, {
        "transactions": ["$15.50 - Grocery Store (Approved)"],
        "decline_reason": "Card expired - Please request new card",
        "additional_info": "Standard account. No overdrafts in last 12 months."
    })
    
    result = f"[BILLING] Customer {customer_id}: Recent transactions: {', '.join(data['transactions'])}. Decline reason: {data['decline_reason']}. Additional: {data['additional_info']}"
    return result


@traceable(name="fraud_tool") 
def fraud_tool(customer_id: str) -> str:
    """Fake fraud detection tool."""
    fake_flags = {
        "12345": "No fraud flags detected. Account in good standing. Last security check: 2024-09-15. Trusted devices: 3.",
        "67890": "⚠️ FRAUD ALERT: Suspicious login from new location (IP: 192.168.1.1, Location: Unknown). Card temporarily frozen. Contacted via SMS.",
        "99999": "Multiple failed PIN attempts detected. Security lockout active. Last attempt: 2024-09-20 14:30. Location: ATM #4421."
    }
    
    result = fake_flags.get(customer_id, "No fraud activity detected. Account secure. Clean history for 24 months.")
    return f"[FRAUD] Customer {customer_id}: {result}"


@traceable(name="faq_tool")
def faq_tool(query: str) -> str:
    """Fake FAQ tool with canned responses."""
    query_lower = query.lower()
    
    if "pin" in query_lower or "password" in query_lower:
        return "[FAQ] PIN Reset: Call 1-800-BANK or use mobile app > Settings > Reset PIN. You'll need your SSN and account number. Process takes 2-3 minutes. Available 24/7."
    elif "address" in query_lower or "move" in query_lower:
        return "[FAQ] Address Update: Log into online banking > Profile > Update Address, or visit any branch with proof of residence. Updates take 1-2 business days. Affects all linked accounts."
    elif "hours" in query_lower or "open" in query_lower:
        return "[FAQ] Branch Hours: Mon-Fri 9AM-5PM, Sat 9AM-1PM, Closed Sundays. ATMs available 24/7. Holiday hours may vary. Drive-thru closes 30min earlier."
    elif "fee" in query_lower:
        return "[FAQ] Fees: No monthly fee with $500+ balance. ATM fees: $2.50 for out-of-network. Overdraft: $35 per incident. Wire transfer: $15 domestic, $45 international."
    else:
        return "[FAQ] For other questions, please visit our website at bank.com or call customer service at 1-800-BANK-HELP (24/7). Live chat available Mon-Fri 8AM-8PM."


# ===== NON-QUARANTINED AGENT =====
class NonQuarantinedAgent:
    """Single agent with full access to all tools and conversation history."""
    
    def __init__(self, llm):
        self.llm = llm
        self.conversation_history = []  # Accumulates ALL conversation context
        
        # Create tools
        tools = [
            Tool(name="billing", func=billing_tool, description="Get billing info and transaction history for a customer"),
            Tool(name="fraud", func=fraud_tool, description="Check fraud status and security flags for a customer"),
            Tool(name="faq", func=faq_tool, description="Get answers to frequently asked questions")
        ]
        
        # Initialize agent with ALL tools
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )
    
    @traceable(name="non_quarantined_agent_query")
    def handle_query(self, customer_id: str, query: str, show_context: bool = False) -> str:
        """Process query with full context and all tools available."""
        try:
            # Add current query to conversation history
            self.conversation_history.append(f"Customer {customer_id}: {query}")
            
            # Build full context with ALL conversation history
            history_context = "\n".join(self.conversation_history[-5:])  # Last 5 interactions
            full_prompt = f"""
CONVERSATION HISTORY:
{history_context}

CURRENT QUERY: {query}
CUSTOMER ID: {customer_id}

You have access to billing, fraud, and FAQ tools. Use any tools you think are relevant.
Provide helpful customer support considering the full conversation context.
"""
            
            if show_context:
                print(f"    📋 Context seen by Non-Quarantined Agent:")
                print(f"       - Current query: '{query}'")
                print(f"       - Customer ID: {customer_id}")
                print(f"       - Conversation history: {len(self.conversation_history)} messages")
                print(f"       - Available tools: billing, fraud, faq (ALL TOOLS)")
                print(f"       - Decision making: Agent chooses which tools to use")
                print(f"       - Full context includes: {history_context[-100:]}..." if len(history_context) > 100 else f"       - Full context: {history_context}")
            
            response = self.agent.run(full_prompt)
            
            # Add response to history for context contamination
            self.conversation_history.append(f"Agent: {response[:100]}...")
            
            return response
        except Exception as e:
            return f"Error processing query: {str(e)}"


# ===== QUARANTINED MULTI-AGENT SUPERVISOR =====
class SupportState(TypedDict):
    """State for the quarantined multi-agent system."""
    customer_id: str
    query: str
    billing_result: str
    fraud_result: str
    faq_result: str
    route_decision: str


@traceable(name="billing_node")
def billing_node(state: SupportState) -> SupportState:
    """Specialized billing agent - only sees billing-related context."""
    print("  🏦 Billing agent activated...")
    print(f"    📋 Context seen: Customer ID {state['customer_id']}, Query: '{state['query']}'")
    print(f"    🔒 Quarantined: ONLY has access to billing_tool")
    result = billing_tool(state["customer_id"])
    return {**state, "billing_result": result}


@traceable(name="fraud_node")
def fraud_node(state: SupportState) -> SupportState:
    """Specialized fraud agent - only sees fraud-related context."""
    print("  🛡️ Fraud agent activated...")
    print(f"    📋 Context seen: Customer ID {state['customer_id']}, Query: '{state['query']}'")
    print(f"    🔒 Quarantined: ONLY has access to fraud_tool")
    result = fraud_tool(state["customer_id"])
    return {**state, "fraud_result": result}


@traceable(name="faq_node")
def faq_node(state: SupportState) -> SupportState:
    """Specialized FAQ agent - only sees the query context."""
    print("  ❓ FAQ agent activated...")
    print(f"    📋 Context seen: Query: '{state['query']}'")
    print(f"    🔒 Quarantined: ONLY has access to faq_tool")
    result = faq_tool(state["query"])
    return {**state, "faq_result": result}


@traceable(name="llm_support_router")
def llm_support_router(state: SupportState, llm) -> Literal["billing", "fraud", "faq"]:
    """Use LLM to intelligently route queries based on semantic understanding."""
    
    routing_prompt = f"""
You are an intelligent customer support routing system for a FinTech company. 

Analyze this customer query and route it to the most appropriate specialist team:

CUSTOMER QUERY: "{state['query']}"

AVAILABLE SPECIALIST TEAMS:
1. BILLING - Handles: account balances, transactions, payments, declined cards, charges, refunds, billing disputes, account statements
2. FRAUD - Handles: security concerns, suspicious activity, unauthorized transactions, account breaches, stolen cards, identity theft, security alerts
3. FAQ - Handles: general questions, account procedures, policies, branch hours, fees, how-to instructions, password/PIN resets, address updates

ROUTING RULES:
- Choose the team that best matches the customer's primary concern
- If the query involves money/transactions → usually BILLING
- If the query involves security/unauthorized activity → usually FRAUD  
- If the query is asking "how to" do something or general info → usually FAQ
- When in doubt, default to FAQ

Respond with EXACTLY ONE WORD: billing, fraud, or faq

DECISION:"""
    
    print(f"  🤖 LLM analyzing query: '{state['query']}'")
    response = llm.invoke(routing_prompt)
    route = response.content.strip().lower()
    
    # Validation
    valid_routes = ["billing", "fraud", "faq"]
    if route not in valid_routes:
        print(f"  ⚠️  Invalid LLM response: '{route}', defaulting to FAQ")
        route = "faq"
    
    print(f"  🧭 LLM Router decision: {route}")
    return route


class QuarantinedSupervisor:
    """Multi-agent system with context quarantine and LLM-based routing."""
    
    def __init__(self, llm):
        self.llm = llm
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the state graph with quarantined agents and LLM router."""
        workflow = StateGraph(SupportState)
        
        # Add specialized nodes
        workflow.add_node("billing", billing_node)
        workflow.add_node("fraud", fraud_node)
        workflow.add_node("faq", faq_node)
        
        # Create routing function that has access to LLM
        def router_with_llm(state):
            return llm_support_router(state, self.llm)
        
        # Set entry point with LLM-based routing
        workflow.set_conditional_entry_point(
            router_with_llm,
            {
                "billing": "billing",
                "fraud": "fraud", 
                "faq": "faq"
            }
        )
        
        # All nodes end the workflow
        workflow.add_edge("billing", END)
        workflow.add_edge("fraud", END)
        workflow.add_edge("faq", END)
        
        return workflow.compile()
    
    @traceable(name="quarantined_supervisor_query")
    def handle_query(self, customer_id: str, query: str, show_context: bool = False) -> str:
        """Process query through quarantined specialist agents."""
        try:
            # Create minimal initial state
            initial_state = SupportState(
                customer_id=customer_id,
                query=query,
                billing_result="",
                fraud_result="",
                faq_result="",
                route_decision=""
            )
            
            if show_context:
                print(f"    📋 Initial state passed to LLM router:")
                print(f"       - Query: '{query}'")
                print(f"       - Customer ID: {customer_id}")
                print(f"       - LLM will analyze semantics and decide routing")
                print(f"       - No conversation history - clean slate every time")
            
            # Run through the graph
            result = self.graph.invoke(initial_state)
            
            # Return the relevant result based on routing
            if result.get("billing_result"):
                return result["billing_result"]
            elif result.get("fraud_result"):
                return result["fraud_result"]
            else:
                return result.get("faq_result", "No response generated")
                
        except Exception as e:
            return f"Error processing query: {str(e)}"


# ===== INTERACTIVE DEMO =====
def run_interactive_demo():
    """Run the interactive comparison demo."""
    print("🚀 Starting FinTech Context Quarantine Demo")
    print("=" * 60)
    
    # Setup
    setup_environment()
    
    # Initialize LLM
    print("🧠 Initializing LLM...")
    try:
        llm = init_chat_model("openai:gpt-4o")
        print("✅ LLM initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return
    
    # Initialize agents
    print("🤖 Setting up agents...")
    non_quarantined = NonQuarantinedAgent(llm)
    quarantined = QuarantinedSupervisor(llm)
    print("✅ Agents ready")
    
    print("\n" + "=" * 60)
    print("DEMO: Context Quarantine vs Non-Quarantined Agents")
    print("=" * 60)
    
    # Show LangSmith info if enabled
    if os.getenv("LANGSMITH_API_KEY"):
        print("🔍 LangSmith Tracing Enabled!")
        print("   Project: fintech-context-quarantine-demo")
        print("   View traces at: https://smith.langchain.com/")
        print("   Each query will create detailed execution traces")
        print()
    
    print("Try these queries to see LLM-based routing vs keyword-based routing:")
    print("• 'Why was my card declined?' → Should route to BILLING")
    print("• 'I think someone hacked my account' → Should route to FRAUD")
    print("• 'How do I reset my PIN?' → Should route to FAQ")
    print("• 'My payment was rejected unexpectedly' → LLM semantic understanding!")
    print("• 'There's a weird charge I didn't make' → Tests LLM intelligence!")
    print("• 'Can you help me understand the fees?' → General question test")
    print("\n🧠 NOTICE: LLM Router uses semantic understanding, not just keywords!")
    print("🔥 CONTEXT: Non-quarantined agent still accumulates conversation history!")
    print("\nType 'quit' to exit")
    print("-" * 60)
    
    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for trying the demo!")
                break
            
            if not user_input:
                continue
            
            # Use a default customer ID for demo
            customer_id = "12345"
            
            print(f"\n🔍 Processing query for Customer {customer_id}...")
            print(f"💭 Query #{len(non_quarantined.conversation_history) + 1}")
            print("-" * 40)
            
            # Non-Quarantined Agent
            print("🔴 Non-Quarantined Agent:")
            try:
                non_q_response = non_quarantined.handle_query(customer_id, user_input, show_context=True)
                print(f"   {non_q_response}")
            except Exception as e:
                print(f"   Error: {e}")
            
            print()
            
            # Quarantined Supervisor
            print("🟢 Quarantined Supervisor:")
            try:
                q_response = quarantined.handle_query(customer_id, user_input, show_context=True)
                print(f"   {q_response}")
            except Exception as e:
                print(f"   Error: {e}")
            
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    run_interactive_demo()