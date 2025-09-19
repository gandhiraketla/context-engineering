import os
import sys
import uuid
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.chat_models import init_chat_model
from langchain.agents import initialize_agent, AgentType
from langchain.agents import AgentExecutor
from langgraph_bigtool.utils import convert_positional_only_function_to_tool

# LangSmith tracing
import langsmith
from langsmith import traceable

# Import Pinecone ToolRetriever
from storage.tool_retriever import ToolRetriever

# Import our DevOps tools
from tools import (
    EC2AndKubernetesTools,
    DatabaseTools,
    MonitoringTools,
    NetworkingTools,
    UtilityTools
)

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Set LangSmith project explicitly
os.environ["LANGSMITH_PROJECT"] = "ContextEngineering-Toolloadout"

# Initialize LLM
llm = init_chat_model(
    "openai:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0
)

class DevOpsReActAgent:
    """DevOps ReAct Agent with Pinecone Tool Retrieval"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        # Initialize ToolRetriever with error handling
        try:
            self.retriever = ToolRetriever()
            if self.verbose:
                print("✅ ToolRetriever initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing ToolRetriever: {e}")
            print("Please check if the ToolRetriever class is properly configured")
            raise e
            
        self.setup_tools()
        
    def setup_tools(self):
        """Convert and index all DevOps tools"""
        if self.verbose:
            print("Converting DevOps tools...")
        
        # Create tool instances
        tool_instances = [
            EC2AndKubernetesTools(),
            DatabaseTools(),
            MonitoringTools(),
            NetworkingTools(),
            UtilityTools()
        ]
        
        # Convert all methods to tools
        all_tools = []
        for instance in tool_instances:
            for method_name in dir(instance):
                if not method_name.startswith('_'):
                    method = getattr(instance, method_name)
                    if callable(method):
                        if tool := convert_positional_only_function_to_tool(method):
                            all_tools.append(tool)
        
        self.all_tools = all_tools
        
        if self.verbose:
            print(f"Total tools converted: {len(all_tools)}")
    
    def select_tools(self, query: str, use_tool_loadout: bool = True):
        """Tool selection using Pinecone ToolRetriever"""
        
        if use_tool_loadout:
            # Tool Loadout: Use Pinecone ToolRetriever
            print(f"🔍 Tool Loadout: Using Pinecone retriever for semantic search")
            
            # Retrieve relevant methods using Pinecone
            methods = self.retriever.retrieve_tools(query, top_k=10)
            
            if methods:
                tool_info = self.retriever.get_tool_info(methods)
                print(f"\n📋 Retrieved {len(tool_info)} Tools from Pinecone:")
                
                selected_tools = []
                tool_names = []
                
                for i, info in enumerate(tool_info, 1):
                    class_name = info['class_name']
                    method_name = info['method_name']
                    
                    print(f"   {i}. {class_name}.{method_name}")
                    if info.get('description'):
                        print(f"      Description: {info['description']}")
                    
                    # Find the corresponding tool object in self.all_tools
                    matching_tool = None
                    for tool in self.all_tools:
                        if tool.name == method_name:
                            matching_tool = tool
                            break
                    
                    if matching_tool:
                        selected_tools.append(matching_tool)
                        tool_names.append(method_name)
                    else:
                        print(f"      ⚠️ Warning: Tool {method_name} not found in self.all_tools")
                
                print(f"🔧 Successfully mapped {len(selected_tools)} Pinecone tools to agent tools")
                
            else:
                print("❌ No tools retrieved from Pinecone")
                selected_tools = []
                tool_names = []
                
        else:
            # Baseline: Use ALL tools
            selected_tools = self.all_tools.copy()
            tool_names = [tool.name for tool in selected_tools]
            print(f"📊 Baseline: Using ALL {len(selected_tools)} tools")
        
        return selected_tools, tool_names
    
    @traceable(name="DevOps_ToolLoadout_Agent")
    def run_toolloadout_agent(self, query: str, selected_tools, tool_names):
        """Tool Loadout execution with custom trace name"""
        agent_executor = initialize_agent(
            tools=selected_tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose,
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        return agent_executor.invoke({"input": query})
    
    @traceable(name="DevOps_Baseline_Agent")
    def run_baseline_agent(self, query: str, selected_tools, tool_names):
        """Baseline execution with custom trace name"""
        agent_executor = initialize_agent(
            tools=selected_tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose,
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        return agent_executor.invoke({"input": query})
    
    def run_agent(self, query: str, mode: str = "toolloadout"):
        """Simple agent execution"""
        print(f"\n🤖 Running {mode.upper()} mode")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Select tools intelligently
            use_tool_loadout = (mode == "toolloadout")
            selected_tools, tool_names = self.select_tools(query, use_tool_loadout)
            
            # Execute with appropriate wrapper for custom trace names
            if use_tool_loadout:
                result = self.run_toolloadout_agent(query, selected_tools, tool_names)
                run_name = f"DevOps-ToolLoadout-{len(selected_tools)}tools"
            else:
                result = self.run_baseline_agent(query, selected_tools, tool_names)
                run_name = f"DevOps-Baseline-{len(selected_tools)}tools"
            
            execution_time = time.time() - start_time
            
            # Print results
            print(f"\n📝 RESULTS")
            print("-" * 40)
            print(f"📝 Query: {query}")
            print(f"🎯 Mode: {mode.upper()}")
            print(f"🔧 Tools Used: {len(selected_tools)}")
            print(f"⏱️ Execution Time: {execution_time:.2f}s")
            print(f"📊 Success: {'Yes' if result.get('output') else 'No'}")
            
            # Show reasoning steps
            if result.get("intermediate_steps"):
                print(f"\n🧠 Reasoning Steps ({len(result['intermediate_steps'])}):")
                for i, (action, observation) in enumerate(result["intermediate_steps"], 1):
                    print(f"\nStep {i}:")
                    print(f"  Action: {getattr(action, 'tool', str(action))}")
                    print(f"  Input: {getattr(action, 'tool_input', {})}")
                    print(f"  Result: {str(observation)[:200]}...")
            
            print(f"\n✅ Final Answer:")
            print(result.get("output", "No output"))
            print("-" * 80)
            
            # Show LangSmith trace info
            print(f"\n🔗 LangSmith Trace: {run_name}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"\n❌ ERROR")
            print("-" * 40)
            print(f"Error: {str(e)}")
            print(f"Execution Time: {execution_time:.2f}s")
            print("-" * 80)
            return {"error": str(e)}
    
    def get_default_queries(self):
        """Get 10 default DevOps queries"""
        return [
            {
                "category": "EC2 & Infrastructure",
                "query": "My EC2 instance seems to be running slow, how can I check what's going on?",
                "description": "EC2 performance troubleshooting"
            },
            {
                "category": "Database Issues",
                "query": "Check the status of my database connections and identify any performance issues",
                "description": "Database connection and performance analysis"
            },
            {
                "category": "CI/CD Pipeline",
                "query": "Our CI/CD pipeline failed, help me diagnose the issue",
                "description": "CI/CD pipeline troubleshooting"
            },
            {
                "category": "Kubernetes",
                "query": "My Kubernetes pods are crashing, help me investigate why",
                "description": "Kubernetes pod failure investigation"
            },
            {
                "category": "Network Issues",
                "query": "Users are reporting slow network connectivity, help me check network status",
                "description": "Network connectivity troubleshooting"
            },
            {
                "category": "System Monitoring",
                "query": "Check overall system health and identify any resource bottlenecks",
                "description": "System health and resource monitoring"
            },
            {
                "category": "Storage Issues",
                "query": "My server is running out of disk space, help me analyze storage usage",
                "description": "Storage and disk usage analysis"
            },
            {
                "category": "Load Balancing",
                "query": "Traffic distribution seems uneven, check load balancer configuration",
                "description": "Load balancer health and configuration"
            },
            {
                "category": "Service Health",
                "query": "Check if all critical services are running properly",
                "description": "Service status and health monitoring"
            },
            {
                "category": "Performance Analysis",
                "query": "Application response time is high, help me identify the bottleneck",
                "description": "Application performance troubleshooting"
            }
        ]

def select_mode_interactive() -> str:
    """Interactive mode selection"""
    print("\n🤖 Agent Mode Selection:")
    print("-" * 40)
    print("1. Tool Loadout (Vector-selected relevant tools)")
    print("2. Baseline (All available tools)")
    
    while True:
        try:
            choice = input("\nSelect mode (1-2): ").strip()
            if choice == "1":
                return "toolloadout"
            elif choice == "2":
                return "baseline"
            else:
                print("Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

def main():
    """Simple interactive main function"""
    print("=" * 80)
    print("🤖 DevOps ReAct Agent Interactive Demo")
    print("=" * 80)
    
    try:
        # Create agent
        print("🚀 Initializing DevOps Agent...")
        agent = DevOpsReActAgent(verbose=True)
        
        # Select mode
        mode = select_mode_interactive()
        
        # Select query
        default_queries = agent.get_default_queries()
        
        print("\n📝 Query Selection:")
        print("-" * 60)
        for i, query_info in enumerate(default_queries, 1):
            print(f"{i:2d}. [{query_info['category']}] {query_info['description']}")
        print(f"{len(default_queries) + 1:2d}. Custom Query")
        
        while True:
            try:
                choice = input(f"\nSelect query (1-{len(default_queries) + 1}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(default_queries):
                    query = default_queries[choice_num - 1]["query"]
                    print(f"\n✅ Selected: {query[:80]}...")
                    break
                elif choice_num == len(default_queries) + 1:
                    query = input("Enter your custom query: ").strip()
                    if query:
                        break
                    else:
                        print("Please enter a valid query.")
                else:
                    print(f"Please enter a number between 1 and {len(default_queries) + 1}")
            except (ValueError, KeyboardInterrupt):
                print("\nExiting...")
                return
        
        # Confirm and execute
        print(f"\n📋 Ready to execute:")
        print(f"Mode: {mode.upper()}")
        print(f"Query: {query[:100]}...")
        
        confirm = input("\nProceed? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Cancelled.")
            return
        
        # Run agent
        result = agent.run_agent(query, mode)
        
        # Ask for another run
        another = input("\nRun another query? (y/N): ").strip().lower()
        if another in ['y', 'yes']:
            main()
        else:
            print("👋 Thanks for using DevOps ReAct Agent!")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")

if __name__ == "__main__":
    main()