import streamlit as st
import time
from typing import Dict, Any
import json
import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import our enhanced agent
try:
    # Import from the actual filename (with hyphens, Python converts to underscores)
    import importlib.util
    spec = importlib.util.spec_from_file_location("devops_agent", "devops-agent-toolloadout.py")
    devops_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(devops_agent)
    
    # Get the classes we need
    DevOpsReActAgent = devops_agent.DevOpsReActAgent
    AgentResult = devops_agent.AgentResult
    
except ImportError as e:
    st.error(f"❌ Could not import devops-agent-toolloadout.py: {e}")
    st.error("Make sure devops-agent-toolloadout.py is in the same directory as this Streamlit app.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading agent: {e}")
    st.error("Check that DevOpsReActAgent and AgentResult classes exist in devops-agent-toolloadout.py")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="DevOps ReAct Agent Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_agent():
    """Cache the agent instance"""
    return DevOpsReActAgent(verbose=False)

def format_reasoning_steps(steps):
    """Format reasoning steps for display"""
    formatted_steps = []
    for step in steps:
        formatted_steps.append(f"""
**Step {step.step_number}** ({step.timestamp})
- **Thought**: {step.thought if step.thought else 'Analyzing...'}
- **Action**: {step.action}
- **Input**: {json.dumps(step.action_input, indent=2) if step.action_input else 'None'}
- **Observation**: {step.observation[:500]}{'...' if len(step.observation) > 500 else ''}
""")
    return formatted_steps

def format_tool_selection(tool_selection):
    """Format tool selection info"""
    return f"""
**Selection Method**: {tool_selection.selection_method}
**Tools Selected**: {len(tool_selection.selected_tools)} out of {tool_selection.total_available}
**Selection Time**: {tool_selection.selection_time:.2f}s

**Selected Tools**:
{', '.join(tool_selection.selected_tools)}
"""

def display_execution_stats(metadata):
    """Display execution statistics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Steps", metadata.total_steps)
    with col2:
        st.metric("Execution Time", f"{metadata.execution_time:.2f}s")
    with col3:
        st.metric("Success", "✅" if metadata.success else "❌")
    with col4:
        if metadata.error_message:
            st.error(f"Error: {metadata.error_message}")

def main():
    # Title
    st.title("🤖 DevOps ReAct Agent Demo")
    st.markdown("Compare **Tool Loadout** vs **Baseline** performance")
    
    # Initialize agent
    agent = get_agent()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Agent mode selection
        mode = st.radio(
            "Select Agent Mode:",
            ["Tool Loadout", "Baseline", "Both (Comparison)"],
            help="Tool Loadout uses vector search to select relevant tools. Baseline uses all available tools."
        )
        
        # Tool limit for Tool Loadout
        tool_limit = st.slider(
            "Tool Limit (Tool Loadout only)",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of tools to select in Tool Loadout mode"
        )
        
        st.divider()
        
        # Default queries
        st.header("📝 Default Queries")
        default_queries = agent.get_default_queries()
        
        selected_default = st.selectbox(
            "Choose a default query:",
            options=["Custom Query"] + [f"{q['category']}: {q['description']}" for q in default_queries],
            help="Select a pre-built DevOps scenario or use custom query"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Query Input")
        
        # Query input
        if selected_default == "Custom Query":
            query = st.text_area(
                "Enter your DevOps query:",
                height=100,
                placeholder="e.g., My EC2 instance is running slow, help me troubleshoot..."
            )
        else:
            # Get selected default query
            query_index = int(selected_default.split(":")[0]) - 1 if selected_default != "Custom Query" else 0
            if query_index >= 0:
                default_query = default_queries[query_index]["query"]
                query = st.text_area(
                    "Query (from default):",
                    value=default_query,
                    height=100
                )
            else:
                query = st.text_area("Enter your DevOps query:", height=100)
    
    with col2:
        st.header("🎯 Execution")
        
        # Execute button
        execute_button = st.button(
            "🚀 Execute Query",
            type="primary",
            disabled=not query.strip(),
            help="Run the DevOps ReAct Agent"
        )
        
        if execute_button and query.strip():
            with st.spinner("🤖 Agent is thinking..."):
                
                if mode == "Both (Comparison)":
                    # Run comparison
                    results = agent.run_comparison(query, limit=tool_limit)
                    
                    # Store results in session state
                    st.session_state['comparison_results'] = results
                    st.session_state['single_result'] = None
                    
                else:
                    # Run single mode
                    use_tool_loadout = (mode == "Tool Loadout")
                    result = agent.execute_query(query, use_tool_loadout=use_tool_loadout, limit=tool_limit)
                    
                    # Store result in session state
                    st.session_state['single_result'] = result
                    st.session_state['comparison_results'] = None
    
    # Display results
    if 'single_result' in st.session_state and st.session_state['single_result']:
        result = st.session_state['single_result']
        
        # Execution stats
        st.header("📊 Execution Statistics")
        display_execution_stats(result.execution_metadata)
        
        # Two-column layout for results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("🧠 ReAct Reasoning Process")
            
            if result.reasoning_steps:
                steps = format_reasoning_steps(result.reasoning_steps)
                for step in steps:
                    with st.expander(f"Step {steps.index(step) + 1}", expanded=True):
                        st.markdown(step)
            else:
                st.info("No reasoning steps available")
        
        with col2:
            st.header("🔧 Tool Selection & Result")
            
            # Tool selection info
            st.subheader("Tool Selection")
            tool_info = format_tool_selection(result.tool_selection)
            st.text(tool_info)
            
            # Final answer
            st.subheader("Final Answer")
            st.success(result.final_answer)
    
    # Display comparison results
    elif 'comparison_results' in st.session_state and st.session_state['comparison_results']:
        results = st.session_state['comparison_results']
        
        st.header("⚖️ Comparison Results")
        
        # Performance comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Tool Loadout")
            loadout_result = results['tool_loadout']
            display_execution_stats(loadout_result.execution_metadata)
            
            # Tool selection
            st.text("Tool Selection:")
            st.text(format_tool_selection(loadout_result.tool_selection))
            
            # Final answer
            st.text("Final Answer:")
            st.success(loadout_result.final_answer)
        
        with col2:
            st.subheader("📊 Baseline")
            baseline_result = results['baseline']
            display_execution_stats(baseline_result.execution_metadata)
            
            # Tool selection
            st.text("Tool Selection:")
            st.text(format_tool_selection(baseline_result.tool_selection))
            
            # Final answer
            st.text("Final Answer:")
            st.success(baseline_result.final_answer)
        
        # Performance comparison summary
        st.header("📈 Performance Summary")
        
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            time_diff = baseline_result.execution_metadata.execution_time - loadout_result.execution_metadata.execution_time
            st.metric(
                "Time Difference", 
                f"{time_diff:.2f}s",
                delta=f"Baseline {'faster' if time_diff < 0 else 'slower'}"
            )
        
        with perf_col2:
            tool_reduction = len(baseline_result.tool_selection.selected_tools) - len(loadout_result.tool_selection.selected_tools)
            st.metric(
                "Tool Reduction",
                f"{tool_reduction} tools",
                delta=f"{(tool_reduction/len(baseline_result.tool_selection.selected_tools)*100):.1f}% reduction"
            )
        
        with perf_col3:
            step_diff = baseline_result.execution_metadata.total_steps - loadout_result.execution_metadata.total_steps
            st.metric(
                "Reasoning Steps",
                f"{step_diff} difference",
                delta="Same logic" if step_diff == 0 else f"{'More' if step_diff > 0 else 'Fewer'} steps"
            )
        
        # Detailed reasoning comparison
        if st.checkbox("Show Detailed Reasoning Comparison"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Tool Loadout Reasoning")
                if loadout_result.reasoning_steps:
                    steps = format_reasoning_steps(loadout_result.reasoning_steps)
                    for i, step in enumerate(steps):
                        with st.expander(f"Step {i + 1}"):
                            st.markdown(step)
                else:
                    st.info("No reasoning steps")
            
            with col2:
                st.subheader("📊 Baseline Reasoning")
                if baseline_result.reasoning_steps:
                    steps = format_reasoning_steps(baseline_result.reasoning_steps)
                    for i, step in enumerate(steps):
                        with st.expander(f"Step {i + 1}"):
                            st.markdown(step)
                else:
                    st.info("No reasoning steps")
    
    # Footer
    st.divider()
    st.markdown("""
    ### About This Demo
    This demo showcases a **DevOps ReAct Agent** that can automatically troubleshoot infrastructure issues using:
    - **Tool Loadout**: Vector search to select relevant tools
    - **Baseline**: Using all available tools
    - **ReAct Reasoning**: Thought → Action → Observation cycles
    
    The agent has access to 50+ DevOps tools for EC2, Kubernetes, databases, monitoring, and more.
    """)

if __name__ == "__main__":
    main()