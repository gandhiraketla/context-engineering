# Financial Advisor - Context Quarantine Demo
# Clean implementation showcasing isolated agent contexts

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
import yfinance as yf
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangSmith Setup for Context Quarantine Tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ContextEngineering-Context Quarantine"

# Note: Individual agent tracing will be handled automatically by LangGraph
# The project name set above will apply to all traces

# Initialize the language model
llm = init_chat_model("openai:deepseek-chat", temperature=0, base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY"))

# Mock portfolio data (10 stocks)
MOCK_PORTFOLIO = {
    "AAPL": {"shares": 100, "avg_cost": 150.00},
    "NVDA": {"shares": 50, "avg_cost": 200.00},
    "MSFT": {"shares": 75, "avg_cost": 280.00},
    "GOOGL": {"shares": 30, "avg_cost": 120.00},
    "AMZN": {"shares": 25, "avg_cost": 140.00},
    "TSLA": {"shares": 40, "avg_cost": 180.00},
    "META": {"shares": 60, "avg_cost": 220.00},
    "AMD": {"shares": 80, "avg_cost": 90.00},
    "NFLX": {"shares": 20, "avg_cost": 380.00},
    "CRM": {"shares": 45, "avg_cost": 190.00}
}

# STOCK AGENT TOOLS
def get_stock_price(symbol: str) -> str:
    """Get current stock price and basic metrics"""
    print(f"📈 STOCK TOOL: Fetching data for {symbol}")
    
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1d")
    info = ticker.info
    
    current_price = hist['Close'].iloc[-1]
    change_percent = ((current_price - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1]) * 100
    
    return f"""Stock Data for {symbol}:
Current Price: ${current_price:.2f}
Daily Change: {change_percent:.2f}%
Volume: {hist['Volume'].iloc[-1]:,}
Market Cap: ${info.get('marketCap', 0):,}"""

# NEWS AGENT TOOLS  
def get_financial_news(query: str) -> str:
    """Get financial news summary using Perplexity API"""
    print(f"📰 NEWS TOOL: Researching {query}")
    
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return "Error: PERPLEXITY_API_KEY not found"
    
    try:
        # Initialize Perplexity client using OpenAI interface
        perplexity_client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        
        response = perplexity_client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": "Provide detailed financial news and market analysis. Focus on recent developments, earnings reports, analyst opinions, and market sentiment. Include key metrics and concrete data when available."},
                {"role": "user", "content": f"Financial news and analysis for: {query}"}
            ],
            max_tokens=1500,
            temperature=0.1
        )
        
        return response.choices[0].message.content or "No financial news found"
    except Exception as e:
        return f"Error fetching financial news: {str(e)}"

# PORTFOLIO AGENT TOOLS
def get_all_portfolio_holdings() -> str:
    """Get complete list of all portfolio holdings"""
    print(f"💼 PORTFOLIO TOOL: Getting all portfolio holdings")
    
    holdings_list = []
    total_investment = 0
    
    for symbol, holding in MOCK_PORTFOLIO.items():
        shares = holding["shares"]
        avg_cost = holding["avg_cost"]
        total_value = shares * avg_cost
        total_investment += total_value
        
        holdings_list.append(f"{symbol}: {shares} shares @ ${avg_cost:.2f} = ${total_value:,.2f}")
    
    holdings_summary = "\n".join(holdings_list)
    
    return f"""Complete Portfolio Holdings:
{holdings_summary}

Total Portfolio Investment: ${total_investment:,.2f}
Holdings Count: {len(MOCK_PORTFOLIO)} stocks"""

def check_portfolio_holdings(symbol: str) -> str:
    """Check if symbol exists in portfolio and return holding details"""
    print(f"💼 PORTFOLIO TOOL: Checking holdings for {symbol}")
    
    symbol = symbol.upper()
    if symbol in MOCK_PORTFOLIO:
        holding = MOCK_PORTFOLIO[symbol]
        total_value = holding["shares"] * holding["avg_cost"]
        return f"""Portfolio Holdings for {symbol}:
Shares Owned: {holding['shares']}
Average Cost: ${holding['avg_cost']:.2f}
Total Investment: ${total_value:,.2f}"""
    else:
        return f"No holdings found for {symbol} in your portfolio."

def calculate_portfolio_impact(symbol: str, current_price: float) -> str:
    """Calculate profit/loss impact for portfolio holding"""
    print(f"💼 PORTFOLIO TOOL: Calculating impact for {symbol}")
    
    symbol = symbol.upper()
    if symbol not in MOCK_PORTFOLIO:
        return f"Cannot calculate impact - no {symbol} holdings in portfolio."
    
    holding = MOCK_PORTFOLIO[symbol]
    shares = holding["shares"]
    avg_cost = holding["avg_cost"]
    
    current_value = shares * current_price
    original_value = shares * avg_cost
    profit_loss = current_value - original_value
    profit_loss_percent = (profit_loss / original_value) * 100
    
    return f"""Portfolio Impact for {symbol}:
Current Position Value: ${current_value:,.2f}
Original Investment: ${original_value:,.2f}
Profit/Loss: ${profit_loss:,.2f} ({profit_loss_percent:.2f}%)
Impact: {'Positive' if profit_loss > 0 else 'Negative'} contribution to portfolio"""

def analyze_portfolio_comprehensive(analysis_request: str) -> str:
    """Perform comprehensive portfolio analysis"""
    print(f"💼 PORTFOLIO TOOL: Performing comprehensive analysis for: {analysis_request}")
    
    # Get all portfolio holdings first
    holdings_result = get_all_portfolio_holdings()
    
    # Extract stock symbols from portfolio
    portfolio_symbols = list(MOCK_PORTFOLIO.keys())
    
    analysis_results = []
    analysis_results.append("=== COMPREHENSIVE PORTFOLIO ANALYSIS ===")
    analysis_results.append("")
    analysis_results.append(f"Analysis Request: {analysis_request}")
    analysis_results.append("")
    analysis_results.append(holdings_result)
    analysis_results.append("")
    
    # For each stock in portfolio, gather data and analyze
    for symbol in portfolio_symbols:
        analysis_results.append(f"--- {symbol} Analysis ---")
        
        # Get current stock data
        try:
            stock_data = get_stock_price(symbol)
            analysis_results.append("Stock Data:")
            analysis_results.append(stock_data)
        except Exception as e:
            analysis_results.append(f"Stock Data: Error fetching data for {symbol}: {str(e)}")
        
        # Get news analysis related to the request
        try:
            news_query = f"{analysis_request} impact on {symbol}"
            news_data = get_financial_news(news_query)
            analysis_results.append("News Analysis:")
            analysis_results.append(news_data)
        except Exception as e:
            analysis_results.append(f"News Analysis: Error fetching news for {symbol}: {str(e)}")
        
        # Show portfolio position
        if symbol in MOCK_PORTFOLIO:
            holding = MOCK_PORTFOLIO[symbol]
            shares = holding["shares"]
            avg_cost = holding["avg_cost"]
            analysis_results.append("Portfolio Position:")
            analysis_results.append(f"Shares: {shares}, Avg Cost: ${avg_cost:.2f}")
        
        analysis_results.append("-" * 50)
    
    return "\n".join(analysis_results)

# CREATE SPECIALIZED AGENTS

# Stock Agent - Only handles stock data
stock_agent = create_react_agent(
    model=llm,
    tools=[get_stock_price],
    name="stock_expert",
    prompt="""You are a stock market data specialist with access to real-time stock information.

Your responsibilities:
- Fetch current stock prices and basic financial metrics for any requested stock
- Provide stock-specific data using your tools
- Focus exclusively on stock market data retrieval

Your role in analysis:
- When asked about a stock in any context (tariffs, earnings, etc.), provide the current stock data
- You don't need to analyze the impact - just provide the current metrics
- Let other experts handle analysis while you provide the data foundation

Constraints:
- Always use your stock price tool when any stock symbol is mentioned
- Keep responses focused on current stock data and metrics
- Do not attempt news research or portfolio calculations"""
)

# News Agent - Only handles financial news
news_agent = create_react_agent(
    model=llm,
    tools=[get_financial_news],
    name="news_expert",
    prompt="""You are a financial news research specialist with access to market news and analysis.

Your responsibilities:
- Research financial news and market sentiment
- Provide news summaries and market analysis
- Focus exclusively on information gathering and news research

Constraints:
- Do NOT fetch stock prices or financial data
- Do NOT access portfolio information or perform calculations
- Always use your news research tool for any news-related queries
- Keep responses focused on news and market sentiment"""
)

# Portfolio Agent - Handles portfolio data and comprehensive analysis
portfolio_agent = create_react_agent(
    model=llm,
    tools=[get_all_portfolio_holdings, check_portfolio_holdings, calculate_portfolio_impact, analyze_portfolio_comprehensive],
    name="portfolio_expert", 
    prompt="""You are a comprehensive portfolio management specialist with access to portfolio holdings and analysis capabilities.

Your responsibilities:
- Check individual portfolio holdings and positions
- Get complete portfolio listings when requested
- Calculate profit/loss and portfolio impact for individual stocks
- Perform comprehensive portfolio-wide analysis when requested

For comprehensive portfolio analysis queries (like "how will X impact my portfolio"):
- Use the analyze_portfolio_comprehensive tool which will:
  * Get all portfolio holdings
  * Gather current stock data for each holding
  * Research relevant market news for each stock
  * Calculate impact on each position
  * Provide detailed stock-by-stock analysis

For simple portfolio queries:
- Use basic portfolio tools for quick lookups

Constraints:
- Work primarily with portfolio data and tools
- For comprehensive analysis, let your tools handle the coordination
- Keep responses focused on portfolio analysis and holdings"""
)

# SUPERVISOR AGENT
supervisor_prompt = """You are a Financial Advisor supervisor managing three specialized experts:

Available Specialists:
- stock_expert: Fetches real-time stock prices and financial data for any stock
- news_expert: Researches financial news and market sentiment
- portfolio_expert: Analyzes portfolio holdings and performs comprehensive portfolio analysis

Smart Routing Rules:

SINGLE STOCK ANALYSIS (Use multiple experts):
- "How do [events] impact [STOCK]?" → news_expert (research impact) + stock_expert (get current data)
- "What's happening with [STOCK]?" → news_expert + stock_expert
- "[STOCK] analysis" → news_expert + stock_expert

SIMPLE QUERIES (Direct routing):
- "[STOCK] price?" → stock_expert only
- "News about [TOPIC]?" → news_expert only  
- "Do I own [STOCK]?" → portfolio_expert only

PORTFOLIO ANALYSIS (Route to portfolio_expert):
- "How will [event] impact my portfolio?" → portfolio_expert (comprehensive)
- "My portfolio exposure to [sector]?" → portfolio_expert

Example Routing:
"What's AMZN's current price?" → stock_expert
"Tesla news?" → news_expert  
"Do I own Tesla?" → portfolio_expert
"How do tariffs impact Amazon stock?" → news_expert (tariff research) + stock_expert (Amazon data)
"How will tariffs impact MY portfolio?" → portfolio_expert (comprehensive analysis)

Instructions for Multi-Expert Queries:
- Call news_expert first for market analysis/research
- Call stock_expert second for current stock data
- Combine both responses into comprehensive answer
- Make it clear you're coordinating multiple experts

Your Role:
- Route queries to appropriate expert(s) based on what information is needed
- For single stock analysis, use both news and stock experts
- For portfolio questions, use portfolio expert
- Always combine responses when using multiple experts
- Never perform analysis yourself - coordinate experts"""

# Create the Financial Advisor supervisor
print("🧑‍💼 INITIALIZING FINANCIAL ADVISOR SYSTEM...")
print("📈 Stock Expert: Ready")
print("📰 News Expert: Ready") 
print("💼 Portfolio Expert: Ready")
print("🧑‍💼 Supervisor: Ready")
print(f"📊 LangSmith Project: {os.environ['LANGCHAIN_PROJECT']}")
print("\n" + "="*50)

financial_advisor = create_supervisor(
    [stock_agent, news_agent, portfolio_agent],
    model=llm,
    prompt=supervisor_prompt
)

# Compile the application
app = financial_advisor.compile()

# Demo function to test the system
def demo_financial_advisor():
    """Run demo queries to showcase context quarantine with LangSmith tracking"""
    
    test_queries = [
        "What's NVDA's current stock price?",
        "Any recent news about NVDA?", 
        "Do I own any NVDA shares?",
        "How is NVDA doing and what's the impact on my portfolio?"
    ]
    
    print("🚀 RUNNING FINANCIAL ADVISOR DEMOS WITH LANGSMITH TRACKING\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"🔍 DEMO {i}: {query}")
        print("-" * 60)
        
        # Run the query through the supervisor with tracking
        result = app.invoke({"messages": [("user", query)]})
        
        # Extract the final response
        final_response = result["messages"][-1].content
        print(f"💡 RESPONSE: {final_response}")
        print(f"📊 Check LangSmith for individual agent traces")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    print("🚀 FINANCIAL ADVISOR - CONTEXT QUARANTINE DEMO")
    print("=" * 60)
    print("This demo showcases how each agent works in isolation:")
    print("📈 Stock Expert - Only sees stock queries")
    print("📰 News Expert - Only sees news queries") 
    print("💼 Portfolio Expert - Only sees portfolio queries")
    print("🧑‍💼 Supervisor - Routes queries and combines responses")
    print("=" * 60)
    print("\nSample queries to try:")
    print("• 'What's NVDA's current price?' → Stock Agent")
    print("• 'Any news about Tesla?' → News Agent")
    print("• 'Do I own Apple stock?' → Portfolio Agent")
    print("• 'How will tariffs impact Amazon?' → Stock + News Agents")
    print("• 'How will tariffs impact my portfolio?' → Portfolio Agent (comprehensive)")
    print("=" * 60)
    
    while True:
        query = input("\n💬 Ask your financial question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            print("\n👋 Thanks for using Financial Advisor Demo!")
            break
        
        print(f"\n🔍 PROCESSING: {query}")
        print("📊 Check LangSmith for individual agent traces...")
        print("-" * 50)
        
        try:
            result = app.invoke({"messages": [("user", query)]})
            print(f"\n💡 RESPONSE:")
            print(result['messages'][-1].content)
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
        
        print("\n" + "=" * 60)