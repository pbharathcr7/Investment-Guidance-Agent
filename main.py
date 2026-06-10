import os
from typing import TypedDict, Annotated, List, Dict
import operator
from datetime import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
import yfinance as yf
import requests
import json
import streamlit as st

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    SERP_API_KEY = os.getenv("SERP_API_KEY")
    Model_Name = os.getenv("Model_Name")
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    SERP_API_KEY = os.getenv("SERP_API_KEY")
    Model_Name = os.getenv("Model_Name")

# ============================================================================
# STATE — 5 fields only.
# Intermediate analysis data lives inside the agent's own message history,
# NOT in graph state — no token waste passing large strings between nodes.
# ============================================================================

class InvestmentAgentState(TypedDict):
    user_query: str
    user_profile: Dict[str, any]
    query_type: str
    final_response: str
    messages: Annotated[List[str], operator.add]


# ============================================================================
# TOOLS — all data access is @tool so agents decide when to call them
# ============================================================================

@tool("fetch_stock_data")
def fetch_stock_data(symbol: str) -> str:
    """Fetch real-time price, fundamentals, and 52-week range for a stock ticker.
    Use standard ticker symbols: AAPL, RELIANCE.NS, TCS.NS, HDFCBANK.NS, etc.
    For Indian stocks append .NS (NSE) or .BO (BSE). US stocks need no suffix."""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")

        # Auto-retry with common suffixes if symbol not found
        if hist.empty:
            base = symbol.split(".")[0]
            for suffix in [".NS", ".BO", ".L", ""]:
                alt = base + suffix if suffix else base
                if alt == symbol:
                    continue
                stock = yf.Ticker(alt)
                hist = stock.history(period="1y")
                if not hist.empty:
                    symbol = alt
                    break

        if hist.empty:
            return json.dumps({"error": f"No market data found for '{symbol}'. Check the ticker symbol."})

        info = stock.info
        current_price = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
        change_pct = ((current_price - prev_close) / prev_close) * 100

        market_cap = info.get("marketCap", "N/A")
        if isinstance(market_cap, (int, float)):
            if market_cap >= 1e12:
                market_cap = f"{market_cap / 1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap = f"{market_cap / 1e9:.2f}B"
            else:
                market_cap = f"{market_cap / 1e6:.2f}M"

        rev_growth = info.get("revenueGrowth", "N/A")
        if isinstance(rev_growth, float):
            rev_growth = f"{rev_growth * 100:.1f}%"

        profit_margin = info.get("profitMargins", "N/A")
        if isinstance(profit_margin, float):
            profit_margin = f"{profit_margin * 100:.1f}%"

        div_yield = info.get("dividendYield", "N/A")
        if isinstance(div_yield, float):
            div_yield = f"{div_yield * 100:.2f}%"

        return json.dumps({
            "symbol": symbol,
            "company_name": info.get("longName", info.get("shortName", symbol)),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "country": info.get("country", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "price": round(current_price, 2),
            "change_percent": f"{change_pct:.2f}%",
            "52_week_high": round(float(hist["High"].max()), 2),
            "52_week_low": round(float(hist["Low"].min()), 2),
            "market_cap": market_cap,
            "pe_ratio": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "dividend_yield": div_yield,
            "revenue_growth": rev_growth,
            "profit_margin": profit_margin,
            "description": (info.get("longBusinessSummary", "") or "")[:300],
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("fetch_stock_news")
def fetch_stock_news(query: str) -> str:
    """Fetch the latest news headlines and snippets for a stock or company.
    Pass ticker + company name as query, e.g. 'AAPL Apple' or 'TCS.NS Tata Consultancy'."""
    try:
        params = {
            "q": f"{query} stock news",
            "api_key": SERP_API_KEY,
            "engine": "google",
            "tbm": "nws",
            "num": 5,
        }
        data = requests.get("https://serpapi.com/search", params=params, timeout=10).json()
        articles = [
            f"- {a.get('title', '')} [{a.get('source', '')}, {a.get('date', '')}]: {a.get('snippet', '')}"
            for a in data.get("news_results", [])[:5]
        ]
        return "\n".join(articles) if articles else "No recent news found."
    except Exception as e:
        return f"News fetch error: {str(e)}"


@tool("search_web")
def search_web(query: str) -> str:
    """Search the web for market context, sector trends, commodity prices, macro news, etc."""
    try:
        params = {
            "q": query,
            "api_key": SERP_API_KEY,
            "engine": "google",
            "num": 5,
        }
        data = requests.get("https://serpapi.com/search", params=params, timeout=10).json()
        results = [
            f"[{r.get('title', '')}]\n{r.get('snippet', '')}"
            for r in data.get("organic_results", [])[:5]
        ]
        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"


@tool("get_market_indices")
def get_market_indices() -> str:
    """Fetch current values and daily % change for S&P 500, NIFTY 50, Dow Jones, and NASDAQ."""
    indices = {
        "S&P 500": "^GSPC",
        "NIFTY 50": "^NSEI",
        "Dow Jones": "^DJI",
        "NASDAQ": "^IXIC",
    }
    lines = []
    for name, sym in indices.items():
        try:
            hist = yf.Ticker(sym).history(period="5d")
            if not hist.empty:
                cur = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else cur
                pct = ((cur - prev) / prev) * 100
                lines.append(f"{name}: {cur:,.2f} ({pct:+.2f}%)")
            else:
                lines.append(f"{name}: N/A")
        except Exception:
            lines.append(f"{name}: N/A")
    return "\n".join(lines)


@tool("calculate_what_if")
def calculate_what_if(symbol: str, months_ago: int, investment_amount: float) -> str:
    """Calculate hypothetical returns if someone had invested a fixed amount X months ago.
    Returns purchase price, shares bought, current value, profit/loss, and return %."""
    try:
        hist = yf.Ticker(symbol).history(period=f"{months_ago + 1}mo")
        if len(hist) < 2:
            return "Insufficient historical data for this period."
        old_price = float(hist["Close"].iloc[0])
        cur_price = float(hist["Close"].iloc[-1])
        shares = investment_amount / old_price
        current_value = shares * cur_price
        profit = current_value - investment_amount
        return_pct = (profit / investment_amount) * 100
        return (
            f"If you had invested {investment_amount:,.0f} in {symbol} {months_ago} months ago:\n"
            f"  Purchase price : {old_price:.2f}\n"
            f"  Shares bought  : {shares:.4f}\n"
            f"  Current value  : {current_value:,.2f}\n"
            f"  Profit / Loss  : {profit:+,.2f} ({return_pct:+.1f}%)"
        )
    except Exception as e:
        return f"Calculation error: {str(e)}"


# ============================================================================
# SHARED REACT LOOP
# One shared helper — LLM decides which tools to call, executes them,
# feeds results back, repeats until LLM returns a final plain-text answer.
# ============================================================================

def _run_react_loop(
    llm_with_tools,
    messages: list,
    tool_registry: dict,
    max_iterations: int = 5,
) -> str:
    response = None
    for _ in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            # LLM returned a final answer — no more tool calls needed
            break

        for call in response.tool_calls:
            tool_name = call.get("name", "") or "unknown_tool"
            tool_args = call.get("args", {}) or {}
            # Gemini requires tool_call_id to be non-empty
            tool_call_id = call.get("id") or tool_name
            tool_fn = tool_registry.get(tool_name)

            if tool_fn is None:
                result = f"Tool '{tool_name}' is not available."
            else:
                try:
                    result = tool_fn.invoke(tool_args)
                    print(f"  [TOOL] {tool_name}({list(tool_args.keys())})")
                except Exception as e:
                    result = f"Tool error from {tool_name}: {str(e)}"

            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call_id, name=tool_name)
            )

    if response is None:
        return "No response generated."
    return response.content if hasattr(response, "content") else str(response)


# ============================================================================
# NODES
# ============================================================================

def classify_query_node(state: InvestmentAgentState) -> Dict:
    """One lightweight LLM call — outputs a single routing word, nothing more."""
    print("[NODE] classify_query")
    llm = ChatGoogleGenerativeAI(
        model=Model_Name, google_api_key=GEMINI_API_KEY, temperature=0
    )
    response = llm.invoke(
        f'Classify this query. Reply with ONLY ONE WORD: single_stock OR general_advice OR off_topic\n\n'
        f'Rules:\n'
        f'- single_stock   : asking about ONE specific company/stock (e.g. "Should I buy Apple?", "Analyze TCS")\n'
        f'- general_advice : general investment questions, markets, sectors, portfolios\n'
        f'- off_topic      : not related to investments or finance\n\n'
        f'Query: "{state["user_query"]}"'
    )
    raw = response.content.strip().lower()
    if "single_stock" in raw:
        query_type = "single_stock"
    elif "off_topic" in raw:
        query_type = "off_topic"
    else:
        query_type = "general_advice"

    print(f"  → {query_type}")
    return {"query_type": query_type, "messages": [f"Classified: {query_type}"]}


def handle_off_topic_node(state: InvestmentAgentState) -> Dict:
    """No LLM call — static redirect. Zero token cost."""
    print("[NODE] handle_off_topic")
    return {
        "final_response": (
            "I'm an Investment Advisor AI — I can only help with stock analysis and financial guidance.\n\n"
            "Try asking:\n"
            '• "Should I buy Apple stock?"\n'
            '• "What sectors are performing well right now?"\n'
            '• "I have $5000, where should I invest?"\n'
            '• "Analyze TCS stock for me"'
        ),
        "messages": ["Off-topic handled"],
    }


def general_advisor_agent(state: InvestmentAgentState) -> Dict:
    """ReAct agent with optional tools. LLM decides whether to search web or fetch indices."""
    print("[NODE] general_advisor_agent")
    llm = ChatGoogleGenerativeAI(
        model=Model_Name, google_api_key=GEMINI_API_KEY, temperature=0.2
    )
    llm_with_tools = llm.bind_tools([search_web, get_market_indices])

    today = datetime.now().strftime("%B %Y")

    system = (
        f"You are a personal investment advisor. Consider ALL investment options — stocks, gold, bonds, ETFs, FDs, etc.\n"
        f"Use tools only when you need current data to answer accurately. You may call none, one, or multiple tools.\n"
        f"Today: {today}\n\n"
        f"After gathering data (if needed), write a clear, structured, actionable response tailored to the user's profile.\n"
        f"End with a short disclaimer that this is not financial advice."
    )

    human = (
        f"USER PROFILE:\n"
        f"- Risk Tolerance : {state['user_profile'].get('risk_tolerance', 'N/A')}\n"
        f"- Horizon        : {state['user_profile'].get('investment_horizon', 'N/A')}\n"
        f"- Budget         : {state['user_profile'].get('budget_currency', '$')}"
        f"{state['user_profile'].get('budget', 'N/A')}\n"
        f"- Goals          : {state['user_profile'].get('investment_goals', 'N/A')}\n\n"
        f"QUESTION: {state['user_query']}"
    )

    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    tool_registry = {"search_web": search_web, "get_market_indices": get_market_indices}

    final_advice = _run_react_loop(llm_with_tools, messages, tool_registry)

    return {
        "final_response": final_advice,
        "messages": ["General advice generated"],
    }


def stock_analysis_agent(state: InvestmentAgentState) -> Dict:
    """ReAct agent: fetches stock data + news via tools, then writes one complete report."""
    print("[NODE] stock_analysis_agent")
    llm = ChatGoogleGenerativeAI(
        model=Model_Name, google_api_key=GEMINI_API_KEY, temperature=0.2
    )
    llm_with_tools = llm.bind_tools(
        [fetch_stock_data, fetch_stock_news, calculate_what_if]
    )

    system = (
        f"You are an expert stock analyst. Follow these steps:\n\n"
        f"STEP 1 — Resolve the ticker symbol from the user query using your own knowledge.\n"
        f"         Indian stocks: append .NS (NSE) or .BO (BSE). US stocks: no suffix needed.\n\n"
        f"STEP 2 — Call fetch_stock_data with the resolved ticker symbol.\n\n"
        f"STEP 3 — Call fetch_stock_news with '<TICKER> <CompanyName>' as the query.\n\n"
        f"STEP 4 — Using data from the tools, write ONE complete investment report covering:\n"
        f"  • Current price, daily change, market cap, sector, P/E ratio\n"
        f"  • Business health: is the company growing and profitable?\n"
        f"  • Valuation: is the stock cheap or expensive? Use simple analogies.\n"
        f"  • Price momentum: where is it in the 52-week range? Trend direction?\n"
        f"  • News sentiment: positive, negative, or mixed?\n"
        f"  • 6-month price scenarios: bull case and bear case with % estimates\n"
        f"  • Risk match: does this stock suit THIS specific investor's profile?\n"
        f"  • Final verdict: BUY / WAIT / DON'T BUY — with clear reasons\n"
        f"  • 2–3 alternative stocks in the same sector (include ticker symbols)\n\n"
        f"USER PROFILE:\n"
        f"- Risk Tolerance : {state['user_profile'].get('risk_tolerance', 'N/A')}\n"
        f"- Horizon        : {state['user_profile'].get('investment_horizon', 'N/A')}\n"
        f"- Budget         : {state['user_profile'].get('budget_currency', '$')}{state['user_profile'].get('budget', 'N/A')}\n"
        f"- Goals          : {state['user_profile'].get('investment_goals', 'N/A')}\n\n"
        f"Write in plain English — beginner-friendly but data-backed. End with a disclaimer."
    )

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Analyze: {state['user_query']}"),
    ]
    tool_registry = {
        "fetch_stock_data": fetch_stock_data,
        "fetch_stock_news": fetch_stock_news,
        "calculate_what_if": calculate_what_if,
    }

    report = _run_react_loop(llm_with_tools, messages, tool_registry, max_iterations=5)

    return {
        "final_response": report,
        "messages": ["Stock analysis complete"],
    }


# ============================================================================
# GRAPH — 3 nodes, clean routing, no dead edges
# ============================================================================

def create_investment_agent():
    workflow = StateGraph(InvestmentAgentState)

    workflow.add_node("classify", classify_query_node)
    workflow.add_node("handle_off_topic", handle_off_topic_node)
    workflow.add_node("general_advisor", general_advisor_agent)
    workflow.add_node("stock_analysis", stock_analysis_agent)

    workflow.set_entry_point("classify")

    workflow.add_conditional_edges(
        "classify",
        lambda s: s["query_type"],
        {
            "single_stock": "stock_analysis",
            "general_advice": "general_advisor",
            "off_topic": "handle_off_topic",
        },
    )

    workflow.add_edge("handle_off_topic", END)
    workflow.add_edge("general_advisor", END)
    workflow.add_edge("stock_analysis", END)

    return workflow.compile()


# ============================================================================
# ENTRY POINT
# ============================================================================

def run_investment_agent(user_query: str, user_profile: Dict = None):
    if user_profile is None:
        user_profile = {
            "risk_tolerance": "moderate",
            "investment_horizon": "medium-term (1-3 years)",
            "budget": 5000,
            "budget_currency": "$",
            "investment_goals": "growth with moderate risk",
        }

    initial_state = {
        "user_query": user_query,
        "user_profile": user_profile,
        "query_type": "",
        "final_response": "",
        "messages": [],
    }

    print("Analyzing your question...")
    print("─" * 60)

    agent = create_investment_agent()
    final_state = agent.invoke(initial_state)

    print("\n" + final_state["final_response"])
    return final_state
