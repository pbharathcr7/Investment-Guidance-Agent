import os
from typing import TypedDict, Annotated, List, Dict
import operator
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
import yfinance as yf
import requests
import json

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
Model_Name = os.getenv("Model_Name")

# ============================================================================
# STATE DEFINITION
# ============================================================================

class InvestmentAgentState(TypedDict):
    user_query: str
    user_profile: Dict[str, any]
    query_type: str
    company_name: str
    stock_symbol: str
    exchange: str
    current_price: float
    stock_data: Dict[str, any]
    news_articles: List[Dict[str, str]]
    market_conditions: str
    business_analysis: str
    valuation_analysis: str
    momentum_analysis: str
    news_sentiment: str
    price_prediction: str
    risk_assessment: str
    recommendation: str
    general_advice: str
    final_response: str
    messages: Annotated[List[str], operator.add]

# ============================================================================
# TOOLS
# ============================================================================

def search_web(query: str) -> str:
    """Search the web using SerpAPI and return formatted results"""
    try:
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": SERP_API_KEY,
            "engine": "google",
            "num": 5
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        results = []
        # Get organic results
        for item in data.get("organic_results", [])[:5]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            })
        
        # Format for LLM
        formatted = "\n\n".join([
            f"Source: {r['title']}\n{r['snippet']}\nURL: {r['link']}"
            for r in results
        ])
        
        return formatted if formatted else "No results found"
    except Exception as e:
        return f"Search error: {str(e)}"


def resolve_stock_symbol(query: str, llm) -> Dict[str, str]:
    """Use LLM to extract company name and resolve to proper ticker symbol"""
    
    prompt = f"""You are a stock symbol resolver. Extract the company/stock from this query and find its ticker symbol.

    Query: "{query}"

    Respond in this EXACT JSON format:
    {{
        "company_name": "Full Company Name",
        "symbol": "TICKER",
        "exchange": "NSE" or "BSE" or "NYSE" or "NASDAQ" or "LSE"
    }}

    Examples:
    - "Reliance Industries" → {{"company_name": "Reliance Industries", "symbol": "RELIANCE.NS", "exchange": "NSE"}}
    - "TCS" → {{"company_name": "Tata Consultancy Services", "symbol": "TCS.NS", "exchange": "NSE"}}
    - "Apple" → {{"company_name": "Apple Inc.", "symbol": "AAPL", "exchange": "NASDAQ"}}
    - "HDFC Bank" → {{"company_name": "HDFC Bank", "symbol": "HDFCBANK.NS", "exchange": "NSE"}}

    For Indian stocks, use .NS suffix (NSE) or .BO suffix (BSE).
    For US stocks, no suffix needed.

    Respond ONLY with the JSON."""
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Clean up response if it has markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        
        return {
            "company_name": result.get("company_name", ""),
            "symbol": result.get("symbol", ""),
            "exchange": result.get("exchange", "")
        }
    except Exception as e:
        query_upper = query.upper().strip()
        return {
            "company_name": query,
            "symbol": query_upper,
            "exchange": "UNKNOWN"
        }


def fetch_global_stock_data(symbol: str) -> Dict[str, any]:
    """Fetch stock data using yfinance (supports global stocks)"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1y")
        
        if hist.empty:
            return {"error": f"No data found for {symbol}"}
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100
        
        high_52w = hist['High'].max()
        low_52w = hist['Low'].min()
        
        return {
            "symbol": symbol,
            "price": round(float(current_price), 2),
            "change": round(float(change), 2),
            "change_percent": f"{change_percent:.2f}%",
            "volume": info.get("volume", "N/A"),
            "company_name": info.get("longName", info.get("shortName", symbol)),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "52_week_high": round(float(high_52w), 2),
            "52_week_low": round(float(low_52w), 2),
            "profit_margin": info.get("profitMargins", "N/A"),
            "revenue_growth": info.get("revenueGrowth", "N/A"),
            "description": info.get("longBusinessSummary", ""),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "country": info.get("country", "N/A")
        }
    except Exception as e:
        return {"error": f"Failed to fetch stock data: {str(e)}"}


def search_stock_news(symbol: str, company_name: str) -> List[Dict[str, str]]:
    """Search for stock news using SerpAPI"""
    try:
        search_query = f"{symbol} {company_name} stock news"
        url = "https://serpapi.com/search"
        params = {
            "q": search_query,
            "api_key": SERP_API_KEY,
            "engine": "google",
            "num": 5,
            "tbm": "nws"
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        news_results = data.get("news_results", [])
        
        articles = []
        for item in news_results[:5]:
            articles.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "source": item.get("source", ""),
                "date": item.get("date", "")
            })
        
        return articles if articles else [{"title": "No recent news found", "snippet": "", "source": "", "date": ""}]
    except Exception as e:
        return [{"title": "Unable to fetch news", "snippet": str(e), "source": "", "date": ""}]


def get_market_indices() -> Dict[str, any]:
    """Get current market indices data"""
    indices = {
        "S&P 500": "^GSPC",
        "NIFTY 50": "^NSEI",
        "Dow Jones": "^DJI",
        "NASDAQ": "^IXIC"
    }
    
    results = {}
    for name, symbol in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change_pct = ((current - prev) / prev) * 100
                results[name] = {
                    "value": round(float(current), 2),
                    "change_percent": f"{change_pct:.2f}%"
                }
        except:
            results[name] = {"value": "N/A", "change_percent": "N/A"}
    
    return results


def calculate_what_if(symbol: str, months_ago: int, investment_amount: float) -> Dict:
    """Calculate hypothetical returns if invested X months ago"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=f"{months_ago+1}mo")
        
        if len(hist) < 2:
            return {"error": "Insufficient historical data"}
        
        old_price = hist['Close'].iloc[0]
        current_price = hist['Close'].iloc[-1]
        
        shares = investment_amount / old_price
        current_value = shares * current_price
        profit = current_value - investment_amount
        return_pct = (profit / investment_amount) * 100
        
        return {
            "investment_amount": investment_amount,
            "purchase_price": round(float(old_price), 2),
            "current_price": round(float(current_price), 2),
            "shares_bought": round(shares, 2),
            "current_value": round(float(current_value), 2),
            "profit_loss": round(float(profit), 2),
            "return_percent": round(float(return_pct), 2)
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# NODE FUNCTIONS - CLASSIFIER
# ============================================================================

def classify_query_node(state: InvestmentAgentState) -> Dict:
    """Classify if query is about single stock, general advice, or off-topic"""
    run_id = uuid.uuid4().hex[:8]
    print(f"[NODE RUN] classify_query - ID: {run_id}")
    
    llm = ChatGoogleGenerativeAI(
        model=Model_Name,
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )
    
    prompt = f"""Classify this investment query:

    Query: "{state['user_query']}"

    Classification Rules:

    1. "single_stock" = Asking about ONE specific company/ticker to analyze or invest in
    Examples: 
    * "Should I buy Apple?"
    * "Analyze Tesla stock"
    * "Is RELIANCE good?"
    * "Tell me about Google stock"
    
    2. "general_advice" = General investment questions about markets, sectors, portfolios
    Examples: 
    * "What should I invest in?"
    * "Which sectors are good now?"
    * "Compare tech vs pharma stocks"
    * "I have $5000, where to invest?"
    * "Recommend some stocks"
    * "Best investment for beginners"

    3. "off_topic" = NOT related to investments, stocks, finance, or money
    Examples:
    * "Do you know Cristiano Ronaldo?"
    * "I like to play video games"
    * "How to make biryani"
    * "What's the weather today?"
    * "Tell me a joke"
    * "Who won the World Cup?"

    Respond with ONLY ONE WORD: single_stock OR general_advice OR off_topic"""
    
    response = llm.invoke(prompt)
    query_type = response.content.strip().lower()
    
    # Fallback if LLM doesn't respond properly
    if "off_topic" in query_type:
        query_type = "off_topic"
    elif "single_stock" in query_type:
        query_type = "single_stock"
    elif "general_advice" in query_type:
        query_type = "general_advice"
    else:
        # Default to general_advice if unclear
        query_type = "general_advice"
    
    print(f"📋 Query classified as: {query_type}")
    
    return {
        "query_type": query_type,
        "messages": [f"📋 Query type: {query_type}"]
    }


# ============================================================================
# NODE FUNCTIONS - OFF-TOPIC PATH
# ============================================================================

def handle_off_topic_node(state: InvestmentAgentState) -> Dict:
    """Handle off-topic queries with a polite redirect"""
    run_id = uuid.uuid4().hex[:8]
    print(f"[NODE RUN] handle_off_topic - ID: {run_id}")
    
    final_response = """
    🤖 I'm an Investment Advisor AI

    I can only help with stock analysis, investment recommendations, and financial guidance.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ❌ Your question doesn't seem to be about investments or stocks.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ✅ WHAT I CAN HELP WITH:

    📊 Single Stock Analysis:
    • "Should I buy Apple stock?"
    • "Analyze Tesla right now"
    • "Is RELIANCE.NS a good investment?"
    • "Tell me about Microsoft stock"

    💡 General Investment Advice:
    • "What should I invest in?"
    • "Which sectors are performing well?"
    • "I have $5000, where should I invest?"
    • "Recommend some tech stocks"
    • "Compare pharma vs banking stocks"

    📈 Market Insights:
    • "What's happening in the stock market today?"
    • "Should I invest now or wait?"
    • "Which emerging markets look good?"

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    💬 Try asking me one of these questions!
    """
    
    return {
        "final_response": final_response,
        "messages": ["⚠️ Off-topic query handled"]
    }


# ============================================================================
# NODE FUNCTIONS - GENERAL ADVICE PATH
# ============================================================================

def general_advisor_node(state: InvestmentAgentState) -> Dict:
    """Provide general investment advice using web search + LLM"""
    run_id = uuid.uuid4().hex[:8]
    print(f"[NODE RUN] general_advisor - ID: {run_id}")
    
    llm = ChatGoogleGenerativeAI(
        model=Model_Name,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
    
    # Get current date
    today = datetime.now().strftime("%B %Y")
    
    # Search for current market conditions
    print("🔍 Searching market conditions...")
    market_search = search_web(f"stock market conditions {today} best sectors invest")
    
    # Search for gold/silver prices and trends
    print("🪙 Searching gold and silver prices...")
    gold_search = search_web(f"gold price today {today} investment trend India")
    silver_search = search_web(f"silver price today {today} investment trend")
    
    # Get market indices
    print("📊 Fetching market indices...")
    indices = get_market_indices()
    indices_text = "\n".join([f"- {name}: {data['value']} ({data['change_percent']})" 
                              for name, data in indices.items()])
    
    # Build comprehensive prompt
    prompt = f"""You are a personal investment advisor who considers ALL investment options, not just stocks.

    USER PROFILE:
    - Risk Tolerance: {state['user_profile'].get('risk_tolerance', 'N/A')}
    - Investment Horizon: {state['user_profile'].get('investment_horizon', 'N/A')}
    - Budget: {state['user_profile'].get('budget_currency', '$')}{state['user_profile'].get('budget', 'N/A')}
    - Goals: {state['user_profile'].get('investment_goals', 'N/A')}

    USER QUESTION:
    {state['user_query']}

    CURRENT MARKET DATA ({today}):
    Major Indices:
    {indices_text}

    Gold Price & Trend:
    {gold_search}

    Silver Price & Trend:
    {silver_search}

    Recent Market News & Analysis:
    {market_search}

    Respond in a clear, structured format."""
    
    response = llm.invoke(prompt)
    
    return {
        "general_advice": response.content,
        "market_conditions": f"Indices: {indices_text}\n\nMarket News: {market_search[:500]}...",
        "messages": ["💡 General advice generated"]
    }

def format_general_response_node(state: InvestmentAgentState) -> Dict:
    """Format the general advice response"""
    run_id = uuid.uuid4().hex[:8]
    print(f"[NODE RUN] format_general_response - ID: {run_id}")
    
    final_response = f"""
    🤖 INVESTMENT GUIDANCE FOR YOU
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    👤 YOUR PROFILE:
    - Risk: {state['user_profile'].get('risk_tolerance', 'N/A').title()}
    - Timeline: {state['user_profile'].get('investment_horizon', 'N/A')}
    - Budget: {state['user_profile'].get('budget_currency', '$')}{state['user_profile'].get('budget', 'N/A'):,}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    {state['general_advice']}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ⚠️ IMPORTANT DISCLAIMER: 
    This analysis is for educational purposes only and is NOT financial advice. 
    Markets are unpredictable. Always:
    - Do your own research
    - Consult with a qualified financial advisor
    - Never invest money you can't afford to lose
    - Diversify your investments
    """
    
    return {
        "final_response": final_response,
        "messages": ["✅ General advice formatted"]
    }


# ============================================================================
# NODE FUNCTIONS - SINGLE STOCK PATH (EXISTING)
# ============================================================================

def resolve_symbol_node(state: InvestmentAgentState) -> Dict:
    run_id = uuid.uuid4().hex[:8]
    print(f"[NODE RUN] resolve_symbol - ID: {run_id}")
    
    llm = ChatGoogleGenerativeAI(
        model=Model_Name,
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )
    
    resolution = resolve_stock_symbol(state['user_query'], llm)
    
    return {
        "company_name": resolution["company_name"],
        "stock_symbol": resolution["symbol"],
        "exchange": resolution["exchange"],
        "messages": [f"🔍 Resolved: {resolution['company_name']} ({resolution['symbol']})"]
    }


def gather_data_node(state: InvestmentAgentState) -> Dict:
    run_id = uuid.uuid4().hex[:8]
    print(f"[NODE RUN] gather_data - ID: {run_id}")
    
    symbol = state["stock_symbol"]
    stock_data = fetch_global_stock_data(symbol)
    
    if "error" in stock_data:
        base_symbol = symbol.split('.')[0]
        alternate_symbols = [
            f"{base_symbol}.NS",
            f"{base_symbol}.BO",
            base_symbol,
            f"{base_symbol}.L"
        ]
        
        for alt_symbol in alternate_symbols:
            if alt_symbol != symbol:
                print(f"Trying alternate: {alt_symbol}")
                stock_data = fetch_global_stock_data(alt_symbol)
                if "error" not in stock_data:
                    symbol = alt_symbol
                    break
    
    company_name = stock_data.get("company_name", state.get("company_name", symbol))
    news = search_stock_news(symbol, company_name)
    
    return {
        "stock_symbol": symbol,
        "stock_data": stock_data,
        "current_price": stock_data.get("price", 0),
        "news_articles": news,
        "messages": [f"✅ Data gathered for {symbol}"]
    }


def analyze_business_node(state: InvestmentAgentState) -> Dict:
    run_id = uuid.uuid4().hex[:8]
    print(f"[NODE RUN] analyze_business - ID: {run_id}")
    
    llm = ChatGoogleGenerativeAI(
        model=Model_Name,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
    
    stock_data = state["stock_data"]
    
    rev_growth = stock_data.get('revenue_growth', 'N/A')
    if rev_growth != 'N/A' and isinstance(rev_growth, (int, float)):
        rev_growth = f"{rev_growth*100:.1f}%"
    
    profit_margin = stock_data.get('profit_margin', 'N/A')
    if profit_margin != 'N/A' and isinstance(profit_margin, (int, float)):
        profit_margin = f"{profit_margin*100:.1f}%"
    
    prompt = f"""Analyze the business health of {stock_data.get('company_name')} ({stock_data.get('country', 'N/A')}) in simple terms.

DATA:
- Industry: {stock_data.get('industry', 'N/A')}
- Sector: {stock_data.get('sector', 'N/A')}
- Revenue Growth: {rev_growth}
- Profit Margin: {profit_margin}
- Market Cap: {stock_data.get('market_cap', 'N/A')}

Write 2-3 sentences in plain English about:
1. Is the company growing and profitable?
2. Is it financially healthy?
3. Any concerns about the business?

Be conversational and explain like talking to a beginner."""
    
    response = llm.invoke(prompt)
    
    return {
        "business_analysis": response.content,
        "messages": ["📊 Business analyzed"]
    }


def analyze_valuation_node(state: InvestmentAgentState) -> Dict:
    run_id = uuid.uuid4().hex[:8]
    print(f"[NODE RUN] analyze_valuation - ID: {run_id}")
    
    llm = ChatGoogleGenerativeAI(
        model=Model_Name,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
    
    stock_data = state["stock_data"]
    
    prompt = f"""Explain if {stock_data.get('symbol')} is expensive or cheap RIGHT NOW.

DATA:
- P/E Ratio: {stock_data.get('pe_ratio', 'N/A')}
- Forward P/E: {stock_data.get('forward_pe', 'N/A')}
- Sector: {stock_data.get('sector', 'N/A')}
- Industry: {stock_data.get('industry', 'N/A')}

Write 3-4 sentences explaining:
1. What does the P/E ratio mean in simple terms?
2. Is this stock expensive compared to others in the same industry?
3. Use analogies (like "paying $X for every $1 of profit")

Make it crystal clear for beginners."""
    
    response = llm.invoke(prompt)
    
    return {
        "valuation_analysis": response.content,
        "messages": ["💸 Valuation analyzed"]
    }


def analyze_momentum_node(state: InvestmentAgentState) -> Dict:
    run_id = uuid.uuid4().hex[:8]
    print(f"[NODE RUN] analyze_momentum - ID: {run_id}")
    
    llm = ChatGoogleGenerativeAI(
        model=Model_Name,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
    
    stock_data = state["stock_data"]
    
    prompt = f"""Describe the price momentum/trend for {stock_data.get('symbol')}.

DATA:
- Current: {stock_data.get('currency', '')} {stock_data.get('price')}
- Today's Change: {stock_data.get('change_percent')}
- 52-Week Range: {stock_data.get('currency', '')} {stock_data.get('52_week_low')} - {stock_data.get('currency', '')} {stock_data.get('52_week_high')}

Write 2-3 sentences:
1. Is the stock going up, down, or sideways recently?
2. Where is it in its 52-week range (near highs/lows)?
3. What does this tell us about investor sentiment?

Simple language, no jargon."""
    
    response = llm.invoke(prompt)
    
    return {
        "momentum_analysis": response.content,
        "messages": ["📈 Momentum analyzed"]
    }


def analyze_news_node(state: InvestmentAgentState) -> Dict:
    run_id = uuid.uuid4().hex[:8]
    print(f"[NODE RUN] analyze_news - ID: {run_id}")
    
    llm = ChatGoogleGenerativeAI(
        model=Model_Name,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
    
    news = state["news_articles"]
    news_text = "\n".join([
        f"- {article.get('title', '')} ({article.get('source', '')})"
        for article in news[:5]
    ])
    
    prompt = f"""Summarize the news sentiment about {state['stock_data'].get('symbol')}.

NEWS:
{news_text}

Write 3-4 sentences:
1. What's the GOOD news?
2. What's the BAD/CONCERNING news?
3. Overall: positive, negative, or mixed?

Simple, conversational tone."""
    
    response = llm.invoke(prompt)
    
    return {
        "news_sentiment": response.content,
        "messages": ["📰 News analyzed"]
    }


def predict_price_node(state: InvestmentAgentState) -> Dict:
    run_id = uuid.uuid4().hex[:8]
    print(f"[NODE RUN] predict_price - ID: {run_id}")
    
    llm = ChatGoogleGenerativeAI(
        model=Model_Name,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
    
    stock_data = state["stock_data"]
    currency = stock_data.get('currency', 'USD')
    
    prompt = f"""Predict potential price ranges for {stock_data.get('symbol')} over 6 months.

CURRENT DATA:
- Price: {currency} {stock_data.get('price')}
- P/E: {stock_data.get('pe_ratio')}
- Sector: {stock_data.get('sector')}
- Industry: {stock_data.get('industry')}
- Country: {stock_data.get('country', 'N/A')}

Provide TWO scenarios in this EXACT format:

🎯 IF THINGS GO WELL: {currency} XXX - {currency} XXX (up/down X-X%)
   Why? [1 sentence explaining what needs to happen]

⚠️ IF THINGS GO WRONG: {currency} XXX - {currency} XXX (up/down X-X%)
   Why? [1 sentence explaining what could go wrong]

Be realistic. Use actual percentages. Keep it brief."""
    
    response = llm.invoke(prompt)
    
    return {
        "price_prediction": response.content,
        "messages": ["🎯 Predictions made"]
    }


def assess_risk_node(state: InvestmentAgentState) -> Dict:
    run_id = uuid.uuid4().hex[:8]
    print(f"[NODE RUN] assess_risk - ID: {run_id}")
    
    llm = ChatGoogleGenerativeAI(
        model=Model_Name,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
    
    user_profile = state["user_profile"]
    stock_data = state["stock_data"]
    
    prompt = f"""Does {stock_data.get('symbol')} match this investor's profile?

INVESTOR:
- Risk: {user_profile.get('risk_tolerance')}
- Timeline: {user_profile.get('investment_horizon')}
- Budget: {user_profile.get('budget_currency', '$')}{user_profile.get('budget')}

STOCK:
- Country: {stock_data.get('country', 'N/A')}
- Sector: {stock_data.get('sector')}
- Industry: {stock_data.get('industry', 'N/A')}
- Volatility: [you judge based on P/E {stock_data.get('pe_ratio')} and sector]

Write 2-3 sentences in this format:
"⚠️ HONEST ANSWER: This is [GOOD/OKAY/NOT GOOD] match for you.

[Explain why in simple terms - is it too risky? Too volatile? Timeline mismatch? Currency risk?]"

Be brutally honest."""
    
    response = llm.invoke(prompt)
    
    return {
        "risk_assessment": response.content,
        "messages": ["⚠️ Risk assessed"]
    }


def generate_recommendation_node(state: InvestmentAgentState) -> Dict:
    run_id = uuid.uuid4().hex[:8]
    print(f"[NODE RUN] generate_recommendation - ID: {run_id}")
    
    llm = ChatGoogleGenerativeAI(
        model=Model_Name,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
    
    prompt = f"""Give final investment verdict for {state['stock_data'].get('symbol')}.

CONTEXT:
User: {state['user_profile'].get('risk_tolerance')}, {state['user_profile'].get('investment_horizon')}, {state['user_profile'].get('budget_currency', '$')}{state['user_profile'].get('budget')}
Price: {state['stock_data'].get('currency', '')} {state['current_price']}
Business: {state['business_analysis'][:200]}
Risk Match: {state['risk_assessment'][:200]}

Provide in this EXACT format:

✅ REASONS TO BUY:
1. [Reason 1 - be specific]
2. [Reason 2]
3. [Reason 3]

❌ REASONS NOT TO BUY (For YOUR situation):
1. [Reason 1 - personalized to this user]
2. [Reason 2]
3. [Reason 3]

🎯 FINAL ANSWER

FOR YOU: [✅ BUY / ⚠️ WAIT / ❌ DON'T BUY RIGHT NOW]

Why? [1-2 sentence summary]

What to do instead:
- [Specific action 1]
- [Specific action 2]
- CONSIDER ALTERNATIVES: [List 2-3 alternatives in the same sector/country with ticker symbols]

If you really want {state['stock_data'].get('symbol')}:
- [Risk mitigation strategy 1]
- [Risk mitigation strategy 2]

Be honest, direct, and actionable."""
    
    response = llm.invoke(prompt)
    
    return {
        "recommendation": response.content,
        "messages": ["💡 Recommendation ready"]
    }


def format_response_node(state: InvestmentAgentState) -> Dict:
    run_id = uuid.uuid4().hex[:8]
    print(f"[NODE RUN] format_response - ID: {run_id}")
    
    stock_data = state["stock_data"]
    currency = stock_data.get('currency', 'USD')
    
    market_cap = stock_data.get('market_cap', 'N/A')
    if market_cap != 'N/A' and market_cap:
        try:
            mc_float = float(market_cap)
            if mc_float >= 1_000_000_000_000:
                market_cap = f"{currency} {mc_float/1_000_000_000_000:.2f} Trillion"
            elif mc_float >= 1_000_000_000:
                market_cap = f"{currency} {mc_float/1_000_000_000:.2f} Billion"
            else:
                market_cap = f"{currency} {mc_float/1_000_000:.2f} Million"
        except:
            pass
    
    div_yield = stock_data.get('dividend_yield', 'N/A')
    if div_yield != 'N/A' and isinstance(div_yield, (int, float)):
        div_yield = f"{div_yield*100:.2f}%"
    
    final_response = f"""
💰 {stock_data.get('company_name', 'N/A')} ({stock_data.get('symbol', 'N/A')})
🌍 {stock_data.get('country', 'N/A')} | {stock_data.get('exchange', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 RIGHT NOW
Price: {currency} {stock_data.get('price', 'N/A')} ({stock_data.get('change_percent', 'N/A')} today)
Market Cap: {market_cap}
Sector: {stock_data.get('sector', 'N/A')}
Industry: {stock_data.get('industry', 'N/A')}
P/E Ratio: {stock_data.get('pe_ratio', 'N/A')}
Dividend Yield: {div_yield}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 WHERE THE PRICE COULD GO (Next 6 months)

{state['price_prediction']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 WHAT YOU NEED TO KNOW

📊 THE BUSINESS
{state['business_analysis']}

💸 THE VALUATION
{state['valuation_analysis']}

📈 THE MOMENTUM
{state['momentum_analysis']}

📰 THE NEWS
{state['news_sentiment']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 DOES THIS MATCH YOU?

Your Profile:
- Risk: {state['user_profile'].get('risk_tolerance', 'N/A').title()}
- Timeline: {state['user_profile'].get('investment_horizon', 'N/A')}
- Budget: {state['user_profile'].get('budget_currency', '$')}{state['user_profile'].get('budget', 'N/A'):,}

{state['risk_assessment']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 SHOULD YOU BUY OR NOT?

{state['recommendation']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ DISCLAIMER: This is NOT financial advice. Always do your own research and consult a financial advisor before investing.
"""
    
    return {
        "final_response": final_response,
        "messages": ["✅ Report complete"]
    }


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_investment_agent():
    workflow = StateGraph(InvestmentAgentState)
    
    # Add classifier node
    workflow.add_node("classify", classify_query_node)
    
    # Add off-topic handler
    workflow.add_node("handle_off_topic", handle_off_topic_node)
    
    # Add general advice path nodes
    workflow.add_node("general_advisor", general_advisor_node)
    workflow.add_node("format_general", format_general_response_node)
    
    # Add single stock path nodes
    workflow.add_node("resolve_symbol", resolve_symbol_node)
    workflow.add_node("gather_data", gather_data_node)
    workflow.add_node("analyze_business", analyze_business_node)
    workflow.add_node("analyze_valuation", analyze_valuation_node)
    workflow.add_node("analyze_momentum", analyze_momentum_node)
    workflow.add_node("analyze_news", analyze_news_node)
    workflow.add_node("predict_price", predict_price_node)
    workflow.add_node("assess_risk", assess_risk_node)
    workflow.add_node("recommend", generate_recommendation_node)
    workflow.add_node("format", format_response_node)
    
    # Set entry point
    workflow.set_entry_point("classify")
    
    # Conditional routing based on query type
    workflow.add_conditional_edges(
        "classify",
        lambda state: state["query_type"],
        {
            "single_stock": "resolve_symbol",
            "general_advice": "general_advisor",
            "off_topic": "handle_off_topic"
        }
    )
    
    # Off-topic path (shortest)
    workflow.add_edge("handle_off_topic", END)
    
    # General advice path
    workflow.add_edge("general_advisor", "format_general")
    workflow.add_edge("format_general", END)
    
    # Single stock path
    workflow.add_edge("resolve_symbol", "gather_data")
    workflow.add_edge("gather_data", "analyze_business")
    workflow.add_edge("analyze_business", "analyze_valuation")
    workflow.add_edge("analyze_valuation", "analyze_momentum")
    workflow.add_edge("analyze_momentum", "analyze_news")
    workflow.add_edge("analyze_news", "predict_price")
    workflow.add_edge("predict_price", "assess_risk")
    workflow.add_edge("assess_risk", "recommend")
    workflow.add_edge("recommend", "format")
    workflow.add_edge("format", END)
    
    return workflow.compile()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_investment_agent(user_query: str, user_profile: Dict = None):
    """
    Run the investment analysis agent
    
    Args:
        user_query: Any question (investment-related or not)
        user_profile: User's investment profile
    
    Example queries:
        Investment-related:
        - "Should I invest in Apple?"
        - "What stocks should I buy right now?"
        - "Which sectors are performing well?"
        
        Off-topic (will be politely redirected):
        - "Do you know Cristiano Ronaldo?"
        - "How to make biryani"
        - "Tell me a joke"
    """
    if user_profile is None:
        user_profile = {
            "risk_tolerance": "moderate",
            "investment_horizon": "medium-term (1-3 years)",
            "budget": 5000,
            "budget_currency": "$",
            "investment_goals": "growth with moderate risk"
        }
    
    initial_state = {
        "user_query": user_query,
        "user_profile": user_profile,
        "query_type": "",
        "company_name": "",
        "stock_symbol": "",
        "exchange": "",
        "current_price": 0.0,
        "stock_data": {},
        "news_articles": [],
        "market_conditions": "",
        "business_analysis": "",
        "valuation_analysis": "",
        "momentum_analysis": "",
        "news_sentiment": "",
        "price_prediction": "",
        "risk_assessment": "",
        "recommendation": "",
        "general_advice": "",
        "final_response": "",
        "messages": []
    }
    
    print("🤖 Analyzing your question...")
    print("━" * 60)
    
    agent = create_investment_agent()
    final_state = agent.invoke(initial_state)
    
    print("\n" + final_state["final_response"])
    
    return final_state