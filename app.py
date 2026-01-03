import streamlit as st
from main import run_investment_agent

# Page config
st.set_page_config(
    page_title="Investment Guidance Agent",
    page_icon="💰",
    layout="wide"
)

# Title
st.title("💰 Investment Guidance Agent")
st.markdown("Get AI-powered investment recommendations based on your profile")

# Sidebar for user profile
st.sidebar.header("Your Investment Profile")

risk_tolerance = st.sidebar.selectbox(
    "Risk Tolerance",
    ["conservative", "moderate", "aggressive"]
)

investment_horizon = st.sidebar.selectbox(
    "Investment Horizon",
    ["short-term (< 1 year)", "medium-term (1-3 years)", "long-term (5+ years)"]
)

budget = st.sidebar.number_input(
    "Investment Budget ($)",
    min_value=100,
    max_value=1000000,
    value=5000,
    step=100
)

investment_goals = st.sidebar.text_input(
    "Investment Goals",
    value="growth with moderate risk"
)

# Main input area
st.header("🔍 What do you want to know?")
user_query = st.text_area(
    "Enter your investment question:",
    placeholder="e.g., Should I invest in Apple stock? Should I buy Tesla? Is Google a good investment?",
    height=100
)

# Analyze button
if st.button("🚀 Analyze Investment", type="primary"):
    if user_query.strip():
        # Create user profile
        user_profile = {
            "risk_tolerance": risk_tolerance,
            "investment_horizon": investment_horizon,
            "budget": budget,
            "investment_goals": investment_goals
        }
        
        # Show loading spinner
        with st.spinner("🤖 Analyzing investment... This may take a minute..."):
            try:
                # Run the agent
                result = run_investment_agent(
                    user_query=user_query,
                    user_profile=user_profile
                )
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Show the formatted report
                st.markdown("---")
                st.markdown(result["final_response"])
                
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    else:
        st.warning("⚠️ Please enter an investment question first!")