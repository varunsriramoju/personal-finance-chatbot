import streamlit as st
import plotly.express as px
import pandas as pd
import json
import logging
import math
from utils.ai_models import GraniteAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === AI Helper ===
def get_ai_assistant():
    return GraniteAI()

@st.cache_resource
def initialize_ai_models():
    """Initialize Granite AI assistant"""
    try:
        return get_ai_assistant()
    except Exception as e:
        logger.error(f"Failed to initialize AI models: {e}")
        return None

# === Financial helpers (shortened for clarity, keep your same functions) ===
def calculate_fire_number(annual_expenses):
    return annual_expenses * 25

# === AI-enhanced response ===
def generate_enhanced_response(prompt, income, expenses, net_savings, currency_symbol, ai_assistant=None):
    """Generate comprehensive financial response"""
    financial_context = {
        "income": income,
        "expenses": expenses,
        "net_savings": net_savings,
        "currency_symbol": currency_symbol,
        "total_expenses": sum(expenses.values())
    }

    # Call Granite first
    if ai_assistant:
        try:
            ai_response = ai_assistant.generate_response(prompt, financial_context)
            return f"🤖 **AI Response:**\n\n{ai_response}"
        except Exception as e:
            logger.error(f"AI response failed: {e}")

    # Fallback if Granite fails
    return "📊 Using fallback analysis (Granite not available)."

# === Streamlit App ===
def main():
    ai_assistant = initialize_ai_models()

    st.set_page_config(page_title="AI Personal Finance Assistant", page_icon="💰", layout="wide")

    st.title("🤖 AI Personal Finance Assistant")
    st.write("Granite 2B Instruct integrated via Hugging Face 🚀")

    # Sidebar
    with st.sidebar:
        st.header("💼 Financial Profile")
        if ai_assistant:
            st.success("✅ Granite AI Active")
        else:
            st.warning("⚠️ Granite AI not available")

        currency_symbol = "₹"
        monthly_income = st.number_input("Monthly Income (₹)", min_value=0, value=5000, step=100)
        expenses = {
            "housing": st.number_input("Housing (₹)", min_value=0, value=1500, step=50),
            "food": st.number_input("Food (₹)", min_value=0, value=1000, step=50),
            "transport": st.number_input("Transport (₹)", min_value=0, value=500, step=50),
            "debt": st.number_input("Debt (₹)", min_value=0, value=0, step=50)
        }

    total_expenses = sum(expenses.values())
    net_savings = monthly_income - total_expenses

    st.markdown(f"### Net Savings: {currency_symbol}{net_savings}")

    # Chat UI
    st.subheader("💬 Ask your AI Assistant")
    user_prompt = st.text_input("Type a financial question:")
    if user_prompt:
        with st.spinner("Granite is thinking..."):
            response = generate_enhanced_response(
                user_prompt, monthly_income, expenses, net_savings, currency_symbol, ai_assistant
            )
        st.markdown(response)

if __name__ == "__main__":
    main()
