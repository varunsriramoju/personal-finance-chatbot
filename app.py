import streamlit as st
import plotly.express as px
import pandas as pd
import json
import logging
import math
from utils.ai_models import GraniteAI, GraniteAISmaller

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === AI Helper ===
@st.cache_resource
def initialize_ai_models():
    """Initialize Granite AI assistant with caching"""
    try:
        # Try the 2B model first (recommended)
        ai_assistant = GraniteAI(model_name="ibm-granite/granite-3.0-2b-instruct")
        
        # Verify it loaded correctly
        if ai_assistant.pipeline is not None:
            logger.info("Granite 2B model loaded successfully")
            return ai_assistant
        else:
            # Fallback to even smaller model
            logger.warning("Trying smaller 1B model...")
            ai_assistant = GraniteAI(model_name="ibm-granite/granite-3.0-1b-a400m-instruct")
            return ai_assistant
            
    except Exception as e:
        logger.error(f"Failed to initialize AI models: {e}")
        return None

# === Financial helpers ===
def calculate_fire_number(annual_expenses):
    """Calculate FIRE number (25x annual expenses)"""
    return annual_expenses * 25

def calculate_savings_rate(income, total_expenses):
    """Calculate savings rate percentage"""
    if income <= 0:
        return 0
    return ((income - total_expenses) / income) * 100

def get_financial_health_grade(savings_rate):
    """Get financial health grade based on savings rate"""
    if savings_rate >= 20:
        return "A+ (Excellent)", "🟢"
    elif savings_rate >= 15:
        return "A (Very Good)", "🟢"
    elif savings_rate >= 10:
        return "B (Good)", "🟡"
    elif savings_rate >= 5:
        return "C (Fair)", "🟠"
    elif savings_rate >= 0:
        return "D (Needs Improvement)", "🔴"
    else:
        return "F (Critical)", "🔴"

# === AI-enhanced response ===
def generate_enhanced_response(prompt, income, expenses, net_savings, currency_symbol, ai_assistant=None):
    """Generate comprehensive financial response using Granite AI"""
    
    # Prepare financial context
    total_expenses = sum(expenses.values())
    savings_rate = calculate_savings_rate(income, total_expenses)
    
    financial_context = {
        "income": income,
        "expenses": expenses,
        "net_savings": net_savings,
        "currency_symbol": currency_symbol,
        "total_expenses": total_expenses,
        "savings_rate": savings_rate
    }

    # Try Granite AI first
    if ai_assistant and ai_assistant.pipeline:
        try:
            ai_response = ai_assistant.generate_response(prompt, financial_context)
            
            # Add some financial metrics to the response
            metrics_info = f"""
            
📊 **Your Financial Snapshot:**
- Savings Rate: {savings_rate:.1f}%
- Monthly Surplus/Deficit: {currency_symbol}{net_savings:,.0f}
- Annual FIRE Number: {currency_symbol}{calculate_fire_number(total_expenses * 12):,.0f}
            """
            
            return f"🤖 **Granite AI Financial Advisor:**\n\n{ai_response}{metrics_info}"
            
        except Exception as e:
            logger.error(f"Granite AI response failed: {e}")
            return generate_fallback_response(prompt, financial_context)
    
    # Fallback response
    return generate_fallback_response(prompt, financial_context)

def generate_fallback_response(prompt, context):
    """Generate fallback response when AI is not available"""
    income = context['income']
    total_expenses = context['total_expenses']
    net_savings = context['net_savings']
    currency_symbol = context['currency_symbol']
    savings_rate = context.get('savings_rate', 0)
    
    # Basic financial analysis
    grade, indicator = get_financial_health_grade(savings_rate)
    
    response = f"""📊 **Financial Analysis (Fallback Mode):**

**Current Status:** {indicator} {grade}
- Monthly Income: {currency_symbol}{income:,.0f}
- Monthly Expenses: {currency_symbol}{total_expenses:,.0f}
- Net Savings: {currency_symbol}{net_savings:,.0f}
- Savings Rate: {savings_rate:.1f}%

**Quick Recommendations:**"""

    if savings_rate < 0:
        response += f"""
🔴 **Urgent:** You're spending more than you earn!
- Review all expenses immediately
- Consider additional income sources
- Cut non-essential spending"""
    elif savings_rate < 10:
        response += f"""
🟡 **Focus Areas:**
- Aim for 10-20% savings rate
- Review largest expense categories
- Look for ways to increase income"""
    else:
        response += f"""
🟢 **Good progress!** Consider:
- Building emergency fund (6 months expenses)
- Investing surplus in index funds
- Planning for long-term goals"""

    response += f"""

💡 **Note:** Granite AI is not available. Install transformers and torch for AI-powered advice."""
    
    return response

# === Streamlit App ===
def main():
    # Page config
    st.set_page_config(
        page_title="AI Personal Finance Assistant", 
        page_icon="💰", 
        layout="wide"
    )

    # Initialize AI
    ai_assistant = initialize_ai_models()

    # Header
    st.title("🤖 AI Personal Finance Assistant")
    st.markdown("*Powered by IBM Granite 2B Instruct Model via Hugging Face* 🚀")

    # Sidebar for financial inputs
    with st.sidebar:
        st.header("💼 Financial Profile")
        
        # AI Status indicator
        if ai_assistant and ai_assistant.pipeline:
            st.success("✅ Granite AI Active")
            
            # Show model info
            with st.expander("🔧 Model Info"):
                model_info = ai_assistant.get_model_info()
                if "error" not in model_info:
                    st.json(model_info)
        else:
            st.warning("⚠️ Granite AI not available")
            st.info("💡 Install: `pip install transformers torch`")

        # Currency selection
        currency_options = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP"}
        currency_symbol = st.selectbox("Currency", options=list(currency_options.keys()), index=0)
        
        st.subheader("Monthly Income & Expenses")
        
        # Income
        monthly_income = st.number_input(
            f"Monthly Income ({currency_symbol})", 
            min_value=0, 
            value=50000, 
            step=1000
        )
        
        st.subheader("Monthly Expenses")
        
        # Expenses
        expenses = {
            "housing": st.number_input(f"Housing ({currency_symbol})", min_value=0, value=15000, step=500),
            "food": st.number_input(f"Food ({currency_symbol})", min_value=0, value=8000, step=500),
            "transport": st.number_input(f"Transport ({currency_symbol})", min_value=0, value=5000, step=500),
            "utilities": st.number_input(f"Utilities ({currency_symbol})", min_value=0, value=3000, step=200),
            "healthcare": st.number_input(f"Healthcare ({currency_symbol})", min_value=0, value=2000, step=200),
            "entertainment": st.number_input(f"Entertainment ({currency_symbol})", min_value=0, value=4000, step=300),
            "debt": st.number_input(f"Debt Payments ({currency_symbol})", min_value=0, value=0, step=500),
            "other": st.number_input(f"Other ({currency_symbol})", min_value=0, value=3000, step=300)
        }

    # Calculate totals
    total_expenses = sum(expenses.values())
    net_savings = monthly_income - total_expenses
    savings_rate = calculate_savings_rate(monthly_income, total_expenses)
    grade, indicator = get_financial_health_grade(savings_rate)

    # Main content area
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Monthly Income", f"{currency_symbol}{monthly_income:,.0f}")
    
    with col2:
        st.metric("Monthly Expenses", f"{currency_symbol}{total_expenses:,.0f}")
    
    with col3:
        color = "normal" if net_savings >= 0 else "inverse"
        st.metric("Net Savings", f"{currency_symbol}{net_savings:,.0f}", delta=None)

    # Financial Health Score
    st.subheader(f"Financial Health: {indicator} {grade}")
    
    progress_value = max(0, min(savings_rate / 25, 1.0))  # Cap at 25% for progress bar
    st.progress(progress_value)

    # Expense Breakdown Chart
    if total_expenses > 0:
        st.subheader("📊 Expense Breakdown")
        
        expense_df = pd.DataFrame([
            {"Category": k.title(), "Amount": v, "Percentage": (v/total_expenses)*100}
            for k, v in expenses.items() if v > 0
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                expense_df, 
                values="Amount", 
                names="Category",
                title="Monthly Expenses Distribution"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(
                expense_df.sort_values("Amount", ascending=True), 
                x="Amount", 
                y="Category",
                orientation='h',
                title="Expense Categories",
                color="Amount",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # AI Chat Interface
    st.subheader("💬 Chat with Your AI Financial Advisor")
    
    # Sample questions
    sample_questions = [
        "How can I improve my savings rate?",
        "What's the best way to build an emergency fund?",
        "Should I focus on debt payoff or investing?",
        "How much should I save for retirement?",
        "What are some ways to reduce my expenses?"
    ]
    
    selected_question = st.selectbox("Quick Questions:", [""] + sample_questions)
    
    user_prompt = st.text_input(
        "Ask a financial question:", 
        value=selected_question if selected_question else "",
        placeholder="e.g., How can I save more money each month?"
    )
    
    if user_prompt:
        with st.spinner("🤖 Granite AI is analyzing your finances..."):
            response = generate_enhanced_response(
                user_prompt, 
                monthly_income, 
                expenses, 
                net_savings, 
                currency_symbol, 
                ai_assistant
            )
        
        st.markdown("### AI Response:")
        st.markdown(response)

    # Additional Financial Insights
    with st.expander("📈 Additional Financial Insights"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Annual Expenses", f"{currency_symbol}{total_expenses * 12:,.0f}")
            st.metric("FIRE Number", f"{currency_symbol}{calculate_fire_number(total_expenses * 12):,.0f}")
            
        with col2:
            st.metric("Years to FIRE", f"{25 / max(savings_rate/100, 0.01):.1f}" if savings_rate > 0 else "∞")
            st.metric("Emergency Fund Target", f"{currency_symbol}{total_expenses * 6:,.0f}")

    # Footer
    st.markdown("---")
    st.markdown("*💡 This app uses IBM Granite 2B Instruct for AI-powered financial advice. All calculations are estimates.*")

if __name__ == "__main__":
    main()