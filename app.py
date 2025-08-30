import streamlit as st
import plotly.express as px
import pandas as pd
import json
import logging
import math
import traceback
from utils.ai_models import GraniteAI, GraniteAISmaller

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === AI Helper with Better Error Handling ===
@st.cache_resource
def initialize_ai_models():
    """Initialize Granite AI assistant with caching and timeout protection"""
    try:
        # Add timeout protection and better error handling
        with st.spinner("Loading AI model... This may take a few minutes on first run."):
            # Try the smaller 1B model first for better performance
            logger.info("Attempting to load Granite 1B model...")
            ai_assistant = GraniteAI(model_name="ibm-granite/granite-3.0-1b-a400m-instruct")
            
            # Verify it loaded correctly with a simple test
            if ai_assistant.pipeline is not None:
                # Quick test to ensure it's working
                test_response = ai_assistant.generate_response("Hello", timeout=10)
                if test_response and "error" not in test_response.lower():
                    logger.info("Granite 1B model loaded and tested successfully")
                    return ai_assistant
                else:
                    logger.warning("Model loaded but test failed, falling back...")
            
            # If 1B fails, don't try 2B as it's likely a resource issue
            logger.warning("AI model failed to load properly")
            return None
            
    except Exception as e:
        logger.error(f"Failed to initialize AI models: {e}")
        logger.error(traceback.format_exc())
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

# === AI-enhanced response with timeout protection ===
def generate_enhanced_response(prompt, income, expenses, net_savings, currency_symbol, ai_assistant=None):
    """Generate comprehensive financial response using Granite AI with timeout protection"""
    
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

    # Try Granite AI with timeout protection
    if ai_assistant and hasattr(ai_assistant, 'pipeline') and ai_assistant.pipeline:
        try:
            # Add a progress indicator
            progress_bar = st.progress(0)
            progress_bar.progress(0.3)
            
            ai_response = ai_assistant.generate_response(prompt, financial_context, timeout=15)
            progress_bar.progress(1.0)
            progress_bar.empty()
            
            if ai_response and "error" not in ai_response.lower() and len(ai_response.strip()) > 10:
                # Add some financial metrics to the response
                metrics_info = f"""
                
📊 *Your Financial Snapshot:*
- Savings Rate: {savings_rate:.1f}%
- Monthly Surplus/Deficit: {currency_symbol}{net_savings:,.0f}
- Annual FIRE Number: {currency_symbol}{calculate_fire_number(total_expenses * 12):,.0f}
                """
                
                return f"🤖 *Granite AI Financial Advisor:*\n\n{ai_response}{metrics_info}"
            else:
                logger.warning("AI response was empty or contained errors")
                return generate_fallback_response(prompt, financial_context)
            
        except Exception as e:
            logger.error(f"Granite AI response failed: {e}")
            if hasattr(st, 'error'):
                st.error(f"AI response timeout. Using fallback analysis.")
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
    
    response = f"""📊 *Financial Analysis:*

*Current Status:* {indicator} {grade}
- Monthly Income: {currency_symbol}{income:,.0f}
- Monthly Expenses: {currency_symbol}{total_expenses:,.0f}
- Net Savings: {currency_symbol}{net_savings:,.0f}
- Savings Rate: {savings_rate:.1f}%

*Recommendations based on your question:*"""

    # Provide contextual advice based on the prompt
    prompt_lower = prompt.lower()
    
    if "save" in prompt_lower or "savings" in prompt_lower:
        if savings_rate < 0:
            response += f"""
🔴 *Priority: Stop the bleeding!*
- Track every expense for 7 days
- Cut all non-essential spending immediately
- Look for ways to increase income"""
        elif savings_rate < 10:
            response += f"""
🟡 *Build your savings foundation:*
- Aim for 10-15% savings rate initially
- Automate savings transfers
- Review your largest expense categories"""
        else:
            response += f"""
🟢 *Optimize your excellent savings:*
- Consider increasing to 20%+ savings rate
- Maximize tax-advantaged accounts
- Build emergency fund to 6 months expenses"""
    
    elif "invest" in prompt_lower:
        if savings_rate < 5:
            response += """
⚠ *Focus on budget first:*
- Build emergency fund before investing
- Ensure stable income and expenses
- Start with employer 401(k) match"""
        else:
            response += """
📈 *Ready to invest:*
- Start with low-cost index funds
- Consider target-date funds for simplicity
- Automate investments for consistency"""
    
    elif "debt" in prompt_lower:
        debt_payment = context['expenses'].get('debt', 0)
        if debt_payment > 0:
            response += f"""
💪 *Debt elimination strategy:*
- Current debt payments: {currency_symbol}{debt_payment:,.0f}/month
- Consider debt avalanche (pay highest interest first)
- Look for ways to increase payments"""
        else:
            response += """
✅ *Great! No debt payments detected:*
- Focus on building wealth through investing
- Maintain good credit habits
- Avoid taking on unnecessary debt"""
    
    else:
        # General advice
        if savings_rate < 0:
            response += """
🚨 *Immediate action needed:*
- Create spending plan immediately
- List all expenses and categorize
- Find areas to cut spending"""
        elif savings_rate < 10:
            response += """
🎯 *Next steps for financial health:*
- Build to 10%+ savings rate
- Create 3-month emergency fund
- Review insurance coverage"""
        else:
            response += """
🌟 *Excellent financial position:*
- Consider advanced strategies
- Maximize tax-advantaged investing
- Plan for long-term wealth building"""

    response += f"""

💡 *Note:* AI assistant not available. For advanced personalized advice, ensure transformers and torch are properly installed."""
    
    return response

# === Streamlit App ===
def main():
    # Page config with error handling
    try:
        st.set_page_config(
            page_title="AI Personal Finance Assistant", 
            page_icon="💰", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception as e:
        logger.error(f"Page config error: {e}")

    # Header
    st.title("🤖 AI Personal Finance Assistant")
    st.markdown("Powered by IBM Granite Instruct Model 🚀")

    # Add performance notice
    with st.expander("⚠ Performance Notice"):
        st.info("""
        *If the app is hanging:*
        1. The AI model loading can take 2-5 minutes on first run
        2. Reduce model size in utils/ai_models.py if needed
        3. The app works fine without AI - financial calculations run locally
        4. Refresh the page if it's been loading for more than 5 minutes
        """)

    # Initialize AI with better error handling
    ai_assistant = None
    try:
        ai_assistant = initialize_ai_models()
    except Exception as e:
        logger.error(f"AI initialization failed: {e}")
        st.warning("⚠ AI model failed to load. App will work in basic mode.")

    # Sidebar for financial inputs
    with st.sidebar:
        st.header("💼 Financial Profile")
        
        # AI Status indicator with more details
        if ai_assistant and hasattr(ai_assistant, 'pipeline') and ai_assistant.pipeline:
            st.success("✅ Granite AI Active")
            
            # Show model info
            with st.expander("🔧 Model Info"):
                try:
                    model_info = ai_assistant.get_model_info()
                    if "error" not in model_info:
                        st.json(model_info)
                    else:
                        st.warning("Model info unavailable")
                except Exception as e:
                    st.error(f"Error getting model info: {e}")
        else:
            st.warning("⚠ Granite AI not available")
            st.info("💡 Install: pip install transformers torch")
            st.info("🔧 App runs in fallback mode with manual financial analysis")

        # Currency selection
        currency_options = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP"}
        currency_symbol = st.selectbox("Currency", options=list(currency_options.keys()), index=0)
        
        st.subheader("Monthly Income & Expenses")
        
        # Income
        monthly_income = st.number_input(
            f"Monthly Income ({currency_symbol})", 
            min_value=0, 
            value=50000, 
            step=1000,
            help="Enter your total monthly income"
        )
        
        st.subheader("Monthly Expenses")
        
        # Expenses with help text
        expenses = {
            "housing": st.number_input(f"Housing ({currency_symbol})", min_value=0, value=15000, step=500, 
                                     help="Rent, mortgage, property taxes"),
            "food": st.number_input(f"Food ({currency_symbol})", min_value=0, value=8000, step=500,
                                  help="Groceries and dining out"),
            "transport": st.number_input(f"Transport ({currency_symbol})", min_value=0, value=5000, step=500,
                                       help="Car payments, gas, public transit"),
            "utilities": st.number_input(f"Utilities ({currency_symbol})", min_value=0, value=3000, step=200,
                                       help="Electricity, water, internet, phone"),
            "healthcare": st.number_input(f"Healthcare ({currency_symbol})", min_value=0, value=2000, step=200,
                                        help="Insurance premiums, medical expenses"),
            "entertainment": st.number_input(f"Entertainment ({currency_symbol})", min_value=0, value=4000, step=300,
                                           help="Movies, subscriptions, hobbies"),
            "debt": st.number_input(f"Debt Payments ({currency_symbol})", min_value=0, value=0, step=500,
                                  help="Credit cards, loans (excluding mortgage)"),
            "other": st.number_input(f"Other ({currency_symbol})", min_value=0, value=3000, step=300,
                                   help="Miscellaneous expenses")
        }

    # Calculate totals with error handling
    try:
        total_expenses = sum(expenses.values())
        net_savings = monthly_income - total_expenses
        savings_rate = calculate_savings_rate(monthly_income, total_expenses)
        grade, indicator = get_financial_health_grade(savings_rate)
    except Exception as e:
        logger.error(f"Calculation error: {e}")
        total_expenses = 0
        net_savings = 0
        savings_rate = 0
        grade, indicator = "Error", "❌"

    # Main content area
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Monthly Income", f"{currency_symbol}{monthly_income:,.0f}")
    
    with col2:
        st.metric("Monthly Expenses", f"{currency_symbol}{total_expenses:,.0f}")
    
    with col3:
        color = "normal" if net_savings >= 0 else "inverse"
        st.metric("Net Savings", f"{currency_symbol}{net_savings:,.0f}")

    # Financial Health Score
    st.subheader(f"Financial Health: {indicator} {grade}")
    
    if savings_rate > 0:
        progress_value = max(0, min(savings_rate / 25, 1.0))  # Cap at 25% for progress bar
        st.progress(progress_value)
    else:
        st.progress(0)

    # Expense Breakdown Chart with error handling
    if total_expenses > 0:
        st.subheader("📊 Expense Breakdown")
        
        try:
            expense_df = pd.DataFrame([
                {"Category": k.title(), "Amount": v, "Percentage": (v/total_expenses)*100}
                for k, v in expenses.items() if v > 0
            ])
            
            if not expense_df.empty:
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
        except Exception as e:
            logger.error(f"Chart creation error: {e}")
            st.error("Unable to create expense charts. Please check your expense data.")

    # AI Chat Interface with timeout protection
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
    
    # Add manual AI toggle for debugging
    use_ai = st.checkbox("Use AI Assistant (uncheck for faster fallback analysis)", 
                        value=True if ai_assistant else False, 
                        disabled=not ai_assistant)
    
    if user_prompt:
        # Show different spinners based on AI availability
        if use_ai and ai_assistant:
            with st.spinner("🤖 Granite AI is analyzing your finances... (this may take 30 seconds)"):
                try:
                    response = generate_enhanced_response(
                        user_prompt, 
                        monthly_income, 
                        expenses, 
                        net_savings, 
                        currency_symbol, 
                        ai_assistant
                    )
                except Exception as e:
                    logger.error(f"AI response generation failed: {e}")
                    st.error("AI response failed. Showing fallback analysis.")
                    response = generate_fallback_response(user_prompt, {
                        "income": monthly_income,
                        "expenses": expenses, 
                        "net_savings": net_savings,
                        "currency_symbol": currency_symbol,
                        "total_expenses": total_expenses,
                        "savings_rate": savings_rate
                    })
        else:
            with st.spinner("📊 Analyzing your finances..."):
                response = generate_fallback_response(user_prompt, {
                    "income": monthly_income,
                    "expenses": expenses,
                    "net_savings": net_savings, 
                    "currency_symbol": currency_symbol,
                    "total_expenses": total_expenses,
                    "savings_rate": savings_rate
                })
        
        st.markdown("### Financial Analysis:")
        st.markdown(response)

    # Additional Financial Insights
    with st.expander("📈 Additional Financial Insights"):
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Annual Expenses", f"{currency_symbol}{total_expenses * 12:,.0f}")
                fire_number = calculate_fire_number(total_expenses * 12)
                st.metric("FIRE Number", f"{currency_symbol}{fire_number:,.0f}")
                
            with col2:
                years_to_fire = 25 / max(savings_rate/100, 0.01) if savings_rate > 0 else float('inf')
                fire_display = f"{years_to_fire:.1f}" if years_to_fire != float('inf') else "∞"
                st.metric("Years to FIRE", fire_display)
                st.metric("Emergency Fund Target", f"{currency_symbol}{total_expenses * 6:,.0f}")
        except Exception as e:
            logger.error(f"Insights calculation error: {e}")
            st.error("Unable to calculate additional insights.")

    # Debug info for troubleshooting
    if st.checkbox("Show Debug Info"):
        st.subheader("🔧 Debug Information")
        st.json({
            "ai_available": ai_assistant is not None,
            "ai_pipeline_loaded": hasattr(ai_assistant, 'pipeline') and ai_assistant.pipeline is not None if ai_assistant else False,
            "total_expenses": total_expenses,
            "net_savings": net_savings,
            "savings_rate": round(savings_rate, 2)
        })

    # Footer
    st.markdown("---")
    st.markdown("💡 This app uses IBM Granite models for AI-powered financial advice. All calculations are estimates.")

# Add error boundary
def safe_main():
    """Run main with error boundary"""
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.error("Please refresh the page and try again.")
        logger.error(f"Main app error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    safe_main()