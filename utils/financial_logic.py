import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime, timedelta
import math

class FinancialAnalyzer:
    """Advanced financial analysis and calculations"""
    
    def __init__(self):
        self.financial_rules = self._load_financial_rules()
    
    def _load_financial_rules(self) -> Dict[str, Any]:
        """Load standard financial planning rules and benchmarks"""
        return {
            "housing_max_percentage": 30,  # Max 30% of income on housing
            "transport_max_percentage": 15,  # Max 15% on transportation
            "food_max_percentage": 12,  # Target food spending
            "entertainment_max_percentage": 5,  # Entertainment budget
            "minimum_emergency_months": 3,  # Minimum emergency fund
            "target_emergency_months": 6,   # Target emergency fund
            "minimum_savings_rate": 10,     # Minimum savings rate %
            "target_savings_rate": 20,      # Target savings rate %
            "debt_to_income_max": 36,       # Max debt-to-income ratio
            "investment_allocation_aggressive": {"stocks": 90, "bonds": 10},
            "investment_allocation_moderate": {"stocks": 70, "bonds": 30},
            "investment_allocation_conservative": {"stocks": 50, "bonds": 50}
        }
    
    def calculate_financial_health_score(self, income: float, expenses: Dict[str, float], 
                                       emergency_fund: float = 0) -> Dict[str, Any]:
        """Calculate comprehensive financial health score (0-100)"""
        
        if income <= 0:
            return {"score": 0, "grade": "F", "factors": ["No income data"]}
        
        total_expenses = sum(expenses.values())
        net_savings = income - total_expenses
        savings_rate = (net_savings / income) * 100
        
        score_components = {}
        
        # 1. Savings Rate Score (30 points)
        if savings_rate >= 20:
            score_components["savings_rate"] = 30
        elif savings_rate >= 10:
            score_components["savings_rate"] = 20
        elif savings_rate > 0:
            score_components["savings_rate"] = 10
        else:
            score_components["savings_rate"] = 0
        
        # 2. Housing Ratio Score (25 points)
        housing_ratio = (expenses.get('housing', 0) / income) * 100
        if housing_ratio <= 25:
            score_components["housing"] = 25
        elif housing_ratio <= 30:
            score_components["housing"] = 20
        elif housing_ratio <= 35:
            score_components["housing"] = 10
        else:
            score_components["housing"] = 0
        
        # 3. Emergency Fund Score (20 points)
        emergency_months = emergency_fund / total_expenses if total_expenses > 0 else 0
        if emergency_months >= 6:
            score_components["emergency"] = 20
        elif emergency_months >= 3:
            score_components["emergency"] = 15
        elif emergency_months >= 1:
            score_components["emergency"] = 10
        else:
            score_components["emergency"] = 0
        
        # 4. Debt Management Score (15 points)
        debt_ratio = (expenses.get('debt', 0) / income) * 100
        if debt_ratio == 0:
            score_components["debt"] = 15
        elif debt_ratio <= 10:
            score_components["debt"] = 12
        elif debt_ratio <= 20:
            score_components["debt"] = 8
        else:
            score_components["debt"] = 0
        
        # 5. Budget Balance Score (10 points)
        total_non_essential = expenses.get('entertainment', 0) + expenses.get('other', 0)
        non_essential_ratio = (total_non_essential / income) * 100
        if non_essential_ratio <= 10:
            score_components["balance"] = 10
        elif non_essential_ratio <= 15:
            score_components["balance"] = 7
        else:
            score_components["balance"] = 3
        
        # Calculate total score
        total_score = sum(score_components.values())
        
        # Determine grade
        if total_score >= 85:
            grade = "A+"
        elif total_score >= 80:
            grade = "A"
        elif total_score >= 75:
            grade = "B+"
        elif total_score >= 70:
            grade = "B"
        elif total_score >= 65:
            grade = "C+"
        elif total_score >= 60:
            grade = "C"
        elif total_score >= 50:
            grade = "D"
        else:
            grade = "F"
        
        # Generate improvement factors
        improvement_factors = []
        if score_components["savings_rate"] < 20:
            improvement_factors.append("Increase savings rate")
        if score_components["housing"] < 20:
            improvement_factors.append("Reduce housing costs")
        if score_components["emergency"] < 15:
            improvement_factors.append("Build emergency fund")
        if score_components["debt"] < 10:
            improvement_factors.append("Pay down debt faster")
        
        return {
            "score": total_score,
            "grade": grade,
            "components": score_components,
            "factors": improvement_factors,
            "savings_rate": savings_rate,
            "housing_ratio": housing_ratio,
            "emergency_months": emergency_months,
            "debt_ratio": debt_ratio
        }
    
    def calculate_investment_allocation(self, age: int, risk_tolerance: str) -> Dict[str, float]:
        """Calculate optimal investment allocation based on age and risk tolerance"""
        
        # Age-based allocation (Rule of 110)
        stock_percentage_base = 110 - age
        
        # Adjust based on risk tolerance
        risk_adjustments = {
            "conservative": -20,
            "moderate": 0,
            "aggressive": +10
        }
        
        adjustment = risk_adjustments.get(risk_tolerance.lower(), 0)
        stock_percentage = max(30, min(90, stock_percentage_base + adjustment))
        bond_percentage = 100 - stock_percentage
        
        return {
            "stocks": stock_percentage,
            "bonds": bond_percentage,
            "recommended_funds": self._get_fund_recommendations(stock_percentage)
        }
    
    def _get_fund_recommendations(self, stock_percentage: float) -> List[str]:
        """Get fund recommendations based on allocation"""
        recommendations = []
        
        if stock_percentage >= 80:
            recommendations.extend([
                "Total Stock Market Index (VTSAX/VTI)",
                "S&P 500 Index Fund (VFIAX/VOO)",
                "International Stock Index (VTIAX/VXUS)"
            ])
        else:
            recommendations.extend([
                "Target-Date Fund (matches your age)",
                "Balanced Index Fund (60/40 stocks/bonds)",
                "Total Bond Market Index (VBTLX/BND)"
            ])
        
        return recommendations
    
    def calculate_debt_payoff_plan(self, debts: List[Dict[str, Any]], 
                                 extra_payment: float = 0) -> Dict[str, Any]:
        """Calculate debt payoff using avalanche and snowball methods"""
        
        if not debts:
            return {"error": "No debts to analyze"}
        
        # Avalanche method (highest interest rate first)
        avalanche_plan = self._calculate_payoff_strategy(debts, extra_payment, "avalanche")
        
        # Snowball method (lowest balance first)  
        snowball_plan = self._calculate_payoff_strategy(debts, extra_payment, "snowball")
        
        return {
            "avalanche": avalanche_plan,
            "snowball": snowball_plan,
            "recommendation": "avalanche" if avalanche_plan["total_interest"] < snowball_plan["total_interest"] else "snowball"
        }
    
    def _calculate_payoff_strategy(self, debts: List[Dict[str, Any]], 
                                 extra_payment: float, method: str) -> Dict[str, Any]:
        """Calculate specific payoff strategy"""
        
        # Sort debts based on method
        if method == "avalanche":
            sorted_debts = sorted(debts, key=lambda x: x["interest_rate"], reverse=True)
        else:  # snowball
            sorted_debts = sorted(debts, key=lambda x: x["balance"])
        
        total_interest = 0
        total_months = 0
        payoff_schedule = []
        
        # Simplified calculation for demo
        for debt in sorted_debts:
            balance = debt["balance"]
            rate = debt["interest_rate"] / 100 / 12
            min_payment = debt["minimum_payment"]
            
            # Calculate months to payoff
            if rate > 0:
                months = math.ceil(math.log(1 + (balance * rate) / min_payment) / math.log(1 + rate))
            else:
                months = math.ceil(balance / min_payment) if min_payment > 0 else 999
            
            interest_paid = (min_payment * months) - balance
            total_interest += interest_paid
            total_months = max(total_months, months)
            
            payoff_schedule.append({
                "name": debt["name"],
                "months": months,
                "interest_paid": interest_paid
            })
        
        return {
            "total_months": total_months,
            "total_interest": total_interest,
            "schedule": payoff_schedule
        }
    
    def calculate_retirement_needs(self, current_age: int, retirement_age: int,
                                 current_income: float, desired_replacement_ratio: float = 0.8) -> Dict[str, Any]:
        """Calculate retirement savings needs"""
        
        years_to_retirement = retirement_age - current_age
        
        if years_to_retirement <= 0:
            return {"error": "Already at or past retirement age"}
        
        # Calculate needed retirement income
        desired_annual_income = current_income * 12 * desired_replacement_ratio
        
        # Use 4% withdrawal rule to calculate needed nest egg
        needed_nest_egg = desired_annual_income / 0.04
        
        # Calculate required monthly savings (assuming 7% annual return)
        monthly_return_rate = 0.07 / 12
        months_to_retirement = years_to_retirement * 12
        
        # Future value of annuity formula
        required_monthly_savings = needed_nest_egg * monthly_return_rate / \
                                 ((1 + monthly_return_rate) ** months_to_retirement - 1)
        
        return {
            "needed_nest_egg": needed_nest_egg,
            "required_monthly_savings": required_monthly_savings,
            "years_to_retirement": years_to_retirement,
            "desired_annual_income": desired_annual_income,
            "replacement_ratio": desired_replacement_ratio
        }
    
    def analyze_cash_flow_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cash flow trends over time"""
        if not historical_data:
            return {"error": "No historical data available"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(historical_data)
        
        # Calculate trends
        income_trend = np.polyfit(range(len(df)), df['income'], 1)[0] if len(df) > 1 else 0
        expense_trend = np.polyfit(range(len(df)), df['total_expenses'], 1)[0] if len(df) > 1 else 0
        savings_trend = np.polyfit(range(len(df)), df['net_savings'], 1)[0] if len(df) > 1 else 0
        
        return {
            "income_trend": "increasing" if income_trend > 0 else "decreasing" if income_trend < 0 else "stable",
            "expense_trend": "increasing" if expense_trend > 0 else "decreasing" if expense_trend < 0 else "stable",
            "savings_trend": "improving" if savings_trend > 0 else "declining" if savings_trend < 0 else "stable",
            "trend_analysis": {
                "income_change_rate": income_trend,
                "expense_change_rate": expense_trend,
                "savings_change_rate": savings_trend
            }
        }
    
    def generate_budget_recommendations(self, financial_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate specific budget recommendations with AI insights"""
        
        income = financial_data.get('income', 0)
        expenses = financial_data.get('expenses', {})
        net_savings = financial_data.get('net_savings', 0)
        
        recommendations = []
        
        if income <= 0:
            recommendations.append({
                "type": "critical",
                "title": "Income Required",
                "description": "Enter your income to receive personalized recommendations",
                "action": "Update income in sidebar"
            })
            return recommendations
        
        # Analyze each expense category
        for category, amount in expenses.items():
            if amount > 0:
                percentage = (amount / income) * 100
                recommendation = self._analyze_expense_category(category, percentage, amount)
                if recommendation:
                    recommendations.append(recommendation)
        
        # Overall savings recommendations
        savings_rate = (net_savings / income) * 100
        savings_rec = self._analyze_savings_rate(savings_rate, net_savings)
        if savings_rec:
            recommendations.append(savings_rec)
        
        # Emergency fund recommendations
        emergency_rec = self._analyze_emergency_fund(expenses, net_savings)
        if emergency_rec:
            recommendations.append(emergency_rec)
        
        return sorted(recommendations, key=lambda x: {"critical": 3, "warning": 2, "suggestion": 1}[x["type"]], reverse=True)
    
    def _analyze_expense_category(self, category: str, percentage: float, amount: float) -> Dict[str, str]:
        """Analyze individual expense category"""
        
        category_rules = {
            "housing": {"max": 30, "target": 25, "tips": "Consider downsizing, roommates, or refinancing"},
            "transport": {"max": 15, "target": 12, "tips": "Use public transit, carpool, or bike more often"},
            "food": {"max": 15, "target": 10, "tips": "Meal prep, cook at home, use grocery coupons"},
            "entertainment": {"max": 8, "target": 5, "tips": "Find free activities, limit subscriptions"},
            "healthcare": {"max": 10, "target": 8, "tips": "Shop for better insurance, use HSA"},
            "other": {"max": 10, "target": 7, "tips": "Track miscellaneous spending carefully"}
        }
        
        if category not in category_rules:
            return None
        
        rules = category_rules[category]
        
        if percentage > rules["max"]:
            return {
                "type": "critical" if percentage > rules["max"] + 5 else "warning",
                "title": f"{category.title()} Cost Alert",
                "description": f"{percentage:.1f}% of income (target: <{rules['max']}%)",
                "action": rules["tips"]
            }
        elif percentage > rules["target"]:
            return {
                "type": "suggestion",
                "title": f"Optimize {category.title()} Spending",
                "description": f"{percentage:.1f}% of income (could target {rules['target']}%)",
                "action": rules["tips"]
            }
        
        return None
    
    def _analyze_savings_rate(self, savings_rate: float, net_savings: float) -> Dict[str, str]:
        """Analyze savings rate and provide recommendations"""
        
        if savings_rate < 0:
            return {
                "type": "critical",
                "title": "Budget Deficit Alert",
                "description": f"Spending {abs(savings_rate):.1f}% more than income",
                "action": "Cut expenses immediately or increase income"
            }
        elif savings_rate < 10:
            return {
                "type": "warning", 
                "title": "Low Savings Rate",
                "description": f"{savings_rate:.1f}% savings rate (target: 10-20%)",
                "action": "Find ways to reduce expenses or increase income"
            }
        elif savings_rate < 20:
            return {
                "type": "suggestion",
                "title": "Good Savings Progress",
                "description": f"{savings_rate:.1f}% savings rate - room for improvement",
                "action": "Aim for 20% savings rate for optimal financial health"
            }
        
        return None
    
    def _analyze_emergency_fund(self, expenses: Dict[str, float], net_savings: float) -> Dict[str, str]:
        """Analyze emergency fund adequacy"""
        
        monthly_expenses = sum(expenses.values())
        emergency_target = monthly_expenses * 6
        
        if net_savings <= 0:
            return {
                "type": "critical",
                "title": "Emergency Fund Priority",
                "description": "No available funds for emergency savings",
                "action": "Create budget surplus first, then build emergency fund"
            }
        
        months_to_target = emergency_target / net_savings if net_savings > 0 else 999
        
        if months_to_target > 24:
            return {
                "type": "warning",
                "title": "Emergency Fund Planning",
                "description": f"Will take {months_to_target:.0f} months to build full fund",
                "action": "Consider allocating more to emergency savings initially"
            }
        elif months_to_target > 12:
            return {
                "type": "suggestion",
                "title": "Emergency Fund Timeline", 
                "description": f"On track to build emergency fund in {months_to_target:.0f} months",
                "action": "Stay consistent with current savings plan"
            }
        
        return None

class FinancialGoalPlanner:
    """Plan and track financial goals"""
    
    def __init__(self):
        pass
    
    def calculate_goal_timeline(self, goal_amount: float, monthly_savings: float,
                              current_savings: float = 0, annual_return: float = 0.07) -> Dict[str, Any]:
        """Calculate timeline to reach financial goal"""
        
        if monthly_savings <= 0:
            return {
                "error": "Monthly savings must be positive",
                "timeline_years": "∞",
                "total_contributions": 0,
                "total_growth": 0
            }
        
        remaining_amount = max(0, goal_amount - current_savings)
        
        if annual_return <= 0:
            # Simple savings without investment growth
            months_needed = remaining_amount / monthly_savings
            total_contributions = remaining_amount
            total_growth = 0
        else:
            # Calculate with compound interest
            monthly_rate = annual_return / 12
            
            # Future value of ordinary annuity formula
            if remaining_amount <= current_savings:
                months_needed = 0
            else:
                # Solve for n in: FV = PMT * [((1 + r)^n - 1) / r] + PV * (1 + r)^n
                months_needed = math.log(1 + (remaining_amount * monthly_rate) / monthly_savings) / math.log(1 + monthly_rate)
            
            total_contributions = monthly_savings * months_needed
            total_growth = goal_amount - current_savings - total_contributions
        
        timeline_years = months_needed / 12
        
        return {
            "timeline_months": months_needed,
            "timeline_years": timeline_years,
            "total_contributions": total_contributions,
            "total_growth": max(0, total_growth),
            "monthly_required": monthly_savings,
            "feasibility": "achievable" if timeline_years <= 10 else "long_term" if timeline_years <= 20 else "challenging"
        }
    
    def optimize_savings_allocation(self, monthly_surplus: float, goals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize allocation across multiple financial goals"""
        
        if monthly_surplus <= 0:
            return {"error": "No surplus available for goal allocation"}
        
        # Priority-based allocation
        priority_weights = {
            "emergency_fund": 10,      # Highest priority
            "debt_payoff": 9,          # Very high priority
            "retirement": 8,           # High priority
            "house_down_payment": 6,   # Medium-high priority
            "vacation": 3,             # Lower priority
            "general_savings": 5       # Medium priority
        }
        
        allocated_amounts = {}
        total_weight = sum(priority_weights.get(goal["type"], 5) for goal in goals)
        
        for goal in goals:
            weight = priority_weights.get(goal["type"], 5)
            allocation_percentage = weight / total_weight
            allocated_amount = monthly_surplus * allocation_percentage
            
            allocated_amounts[goal["name"]] = {
                "monthly_allocation": allocated_amount,
                "percentage": allocation_percentage * 100,
                "priority": weight
            }
        
        return {
            "total_surplus": monthly_surplus,
            "allocations": allocated_amounts,
            "strategy": "priority_weighted"
        }

class FinancialEducator:
    """Provide financial education and insights"""
    
    def __init__(self):
        self.financial_concepts = self._load_financial_concepts()
    
    def _load_financial_concepts(self) -> Dict[str, Dict[str, str]]:
        """Load financial education content"""
        return {
            "compound_interest": {
                "definition": "Interest earned on both principal and previously earned interest",
                "importance": "Makes your money grow exponentially over time",
                "example": "$1000 at 7% for 30 years becomes $7,612"
            },
            "diversification": {
                "definition": "Spreading investments across different assets to reduce risk",
                "importance": "Don't put all eggs in one basket",
                "example": "Mix of stocks, bonds, real estate, international investments"
            },
            "dollar_cost_averaging": {
                "definition": "Investing fixed amounts regularly regardless of market price",
                "importance": "Reduces impact of market volatility",
                "example": "Investing $500/month in index funds regardless of price"
            },
            "emergency_fund": {
                "definition": "Savings reserved for unexpected expenses or income loss",
                "importance": "Prevents debt during financial emergencies", 
                "example": "6 months of expenses in high-yield savings account"
            }
        }
    
    def get_concept_explanation(self, concept: str) -> Dict[str, str]:
        """Get explanation of financial concept"""
        concept_key = concept.lower().replace(" ", "_")
        return self.financial_concepts.get(concept_key, {
            "definition": "Financial concept not found",
            "importance": "Please try a different term",
            "example": "Available concepts: compound interest, diversification, dollar cost averaging"
        })
    
    def generate_learning_path(self, user_level: str, financial_goals: List[str]) -> List[Dict[str, str]]:
        """Generate personalized learning path"""
        
        beginner_path = [
            {"topic": "Budgeting Basics", "priority": 1, "description": "Learn to track income and expenses"},
            {"topic": "Emergency Fund", "priority": 2, "description": "Build financial safety net"},
            {"topic": "Debt Management", "priority": 3, "description": "Strategies to pay off debt efficiently"},
            {"topic": "Basic Investing", "priority": 4, "description": "Introduction to stocks, bonds, and index funds"}
        ]
        
        intermediate_path = [
            {"topic": "Investment Allocation", "priority": 1, "description": "Asset allocation strategies"},
            {"topic": "Tax Optimization", "priority": 2, "description": "Tax-advantaged accounts and strategies"},
            {"topic": "Retirement Planning", "priority": 3, "description": "401k, IRA, and retirement calculations"},
            {"topic": "Real Estate", "priority": 4, "description": "Home buying and real estate investing"}
        ]
        
        advanced_path = [
            {"topic": "Advanced Investing", "priority": 1, "description": "Options, REITs, international investing"},
            {"topic": "Estate Planning", "priority": 2, "description": "Wills, trusts, and inheritance planning"},
            {"topic": "Business Finance", "priority": 3, "description": "Entrepreneurship and business investing"},
            {"topic": "Alternative Investments", "priority": 4, "description": "Crypto, commodities, private equity"}
        ]
        
        if user_level.lower() == "beginner":
            return beginner_path
        elif user_level.lower() == "intermediate":
            return intermediate_path
        else:
            return advanced_path

# Utility functions for the main app
def calculate_fire_number(annual_expenses: float, withdrawal_rate: float = 0.04) -> float:
    """Calculate Financial Independence Retire Early (FIRE) number"""
    return annual_expenses / withdrawal_rate

def calculate_savings_needed_for_goal(goal_amount: float, months: int, annual_return: float = 0.07) -> float:
    """Calculate monthly savings needed to reach goal"""
    if months <= 0:
        return goal_amount
    
    monthly_rate = annual_return / 12
    
    if monthly_rate == 0:
        return goal_amount / months
    
    # PMT calculation for future value
    return goal_amount * monthly_rate / ((1 + monthly_rate) ** months - 1)

def format_currency(amount: float, currency_symbol: str = "$") -> str:
    """Format currency with proper comma separation"""
    return f"{currency_symbol}{amount:,.0f}"

def calculate_net_worth(assets: Dict[str, float], liabilities: Dict[str, float]) -> float:
    """Calculate net worth (assets - liabilities)"""
    total_assets = sum(assets.values())
    total_liabilities = sum(liabilities.values())
    return total_assets - total_liabilities