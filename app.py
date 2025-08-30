import streamlit as st
from transformers import pipeline
import joblib
import numpy as np
import pandas as pd
import sqlite3
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import random

# --- Page Configuration ---
st.set_page_config(
    page_title="SecureBank AI Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS with Vizhibot Animations ---
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #1e3a8a; }
    .stButton>button { width: 100%; border-radius: 8px; }
    .stAlert { border-radius: 10px; }
    .transaction-box { border: 1px solid #ccc; padding: 20px; border-radius: 10px; margin: 10px 0; }
    .risk-low { color: #00cc00; font-weight: bold; }
    .risk-medium { color: #ff9900; font-weight: bold; }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .xai-breakdown { background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0; }
    
    /* Vizhibot Animation Styles */
    .vizhibot-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    
    .vizhibot {
        font-size: 3rem;
        transition: all 0.3s ease;
        position: relative;
    }
    
    /* Animations */
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes thinking {
        0%, 100% { transform: rotate(0deg); }
        25% { transform: rotate(5deg); }
        75% { transform: rotate(-5deg); }
    }
    
    @keyframes celebrate {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.2); }
    }
    
    @keyframes worried {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
    @keyframes processing {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .vizhibot-bounce { animation: bounce 1s infinite; }
    .vizhibot-thinking { animation: thinking 1.5s infinite; }
    .vizhibot-celebrate { animation: celebrate 0.5s infinite; }
    .vizhibot-worried { animation: worried 0.8s infinite; }
    .vizhibot-processing { animation: processing 2s linear infinite; }
    
    .vizhibot-message {
        background: #f0f2f6;
        padding: 12px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #1e3a8a;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown('<h1 class="main-header">🏦 SecureBank AI Assistant</h1>', unsafe_allow_html=True)
st.markdown("### Your 24/7 Banking Support with Advanced Fraud Protection")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "transaction_history" not in st.session_state:
    st.session_state.transaction_history = []

if "alerts" not in st.session_state:
    st.session_state.alerts = []

# --- Vizhibot Component ---
class Vizhibot:
    def __init__(self):
        self.state = "idle"
        self.last_animation = datetime.now()
    
    def display(self, state="idle", message=None):
        """Display Vizhibot with specified animation state"""
        self.state = state
        self.last_animation = datetime.now()
        
        # Map states to emojis and animations
        states = {
            "idle": {"emoji": "🤖", "animation": "vizhibot-bounce", "title": "Vizhibot is ready!"},
            "thinking": {"emoji": "🤔", "animation": "vizhibot-thinking", "title": "Vizhibot is thinking..."},
            "processing": {"emoji": "⚙️", "animation": "vizhibot-processing", "title": "Processing your request"},
            "success": {"emoji": "✅", "animation": "vizhibot-celebrate", "title": "Success!"},
            "warning": {"emoji": "⚠️", "animation": "vizhibot-worried", "title": "Warning detected"},
            "error": {"emoji": "❌", "animation": "vizhibot-worried", "title": "Error occurred"},
            "celebrate": {"emoji": "🎉", "animation": "vizhibot-celebrate", "title": "Celebration time!"},
            "analyzing": {"emoji": "🔍", "animation": "vizhibot-thinking", "title": "Analyzing patterns..."},
            "security": {"emoji": "🛡️", "animation": "vizhibot-bounce", "title": "Security check"},
            "money": {"emoji": "💰", "animation": "vizhibot-celebrate", "title": "Money matters!"}
        }
        
        current_state = states.get(state, states["idle"])
        
        # Create Vizhibot display
        vizhibot_html = f"""
        <div class="vizhibot-container">
            <div class="vizhibot {current_state['animation']}" title="{current_state['title']}">
                {current_state['emoji']}
            </div>
        </div>
        """
        
        if message:
            vizhibot_html += f'<div class="vizhibot-message">💬 {message}</div>'
        
        st.markdown(vizhibot_html, unsafe_allow_html=True)
        return current_state
    
    def random_idle_animation(self):
        """Random idle animation to keep Vizhibot lively"""
        if (datetime.now() - self.last_animation).seconds > 10:
            animations = ["bounce", "thinking", "celebrate"]
            random_anim = random.choice(animations)
            self.display(random_anim)
    
    def chat_response(self, message):
        """Vizhibot responds in chat with appropriate animation"""
        responses = {
            "balance": ("💰", "Looking up your account balance..."),
            "loan": ("🤔", "Checking loan options for you..."),
            "fraud": ("🛡️", "Investigating security matters..."),
            "transaction": ("🔍", "Analyzing transaction patterns..."),
            "help": ("🤖", "I'm here to help! What do you need?"),
            "thank": ("😊", "You're welcome! Happy to assist!"),
            "hello": ("👋", "Hello! I'm Vizhibot, your banking assistant!"),
            "bye": ("👋", "Goodbye! Stay secure!")
        }
        
        for key, (emoji, response) in responses.items():
            if key in message.lower():
                return (emoji, response)
        
        return ("🤖", "I'm analyzing your request...")

# Initialize Vizhibot
if "vizhibot" not in st.session_state:
    st.session_state.vizhibot = Vizhibot()

# --- Known Fraudulent Merchants Database ---
FRAUDULENT_MERCHANTS = {
    "QuickCash Now": {"risk_score": 0.8, "reports": 42, "last_reported": "2025-08-15"},
    "CryptoInvest Pro": {"risk_score": 0.7, "reports": 28, "last_reported": "2025-08-20"},
    "Global Deals LLC": {"risk_score": 0.9, "reports": 57, "last_reported": "2025-08-10"},
    "Discount Electronics": {"risk_score": 0.6, "reports": 19, "last_reported": "2025-08-18"},
    "Luxury Watches Direct": {"risk_score": 0.75, "reports": 35, "last_reported": "2025-08-22"}
}

# --- Unique Feature 1: Spending Profile AI ---
class SpendingProfile:
    def __init__(self):
        self.transaction_history = []
        self.weekly_spending = 0
        self.typical_spending = {
            'morning': {'amount': 1500, 'count': 0},  # In rupees
            'afternoon': {'amount': 2500, 'count': 0},
            'evening': {'amount': 3500, 'count': 0},
            'night': {'amount': 2000, 'count': 0}
        }
        self.category_patterns = {}
        self.last_updated = datetime.now()
    
    def learn_from_transaction(self, amount, category, hour):
        """AI learns from each transaction to build spending profile"""
        transaction = {
            'amount': amount,
            'category': category,
            'hour': hour,
            'timestamp': datetime.now(),
            'day_of_week': datetime.now().weekday()
        }
        self.transaction_history.append(transaction)
        
        # Update time patterns
        time_period = self._get_time_period(hour)
        self.typical_spending[time_period]['amount'] = (
            self.typical_spending[time_period]['amount'] + amount
        ) / 2
        self.typical_spending[time_period]['count'] += 1
        
        # Update category patterns
        if category not in self.category_patterns:
            self.category_patterns[category] = {'total': 0, 'count': 0}
        self.category_patterns[category]['total'] += amount
        self.category_patterns[category]['count'] += 1
        
        # Update weekly spending (last 7 days only)
        one_week_ago = datetime.now() - timedelta(days=7)
        self.weekly_spending = sum(t['amount'] for t in self.transaction_history 
                                 if t['timestamp'] > one_week_ago)
    
    def _get_time_period(self, hour):
        if 5 <= hour < 12: return 'morning'
        elif 12 <= hour < 17: return 'afternoon'
        elif 17 <= hour < 22: return 'evening'
        else: return 'night'
    
    def calculate_behavior_risk(self, amount, category, hour):
        """Calculate how unusual this transaction is for this user"""
        risks = []
        risk_factors = []
        
        # 1. Time pattern risk
        time_period = self._get_time_period(hour)
        typical_amount = self.typical_spending[time_period]['amount']
        if typical_amount > 0:
            amount_ratio = amount / typical_amount
            if amount_ratio > 3.0:
                risk_score = min(0.6 + (amount_ratio - 3.0) * 0.1, 0.9)
                risks.append(risk_score)
                risk_factors.append((
                    f"Amount is {amount_ratio:.1f}x your {time_period} average", 
                    risk_score
                ))
            elif amount_ratio > 2.0:
                risk_score = min(0.4 + (amount_ratio - 2.0) * 0.2, 0.6)
                risks.append(risk_score)
                risk_factors.append((
                    f"Large amount for {time_period} ({amount_ratio:.1f}x average)", 
                    risk_score
                ))
        
        # 2. Category risk
        if category in self.category_patterns:
            avg_category = self.category_patterns[category]['total'] / self.category_patterns[category]['count']
            if amount > avg_category * 4:
                risk_score = min(0.7 + (amount / avg_category - 4) * 0.05, 0.9)
                risks.append(risk_score)
                risk_factors.append((
                    f"Unusually large for {category} ({amount/avg_category:.1f}x average)", 
                    risk_score
                ))
            elif amount > avg_category * 2:
                risk_score = min(0.4 + (amount / avg_category - 2) * 0.15, 0.7)
                risks.append(risk_score)
                risk_factors.append((
                    f"Large for {category} ({amount/avg_category:.1f}x average)", 
                    risk_score
                ))
        else:
            risks.append(0.3)
            risk_factors.append(("New spending category", 0.3))
        
        # 3. Weekly budget risk
        if self.weekly_spending + amount > 50000:  # Weekly limit in rupees
            overspend_ratio = (self.weekly_spending + amount) / 50000
            risk_score = min(0.5 + (overspend_ratio - 1) * 0.5, 0.9)
            risks.append(risk_score)
            risk_factors.append((
                f"Exceeds weekly spending pattern ({overspend_ratio:.1f}x limit)", 
                risk_score
            ))
        
        # Calculate overall risk score
        if not risks:
            return 0.1, ["Normal spending pattern"], []
        
        # Use maximum risk with weighted average
        max_risk = max(risks) if risks else 0
        avg_risk = sum(risks) / len(risks) if risks else 0
        total_risk = max_risk * 0.7 + avg_risk * 0.3
        
        return min(total_risk, 0.95), risk_factors

# Initialize user spending profile
if "spending_profile" not in st.session_state:
    st.session_state.spending_profile = SpendingProfile()
    # Pre-train with some typical spending
    typical_transactions = [
        (2500, "Restaurant", 19),
        (5000, "Online Shopping", 20),
        (1500, "Coffee", 8),
        (8000, "Electronics", 15),
        (2000, "Groceries", 18),
        (1000, "Transport", 9),
        (4000, "Entertainment", 21)
    ]
    for amount, category, hour in typical_transactions:
        st.session_state.spending_profile.learn_from_transaction(amount, category, hour)

# --- Sidebar for Navigation ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3698/3698156.png", width=100)
    st.title("Navigation")
    app_mode = st.radio("Choose a module:", 
                       ["💬 Chat Support", "🔍 Fraud Detection", "💳 Make Payment", "📊 Account Dashboard"])
    
    st.divider()
    st.header("Account Overview")
    st.metric("Available Balance", "₹82,450")
    st.metric("Total Assets", "₹3,45,320")
    if hasattr(st.session_state, 'spending_profile'):
        st.metric("Weekly Spending", f"₹{st.session_state.spending_profile.weekly_spending:,.0f}")
    
    # Alert notifications
    if st.session_state.alerts:
        st.divider()
        st.header("🔔 Alerts")
        for i, alert in enumerate(st.session_state.alerts[-3:]):  # Show last 3 alerts
            st.warning(f"{alert['type']}: {alert['message']}")
            if st.button(f"Dismiss", key=f"dismiss_{i}"):
                st.session_state.alerts.pop(i)
                st.rerun()

# --- Load AI Models ---
@st.cache_resource
def load_models():
    # Load QA Model
    try:
        qa_model = pipeline("question-answering")
    except:
        qa_model = None
    
    # Load Fraud Model
    try:
        fraud_model = joblib.load('fraud_model.pkl')
    except:
        fraud_model = None
    
    return qa_model, fraud_model

qa_model, fraud_model = load_models()

# --- Banking Knowledge Base ---
banking_context = """SecureBank offers comprehensive financial services including savings accounts, loans, and investment products.
Our customer support is available 24/7 at 1-800-SECURE-BANK (1-800-732-8732).
Personal loan interest rates start at 8.5% APR based on creditworthiness.
Savings accounts earn 3.2% APY with a minimum balance of ₹5000.
To apply for any loan product, you'll need government-issued ID, proof of address, and recent income verification.
The daily ATM withdrawal limit is ₹50,000 for most account types.
The maximum mobile check deposit is ₹25,000 per day.
If you suspect fraudulent activity, immediately call our security team at 1-800-732-8732.
Fixed term deposits require a minimum of ₹25,000 for 12-month terms earning 4.5% APY.
We have over 200 branches across major Indian cities."""

# --- Vizhibot Introduction and Capabilities ---
def get_vizhibot_introduction():
    """Return Vizhibot's introduction and capabilities"""
    return """
    **Namaste! I'm Vizhibot, your AI-powered banking assistant.** 👁️

    I'm here to help you with all your banking needs. Here's what I can do:

    💬 **Answer Questions**: I can help with account information, loan details, interest rates, and banking policies.

    🔍 **Fraud Detection**: I analyze transactions in real-time to protect you from fraudulent activities.

    💳 **Payment Monitoring**: I assess risks before payments are processed to keep your money safe.

    📊 **Financial Insights**: I provide spending analysis and help you understand your financial patterns.

    🛡️ **Security Alerts**: I notify you immediately of any suspicious activities on your account.

    How can I assist you today?
    """

# --- Enhanced Fraud Detection Function with XAI ---
def analyze_transaction(amount, hour, merchant):
    """Analyze transaction with explainable AI components"""
    risk_factors = []
    total_risk = 0
    
    # 1. Check against known fraudulent merchants
    if merchant in FRAUDULENT_MERCHANTS:
        merchant_risk = FRAUDULENT_MERCHANTS[merchant]["risk_score"]
        risk_factors.append((
            f"Known risky merchant ({merchant})", 
            merchant_risk,
            f"This merchant has {FRAUDULENT_MERCHANTS[merchant]['reports']} fraud reports"
        ))
        total_risk = max(total_risk, merchant_risk)
    
    # 2. ML model prediction
    if fraud_model is not None:
        try:
            amount_to_hour_ratio = amount / (hour + 1)
            input_features = np.array([[amount, hour, amount_to_hour_ratio]])
            
            prediction = fraud_model.predict(input_features)[0]
            probability = fraud_model.predict_proba(input_features)[0][1]
            
            if probability > 0.3:
                risk_factors.append((
                    "ML Model Detection", 
                    probability,
                    "AI model identified suspicious patterns"
                ))
                total_risk = max(total_risk, probability)
        except Exception as e:
            risk_factors.append(("Model Error", 0.2, f"Prediction error: {str(e)}"))
    
    # 3. Rule-based checks
    if amount > 15000:  # High amount threshold in rupees
        amount_risk = min(0.3 + (amount - 15000) / 50000, 0.8)
        risk_factors.append((
            f"High amount (₹{amount:,.0f})", 
            amount_risk,
            "Transaction amount is significantly above average"
        ))
        total_risk = max(total_risk, amount_risk)
    
    if hour < 6 or hour > 22:
        time_risk = 0.5 if hour < 6 else 0.4
        risk_factors.append((
            f"Unusual time ({hour}:00)", 
            time_risk,
            "Transaction occurred during non-typical hours"
        ))
        total_risk = max(total_risk, time_risk)
    
    # Calculate combined risk (weighted towards highest risk factor)
    if risk_factors:
        # Use maximum risk with contribution from other factors
        sorted_factors = sorted(risk_factors, key=lambda x: x[1], reverse=True)
        total_risk = sorted_factors[0][1] * 0.7
        if len(sorted_factors) > 1:
            total_risk += sum(f[1] for f in sorted_factors[1:]) / len(sorted_factors[1:]) * 0.3
        
        total_risk = min(total_risk, 0.95)
    
    return total_risk, risk_factors

# --- Unique Feature 2: Predictive Risk Engine ---
def predict_payment_risk(amount, category, hour, merchant):
    """Predict risk before payment is processed"""
    # Get behavioral risk from spending profile
    behavior_risk, behavior_factors = st.session_state.spending_profile.calculate_behavior_risk(amount, category, hour)
    
    # Get fraud model risk
    fraud_risk, fraud_factors = analyze_transaction(amount, hour, merchant)
    
    # Combine all risk factors
    all_factors = []
    
    # Add behavioral factors
    for factor, score in behavior_factors:
        all_factors.append((f"Behavior: {factor}", score * 0.7, "Based on your spending patterns"))
    
    # Add fraud factors
    for factor, score, explanation in fraud_factors:
        all_factors.append((factor, score, explanation))
    
    # Calculate combined risk (weighted average)
    if all_factors:
        total_risk = sum(score for _, score, _ in all_factors) / len(all_factors)
    else:
        total_risk = 0.1
    
    return total_risk, all_factors

# --- Anomaly Detection Function ---
def detect_spending_anomalies():
    """Detect anomalies in spending patterns"""
    if not hasattr(st.session_state, 'spending_profile') or not st.session_state.spending_profile.transaction_history:
        return [], []
    
    # Get recent transactions (last 30 days)
    thirty_days_ago = datetime.now() - timedelta(days=30)
    recent_transactions = [
        t for t in st.session_state.spending_profile.transaction_history 
        if t['timestamp'] > thirty_days_ago
    ]
    
    if len(recent_transactions) < 5:
        return [], []  # Not enough data
    
    amounts = [t['amount'] for t in recent_transactions]
    mean = np.mean(amounts)
    std = np.std(amounts)
    
    # Find anomalies (more than 2 standard deviations from mean)
    anomalies = []
    for i, transaction in enumerate(recent_transactions):
        z_score = abs(transaction['amount'] - mean) / std if std > 0 else 0
        if z_score > 2.0:
            anomalies.append({
                'index': i,
                'transaction': transaction,
                'z_score': z_score,
                'deviation': f"{z_score:.1f} standard deviations from mean"
            })
    
    return recent_transactions, anomalies

# --- Alert System ---
def add_alert(alert_type, message):
    """Add an alert to the system"""
    alert = {
        'type': alert_type,
        'message': message,
        'timestamp': datetime.now(),
        'read': False
    }
    st.session_state.alerts.append(alert)
    
    # Simulate email/SMS (in a real app, this would call an API)
    if alert_type == "Fraud Alert":
        st.toast(f"📧 Email sent: {message}")
        st.toast(f"📱 SMS sent: {message}")

# --- Main Application Logic ---
# Display Vizhibot based on app mode
if app_mode == "💬 Chat Support":
    st.session_state.vizhibot.display("idle", "Ready to chat with you!")
elif app_mode == "🔍 Fraud Detection":
    st.session_state.vizhibot.display("security", "Monitoring for suspicious activity...")
elif app_mode == "💳 Make Payment":
    st.session_state.vizhibot.display("processing", "Securing your transaction...")
else:
    st.session_state.vizhibot.display("money", "Analyzing your financial health...")

# Now handle the actual app mode content
if app_mode == "💬 Chat Support":
    st.header("💬 AI Customer Support")
    
    # Display Vizhibot introduction at the start
    if not st.session_state.chat_history:
        intro_message = get_vizhibot_introduction()
        st.session_state.chat_history.append(("assistant", intro_message))
    
    # Display chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "user":
            st.chat_message("user").markdown(f"**You:** {message}")
        else:
            st.chat_message("assistant").markdown(f"**Vizhibot:** {message}")
    
    # Chat input
    if prompt := st.chat_input("Ask me about banking, fraud detection, or account help..."):
        # Show Vizhibot thinking
        st.session_state.vizhibot.display("thinking", "Processing your question...")
        
        # Add to chat history
        st.session_state.chat_history.append(("user", prompt))
        
        # Get Vizhibot's response animation
        emoji, vizhi_response = st.session_state.vizhibot.chat_response(prompt)
        
        # Convert to lowercase for easier matching
        prompt_lower = prompt.lower()
        
        # Handle specific queries about the bot
        if any(word in prompt_lower for word in ["who are you", "what is your name", "your name", "introduce yourself"]):
            response = get_vizhibot_introduction()
            st.session_state.vizhibot.display("celebrate", "That's me!")
        elif any(word in prompt_lower for word in ["balance", "amount", "money"]):
            response = "Your current available balance is **₹82,450**. Would you like to see recent transactions?"
            st.session_state.vizhibot.display("money", "Balance checked!")
        elif any(word in prompt_lower for word in ["loan", "borrow", "interest"]):
            response = "We offer personal loans starting at **8.5% APR**. Would you like to check your eligibility?"
            st.session_state.vizhibot.display("thinking", "Checking loan options...")
        elif any(word in prompt_lower for word in ["fraud", "stolen", "unauthorized"]):
            response = "For immediate fraud assistance, please call our 24/7 security line at **1-800-732-8732**. Would you like me to help you analyze a suspicious transaction?"
            st.session_state.vizhibot.display("security", "Security matters!")
        else:
            # Use AI for other questions
            if qa_model:
                try:
                    result = qa_model(question=prompt, context=banking_context)
                    response = result['answer'] if result['score'] > 0.2 else "I'm not sure about that. Please contact our support team for assistance."
                    st.session_state.vizhibot.display("thinking", "Found an answer for you!")
                except:
                    response = "I'm having trouble connecting to the knowledge base. Please try again or call customer support."
                    st.session_state.vizhibot.display("error", "Having some trouble...")
            else:
                response = "I'm currently learning about banking services. Please contact customer support for detailed assistance."
                st.session_state.vizhibot.display("idle", "Still learning...")
        
        st.session_state.chat_history.append(("assistant", response))
        st.rerun()

elif app_mode == "🔍 Fraud Detection":
    st.header("🔍 Advanced Fraud Detection")
    st.warning("This AI-powered system analyzes transactions for suspicious patterns in real-time.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Analyze a Transaction")
        transaction_amount = st.number_input("Transaction Amount (₹)", min_value=0.0, value=2500.0, step=500.0)
        transaction_hour = st.slider("Time of Day (Hour)", 0, 23, 14)
        merchant_name = st.text_input("Merchant Name", "Online Store")
        
        if st.button("🔎 Analyze for Fraud", type="primary"):
            st.session_state.vizhibot.display("analyzing", "Scanning transaction patterns...")
            risk_score, risk_factors = analyze_transaction(transaction_amount, transaction_hour, merchant_name)
            
            # Add to transaction history
            transaction_data = {
                "amount": transaction_amount,
                "time": transaction_hour,
                "merchant": merchant_name,
                "fraud_risk": risk_score,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "result": "High Risk" if risk_score > 0.7 else "Medium Risk" if risk_score > 0.4 else "Low Risk"
            }
            st.session_state.transaction_history.append(transaction_data)
            
            # Display results
            if risk_score > 0.7:
                st.error(f"🚨 **High Fraud Risk Detected!** ({risk_score:.1%} confidence)")
                st.session_state.vizhibot.display("warning", "High risk detected!")
                # Add alert
                add_alert("Fraud Alert", f"High risk transaction: ₹{transaction_amount:,.0f} at {merchant_name}")
                
            elif risk_score > 0.4:
                st.warning(f"⚠️ **Suspicious Transaction** ({risk_score:.1%} risk)")
                st.session_state.vizhibot.display("worried", "Suspicious activity found")
            else:
                st.success(f"✅ **Transaction Appears Safe** ({risk_score:.1%} risk)")
                st.session_state.vizhibot.display("success", "Transaction looks safe!")
            
            # Show XAI breakdown
            if risk_factors:
                st.subheader("Risk Breakdown")
                st.markdown('<div class="xai-breakdown">', unsafe_allow_html=True)
                
                for factor, score, explanation in risk_factors:
                    st.write(f"**{factor}** ({score:.0%} risk)")
                    st.caption(f"{explanation}")
                    st.progress(score)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Recent Analyses")
        if st.session_state.transaction_history:
            for i, transaction in enumerate(reversed(st.session_state.transaction_history[-5:])):
                with st.container():
                    risk_color = "#ff4b4b" if transaction["fraud_risk"] > 0.7 else "#ff9900" if transaction["fraud_risk"] > 0.4 else "#00cc00"
                    st.markdown(f"""
                    <div class="transaction-box">
                        <strong>₹{transaction['amount']:,.0f}</strong> at {transaction['time']}:00 | 
                        <span style="color: {risk_color};"><strong>{transaction['result']}</strong></span><br>
                        <small>{transaction['timestamp']} | {transaction['merchant']}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No transactions analyzed yet. Use the panel to check a transaction.")
        
        # Known fraudulent merchants
        st.subheader("Known Risky Merchants")
        for merchant, info in list(FRAUDULENT_MERCHANTS.items())[:3]:
            st.write(f"**{merchant}** ({info['reports']} reports)")
            st.progress(info['risk_score'])

elif app_mode == "💳 Make Payment":
    st.header("💳 AI-Powered Payment Gateway")
    st.success("Your AI Financial Guardian is monitoring this transaction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Payment Details")
        
        # Payment form
        merchant = st.selectbox("Merchant", ["Amazon", "Flipkart", "Apple Store", "Local Supermarket", 
                                           "Uber", "MakeMyTrip", "Luxury Boutique", "Travel Agency", "QuickCash Now"])
        category = st.selectbox("Category", ["Shopping", "Food & Drink", "Entertainment", "Travel", 
                                           "Electronics", "Luxury Goods", "Services"])
        amount = st.number_input("Amount (₹)", min_value=0.01, value=2500.0, step=500.0)
        hour = st.slider("Transaction Time", 0, 23, datetime.now().hour)
        
        if st.button("🚀 Process Payment", type="primary", use_container_width=True):
            st.session_state.vizhibot.display("processing", "Analyzing transaction...")
            # Get AI risk prediction
            risk_score, risk_factors = predict_payment_risk(amount, category, hour, merchant)
            
            # Display results
            st.divider()
            st.subheader("AI Risk Assessment")
            
            # Risk meter
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Risk Score", f"{risk_score:.0%}")
            with col_b:
                st.progress(risk_score)
            with col_c:
                if risk_score < 0.3:
                    st.success("✅ Low Risk")
                    st.session_state.vizhibot.display("success", "Low risk transaction!")
                elif risk_score < 0.7:
                    st.warning("⚠️ Medium Risk")
                    st.session_state.vizhibot.display("warning", "Medium risk detected")
                else:
                    st.error("🚨 High Risk")
                    st.session_state.vizhibot.display("error", "High risk detected!")
            
            # Risk reasons with XAI breakdown
            if risk_factors:
                st.subheader("Risk Factors")
                st.markdown('<div class="xai-breakdown">', unsafe_allow_html=True)
                
                for factor, score, explanation in risk_factors:
                    st.write(f"**{factor}** ({score:.0%} risk)")
                    st.caption(f"{explanation}")
                    st.progress(score)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Action based on risk level
            if risk_score > 0.7:
                st.error("""
                **🚫 Payment Blocked**
                This transaction poses significant risk to your account.
                Please contact your bank or use a different payment method.
                """)
                add_alert("Payment Blocked", f"Blocked payment: ₹{amount:,.0f} to {merchant}")
                
            elif risk_score > 0.4:
                st.warning("""
                **⚠️ Additional Verification Required**
                Please complete two-factor authentication to proceed.
                """)
                # 2FA simulation
                verification = st.text_input("Enter verification code sent to your phone", 
                                           placeholder="123456")
                if st.button("Verify & Complete Payment"):
                    if verification == "123456":
                        st.success("✅ Payment successful!")
                        st.session_state.vizhibot.display("celebrate", "Payment approved!")
                        # Learn from this transaction
                        st.session_state.spending_profile.learn_from_transaction(amount, category, hour)
                    else:
                        st.error("❌ Invalid verification code")
                        st.session_state.vizhibot.display("error", "Verification failed")
            else:
                st.success("""
                **✅ Payment Approved**
                Transaction completed successfully.
                """)
                st.session_state.vizhibot.display("celebrate", "Payment successful!")
                # Learn from this transaction
                st.session_state.spending_profile.learn_from_transaction(amount, category, hour)
    
    with col2:
        st.subheader("Your Spending Profile")
        if hasattr(st.session_state.spending_profile, 'weekly_spending'):
            st.metric("Weekly Spending", f"₹{st.session_state.spending_profile.weekly_spending:,.0f}")
            st.metric("Transaction History", f"{len(st.session_state.spending_profile.transaction_history)}")
        
        st.info("""
        **AI Guardian is analyzing:**
        - Your spending patterns
        - Time of transaction
        - Merchant category
        - Amount consistency
        - Fraud database patterns
        """)

else:  # Account Dashboard
    st.header("📊 Account Dashboard")
    
    # Mock account data
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Checking Balance", "₹62,450", "+₹12,245 this month")
    with col2:
        st.metric("Savings Balance", "₹1,25,872", "+₹5,780 this month")
    with col3:
        st.metric("Credit Card", "₹12,345", "-₹4,567 this month")
    
    # Spending analysis with anomaly detection
    st.subheader("Spending Analysis")
    recent_transactions, anomalies = detect_spending_anomalies()
    
    if recent_transactions:
        # Create spending chart
        dates = [t['timestamp'].strftime("%m-%d") for t in recent_transactions]
        amounts = [t['amount'] for t in recent_transactions]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(dates, amounts, color='skyblue')
        
        # Highlight anomalies
        for anomaly in anomalies:
            if anomaly['index'] < len(bars):
                bars[anomaly['index']].set_color('red')
        
        ax.set_title('Recent Spending (Anomalies in Red)')
        ax.set_ylabel('Amount (₹)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Show anomaly details
        if anomalies:
            st.subheader("Spending Anomalies Detected")
            for anomaly in anomalies:
                t = anomaly['transaction']
                st.warning(f"**₹{t['amount']:,.0f}** on {t['timestamp'].strftime('%m/%d')} at {t['hour']}:00")
                st.caption(f"Category: {t['category']} | {anomaly['deviation']}")
    else:
        st.info("Not enough transaction data for analysis yet.")
    
    # Recent transactions mockup
    st.subheader("Recent Transactions")
    transactions_df = pd.DataFrame({
        "Date": ["2025-08-29", "2025-08-28", "2025-08-27", "2025-08-25", "2025-08-24"],
        "Description": ["Amazon Purchase", "Starbucks Coffee", "Electricity Bill", "Salary Deposit", "Grocery Store"],
        "Amount": ["-₹2,899", "-₹425", "-₹7,450", "+₹45,000", "-₹2,834"],
        "Status": ["Completed", "Completed", "Pending", "Completed", "Completed"]
    })
    st.dataframe(transactions_df, use_container_width=True, hide_index=True)
    
    st.subheader("Financial Health")
    st.progress(0.78)
    st.caption("Credit Score: 780/850 (Excellent)")

# --- Footer ---
st.divider()
st.caption("© 2025 SecureBank - AI Banking Assistant with Vizhibot | All transactions are monitored for your protection")
