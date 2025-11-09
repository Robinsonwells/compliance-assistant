"""
Admin Dashboard for Chat Logs
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import hashlib
import pytz
from chat_logger import ChatLogger

# Initialize logger
logger = ChatLogger()

# Page config
st.set_page_config(
    page_title="Admin Dashboard - Chat Logs",
    page_icon="📊",
    layout="wide"
)

def check_password(password):
    """Check if password matches stored admin password"""
    # Check if we're using hashed password
    stored_hash = os.getenv("ADMIN_PASSWORD_HASH")
    if stored_hash:
        entered_hash = hashlib.sha256(password.encode()).hexdigest()
        return entered_hash == stored_hash
    else:
        # Fallback to plain text comparison
        return password == os.getenv("ADMIN_PASSWORD")

# Authentication
st.title("📊 Admin Dashboard - Chat Logs")

with st.sidebar:
    admin_password = st.text_input("Admin Password", type="password")
    if not check_password(admin_password):
        st.error("Invalid password")
        st.stop()
    st.success("✅ Authenticated")

# Date filters with timezone awareness
timezone = pytz.timezone('America/Denver')  # MST timezone
current_time = datetime.now(timezone)

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "From Date",
        value=(current_time - timedelta(days=7)).date()
    )
with col2:
    end_date = st.date_input(
        "To Date",
        value=current_time.date()
    )

# Fetch logs
logs = logger.get_all_logs(
    limit=1000,
    start_date=datetime.combine(start_date, datetime.min.time()),
    end_date=datetime.combine(end_date, datetime.max.time())
)

if not logs:
    st.warning("No chat logs found for the selected date range.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(logs)
df['created_at'] = pd.to_datetime(df['created_at'])

# Summary Statistics
st.header("📈 Summary Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Queries", len(df))
with col2:
    st.metric("Unique Users", df['access_code'].nunique())
with col3:
    if 'tokens_used' in df and df['tokens_used'].notna().any():
        avg_tokens = int(df['tokens_used'].mean()) if df['tokens_used'].mean() > 0 else 0
        st.metric("Avg Tokens", f"{avg_tokens:,}")
    else:
        st.metric("Avg Tokens", "N/A")
with col4:
    if 'cost_estimate' in df and df['cost_estimate'].notna().any():
        total_cost = df['cost_estimate'].sum()
        st.metric("Total Cost", f"${total_cost:.2f}")
    else:
        st.metric("Total Cost", "N/A")

# Usage by Access Code
st.header("👥 Usage by Access Code")
usage_by_code = df.groupby('access_code').agg({
    'id': 'count',
    'cost_estimate': 'sum',
    'tokens_used': 'sum'
}).rename(columns={
    'id': 'queries', 
    'cost_estimate': 'total_cost',
    'tokens_used': 'total_tokens'
})
usage_by_code = usage_by_code.sort_values('queries', ascending=False)

# Format the display
usage_display = usage_by_code.copy()
usage_display['total_cost'] = usage_display['total_cost'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
usage_display['total_tokens'] = usage_display['total_tokens'].apply(lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "N/A")

st.dataframe(usage_display, use_container_width=True)

# Query Timeline
st.header("📅 Queries Over Time")
df_timeline = df.set_index('created_at').resample('D').size()
st.line_chart(df_timeline)

# Reasoning Effort Distribution
st.header("🧠 Reasoning Effort Distribution")
if 'reasoning_effort' in df and df['reasoning_effort'].notna().any():
    effort_counts = df['reasoning_effort'].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(effort_counts)
    with col2:
        for effort, count in effort_counts.items():
            st.metric(f"{effort.title()} Effort", count)
else:
    st.info("No reasoning effort data available")

# Individual Chat Logs
st.header("💬 Individual Chat Logs")

# Filter by access code
selected_code = st.selectbox(
    "Filter by Access Code (optional)",
    options=["All"] + sorted(df['access_code'].unique().tolist())
)

if selected_code != "All":
    filtered_df = df[df['access_code'] == selected_code]
else:
    filtered_df = df

# Sort by most recent first
filtered_df = filtered_df.sort_values('created_at', ascending=False)

# Display each chat log
st.write(f"Showing {len(filtered_df)} logs")

for idx, row in filtered_df.iterrows():
    # Format timestamp
    timestamp = row['created_at'].strftime("%Y-%m-%d %H:%M:%S")
    
    # Create expander for each log
    with st.expander(
        f"📝 {timestamp} | User: {row['access_code']} | "
        f"{row.get('reasoning_effort', 'N/A')} effort | "
        f"Tokens: {row.get('tokens_used', 'N/A')} | "
        f"Cost: ${row.get('cost_estimate', 0):.4f}"
    ):
        st.markdown("### 🙋 User Query")
        st.write(row['user_query'])
        
        st.markdown("### 🤖 GPT-5 Response")
        if row['gpt5_response']:
            # Check if response was truncated
            if row['gpt5_response'].endswith("... [TRUNCATED]"):
                st.warning("⚠️ Response was truncated due to length")
            st.markdown(row['gpt5_response'])
        else:
            st.info("No GPT-5 response recorded")
        
        st.markdown("### 🔍 Perplexity Audit")
        if row['perplexity_response']:
            # Check if response was truncated
            if row['perplexity_response'].endswith("... [TRUNCATED]"):
                st.warning("⚠️ Audit response was truncated due to length")
            st.markdown(row['perplexity_response'])
        else:
            st.info("No Perplexity response recorded")
        
        # Additional metadata
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.caption(f"**Session ID:** {row.get('session_id', 'N/A')}")
        with col2:
            st.caption(f"**Tokens:** {row.get('tokens_used', 'N/A')}")
        with col3:
            st.caption(f"**Cost:** ${row.get('cost_estimate', 0):.4f}")
        with col4:
            st.caption(f"**Effort:** {row.get('reasoning_effort', 'N/A')}")

# Export functionality
st.header("💾 Export Data")
csv = df.to_csv(index=False)
st.download_button(
    label="📥 Download CSV",
    data=csv,
    file_name=f"chat_logs_{start_date}_to_{end_date}.csv",
    mime="text/csv"
)

# Additional Analytics
st.header("📊 Additional Analytics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Query Types")
    # Simple analysis of query patterns
    if len(df) > 0:
        query_words = []
        for query in df['user_query'].dropna():
            words = query.lower().split()
            # Look for common legal terms
            legal_terms = ['wage', 'overtime', 'break', 'vacation', 'sick', 'leave', 'classification', 'contractor', 'employee']
            for term in legal_terms:
                if term in ' '.join(words):
                    query_words.append(term)
        
        if query_words:
            word_counts = pd.Series(query_words).value_counts().head(10)
            st.bar_chart(word_counts)
        else:
            st.info("No common legal terms detected")

with col2:
    st.subheader("Response Times by Effort")
    # This would require storing response times in the future
    st.info("Response time tracking not yet implemented")

# Footer
st.markdown("---")
st.caption(f"Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("💡 Tip: Use the date filters to analyze specific time periods")