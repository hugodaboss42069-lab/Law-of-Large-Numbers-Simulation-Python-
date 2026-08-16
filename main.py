import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from scipy import stats
from collections import deque
import math

# Page configuration
st.set_page_config(
    page_title="Law of Large Numbers - Advanced Statistical Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFD700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 0.3rem;
    }
    .converged {
        color: #00ff00;
        font-weight: bold;
    }
    .converging {
        color: #ffa500;
        font-weight: bold;
    }
    .far {
        color: #ff4444;
        font-weight: bold;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        transition: 0.2s;
    }
    .insight-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedCoinFlipSimulator:
    def __init__(self):
        self.total_flips = 0
        self.heads = 0
        self.tails = 0
        self.history = []
        self.running = False
        self.max_flips = 1000000
        self.flip_results = []  # Store individual results for advanced analysis
        
    def flip_coin(self, heads_weight, flip_force):
        """Simulate a coin flip with weight and force"""
        force_factor = flip_force / 10.0
        effective_weight = 0.5 + (heads_weight - 0.5) * (1 - force_factor * 0.7)
        noise_scale = 0.05 + (flip_force / 10.0) * 0.1
        noise = np.random.normal(0, noise_scale)
        final_prob = np.clip(effective_weight + noise, 0.01, 0.99)
        return 1 if np.random.random() < final_prob else 0
    
    def run_flips(self, num_flips, heads_weight, flip_force):
        """Run multiple flips and update state"""
        new_heads = 0
        new_tails = 0
        
        for _ in range(num_flips):
            result = self.flip_coin(heads_weight, flip_force)
            self.flip_results.append(result)
            
            if result == 1:
                self.heads += 1
                new_heads += 1
            else:
                self.tails += 1
                new_tails += 1
            self.total_flips += 1
            
            # Store history at intervals
            if self.total_flips % max(1, num_flips // 50) == 0:
                proportion = self.heads / self.total_flips
                std_error = math.sqrt((proportion * (1 - proportion)) / self.total_flips)
                self.history.append({
                    'flips': self.total_flips,
                    'proportion': proportion,
                    'std_error': std_error,
                    'heads': self.heads,
                    'tails': self.tails
                })
        
        return new_heads, new_tails
    
    def get_statistics(self, heads_weight):
        """Calculate comprehensive statistics"""
        if self.total_flips == 0:
            return None
        
        current_prop = self.heads / self.total_flips
        diff = abs(current_prop - heads_weight)
        std_error = math.sqrt((current_prop * (1 - current_prop)) / self.total_flips)
        
        # Z-score
        z_score = (current_prop - heads_weight) / std_error if std_error > 0 else 0
        
        # Confidence interval (95%)
        ci_lower = current_prop - 1.96 * std_error
        ci_upper = current_prop + 1.96 * std_error
        
        # Chi-square test
        expected_heads = self.total_flips * heads_weight
        expected_tails = self.total_flips * (1 - heads_weight)
        chi2 = ((self.heads - expected_heads)**2 / expected_heads + 
                (self.tails - expected_tails)**2 / expected_tails)
        chi2_p_value = 1 - stats.chi2.cdf(chi2, 1)
        
        # Running statistics if we have history
        if len(self.history) > 1:
            proportions = [h['proportion'] for h in self.history]
            std_props = np.std(proportions)
            trend = np.polyfit(range(len(proportions)), proportions, 1)[0]
        else:
            std_props = 0
            trend = 0
        
        # Convergence analysis
        if len(self.history) > 10:
            recent_props = [h['proportion'] for h in self.history[-10:]]
            recent_std = np.std(recent_props)
            convergence_rate = -trend if trend < 0 else 0
        else:
            recent_std = 0
            convergence_rate = 0
        
        # Probability within range
        prob_within_1pct = 0
        prob_within_5pct = 0
        if len(self.history) > 0:
            recent_diffs = [abs(h['proportion'] - heads_weight) for h in self.history[-50:]]
            prob_within_1pct = sum(1 for d in recent_diffs if d < 0.01) / len(recent_diffs) * 100
            prob_within_5pct = sum(1 for d in recent_diffs if d < 0.05) / len(recent_diffs) * 100
        
        # Expected flips to convergence
        expected_flips_to_conv = 0
        if diff > 0.01 and convergence_rate > 0:
            expected_flips_to_conv = (diff - 0.01) / convergence_rate
        
        return {
            'total_flips': self.total_flips,
            'heads': self.heads,
            'tails': self.tails,
            'current_prop': current_prop,
            'expected_prop': heads_weight,
            'diff': diff,
            'std_error': std_error,
            'z_score': z_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'chi2': chi2,
            'chi2_p_value': chi2_p_value,
            'std_props': std_props,
            'trend': trend,
            'recent_std': recent_std,
            'convergence_rate': convergence_rate,
            'prob_within_1pct': prob_within_1pct,
            'prob_within_5pct': prob_within_5pct,
            'expected_flips_to_conv': expected_flips_to_conv
        }

def create_advanced_plots(history, stats, heads_weight):
    """Create enhanced plots with statistical analysis"""
    
    if not history:
        fig = go.Figure()
        fig.add_annotation(text="Run the simulation to see plots", showarrow=False)
        return fig
    
    # Extract data
    flips = [h['flips'] for h in history]
    proportions = [h['proportion'] for h in history]
    std_errors = [h['std_error'] for h in history]
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Running Proportion with Confidence Intervals',
            'Standard Error Over Time',
            'Convergence Analysis',
            'Z-Score Tracking',
            'Distribution of Outcomes',
            'Statistical Summary'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.08,
        horizontal_spacing=0.12
    )
    
    # Plot 1: Running proportion with confidence intervals
    fig.add_trace(
        go.Scatter(
            x=flips,
            y=proportions,
            mode='lines',
            name='Proportion',
            line=dict(color='#4ECDC4', width=2)
        ),
        row=1, col=1
    )
    
    # Add confidence interval
    if stats:
        fig.add_trace(
            go.Scatter(
                x=flips,
                y=[p + 1.96*se for p, se in zip(proportions, std_errors)],
                mode='lines',
                name='95% CI Upper',
                line=dict(color='rgba(78,205,196,0.2)', width=0),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=flips,
                y=[p - 1.96*se for p, se in zip(proportions, std_errors)],
                mode='lines',
                name='95% CI Lower',
                line=dict(color='rgba(78,205,196,0.2)', width=0),
                fill='tonexty',
                fillcolor='rgba(78,205,196,0.2)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Expected value line
    fig.add_hline(
        y=heads_weight,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Expected: {heads_weight*100:.1f}%",
        annotation_position="bottom right",
        row=1, col=1
    )
    
    # Convergence zone
    fig.add_hrect(
        y0=heads_weight - 0.01,
        y1=heads_weight + 0.01,
        line_width=0,
        fillcolor="green",
        opacity=0.1,
        row=1, col=1
    )
    
    fig.update_xaxes(title_text="Number of Flips", row=1, col=1)
    fig.update_yaxes(title_text="Proportion", range=[0, 1], row=1, col=1)
    
    # Plot 2: Standard Error
    fig.add_trace(
        go.Scatter(
            x=flips,
            y=std_errors,
            mode='lines',
            name='Std Error',
            line=dict(color='#FF6B6B', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,107,107,0.2)'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Number of Flips", row=1, col=2)
    fig.update_yaxes(title_text="Standard Error", type="log", row=1, col=2)
    
    # Plot 3: Convergence Analysis
    if stats and stats['total_flips'] > 0:
        convergence_data = []
        for i in range(10, len(history), max(1, len(history)//100)):
            subset = history[:i]
            props = [h['proportion'] for h in subset]
            convergence_data.append({
                'flips': subset[-1]['flips'],
                'convergence': np.std(props)
            })
        
        if convergence_data:
            fig.add_trace(
                go.Scatter(
                    x=[d['flips'] for d in convergence_data],
                    y=[d['convergence'] for d in convergence_data],
                    mode='lines',
                    name='Convergence',
                    line=dict(color='#764ba2', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_hline(
                y=0.01,
                line_dash="dash",
                line_color="green",
                annotation_text="Convergence Threshold",
                annotation_position="bottom right",
                row=2, col=1
            )
    
    fig.update_xaxes(title_text="Number of Flips", row=2, col=1)
    fig.update_yaxes(title_text="Std of Proportions", type="log", row=2, col=1)
    
    # Plot 4: Z-Score
    if stats and len(history) > 1:
        z_scores = []
        for h in history:
            if h['std_error'] > 0:
                z = (h['proportion'] - heads_weight) / h['std_error']
                z_scores.append(z)
            else:
                z_scores.append(0)
        
        fig.add_trace(
            go.Scatter(
                x=flips,
                y=z_scores,
                mode='lines',
                name='Z-Score',
                line=dict(color='#FFA500', width=2)
            ),
            row=2, col=2
        )
        
        # Critical regions
        fig.add_hrect(
            y0=-1.96,
            y1=1.96,
            line_width=0,
            fillcolor="green",
            opacity=0.1,
            annotation_text="Acceptance Region (±1.96)",
            annotation_position="top right",
            row=2, col=2
        )
    
    fig.update_xaxes(title_text="Number of Flips", row=2, col=2)
    fig.update_yaxes(title_text="Z-Score", row=2, col=2)
    
    # Plot 5: Distribution
    if stats:
        fig.add_trace(
            go.Bar(
                x=['Tails', 'Heads'],
                y=[stats['tails'], stats['heads']],
                name='Observed',
                marker_color=['#FF6B6B', '#4ECDC4'],
                text=[f"{stats['tails']:,}", f"{stats['heads']:,}"],
                textposition='outside',
            ),
            row=3, col=1
        )
        
        expected_heads = stats['total_flips'] * stats['expected_prop']
        expected_tails = stats['total_flips'] * (1 - stats['expected_prop'])
        
        fig.add_trace(
            go.Bar(
                x=['Tails', 'Heads'],
                y=[expected_tails, expected_heads],
                name='Expected',
                marker_color=['rgba(255,107,107,0.3)', 'rgba(78,205,196,0.3)'],
                text=[f"{expected_tails:,.0f}", f"{expected_heads:,.0f}"],
                textposition='outside',
            ),
            row=3, col=1
        )
    
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_xaxes(title_text="Outcome", row=3, col=1)
    
    # Plot 6: Statistical Summary (as text/table)
    if stats:
        # We'll use a table instead of a traditional plot
        fig.add_annotation(
            text=f"""
            <b>Statistical Summary</b><br>
            • Total Flips: {stats['total_flips']:,}<br>
            • Current Proportion: {stats['current_prop']*100:.2f}%<br>
            • Expected: {stats['expected_prop']*100:.1f}%<br>
            • Difference: {stats['diff']*100:.2f}%<br>
            • Std Error: {stats['std_error']:.4f}<br>
            • Z-Score: {stats['z_score']:.3f}<br>
            • 95% CI: [{stats['ci_lower']*100:.1f}%, {stats['ci_upper']*100:.1f}%]<br>
            • Chi² p-value: {stats['chi2_p_value']:.4f}<br>
            • Convergence Rate: {stats['convergence_rate']:.6f}<br>
            • P(within 1%): {stats['prob_within_1pct']:.1f}%<br>
            • P(within 5%): {stats['prob_within_5pct']:.1f}%
            """,
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=12, color="white"),
            bgcolor="rgba(30,30,60,0.8)",
            bordercolor="white",
            borderwidth=1,
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=True,
        hovermode='x unified',
        template='plotly_dark',
        bargap=0.2,
    )
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">📊 Advanced Law of Large Numbers Simulator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This advanced simulation demonstrates the Law of Large Numbers with comprehensive statistical analysis.
    Watch how the proportion converges to the expected value while tracking various statistical measures in real-time.
    """)
    
    # Initialize session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = AdvancedCoinFlipSimulator()
        st.session_state.running = False
        st.session_state.flips_per_step = 500
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("## 🎮 Controls")
        
        # Coin parameters
        st.markdown("### Coin Properties")
        heads_weight = st.slider(
            "Heads Probability",
            min_value=0.01,
            max_value=0.99,
            value=0.5,
            step=0.01,
            help="The true probability of getting heads"
        )
        
        flip_force = st.slider(
            "Flip Force",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="1=Light (predictable), 10=Forceful (more random)"
        )
        
        # Simulation controls
        st.markdown("### Simulation Speed")
        flips_per_step = st.slider(
            "Flips per Step",
            min_value=10,
            max_value=5000,
            value=500,
            step=50,
            help="Higher values = faster simulation"
        )
        
        st.markdown("### Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button("▶ Start", use_container_width=True)
        with col2:
            reset_button = st.button("🔄 Reset", use_container_width=True)
        
        auto_run = st.checkbox("Auto-run", value=True)
        
        max_flips = st.selectbox(
            "Max Flips",
            options=[1000, 10000, 100000, 500000, 1000000],
            index=2,
            help="Maximum number of flips to simulate"
        )
        st.session_state.simulator.max_flips = max_flips
        
        # Statistical insights
        st.markdown("---")
        st.markdown("### 📊 Statistical Insights")
        st.info("""
        **Key Statistical Concepts:**
        - **Confidence Intervals:** Show the range where the true proportion likely lies
        - **Z-Score:** How many standard deviations from expected
        - **Chi-Square Test:** Tests if observed differs significantly from expected
        - **Standard Error:** Measures the precision of the estimate
        - **Convergence Rate:** How quickly the proportion is stabilizing
        """)
    
    # Main content
    sim = st.session_state.simulator
    
    # Handle buttons
    if reset_button:
        sim = AdvancedCoinFlipSimulator()
        st.session_state.simulator = sim
        st.session_state.running = False
        st.rerun()
    
    if start_button:
        st.session_state.running = True
    
    # Run simulation
    if (auto_run or st.session_state.running) and sim.total_flips < sim.max_flips:
        sim.run_flips(flips_per_step, heads_weight, flip_force)
        st.session_state.simulator = sim
        
        progress = min(sim.total_flips / sim.max_flips, 1.0)
        st.progress(progress, text=f"Simulating... {sim.total_flips:,} / {sim.max_flips:,} flips")
        
        if sim.total_flips < sim.max_flips and auto_run:
            time.sleep(0.01)
            st.rerun()
        elif sim.total_flips >= sim.max_flips:
            st.session_state.running = False
            st.balloons()
            st.success(f"✅ Completed {sim.total_flips:,} flips! The Law of Large Numbers is demonstrated!")
    
    # Get statistics
    stats = sim.get_statistics(heads_weight)
    
    # Display key metrics
    if stats:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Flips",
                f"{stats['total_flips']:,}",
                delta=f"{stats['diff']*100:.2f}% from expected" if stats['total_flips'] > 0 else None
            )
        
        with col2:
            st.metric(
                "Heads Proportion",
                f"{stats['current_prop']*100:.2f}%",
                delta=f"Expected: {stats['expected_prop']*100:.1f}%"
            )
        
        with col3:
            st.metric(
                "Z-Score",
                f"{stats['z_score']:.3f}",
                delta="Within 1.96" if abs(stats['z_score']) < 1.96 else "Outside 1.96"
            )
        
        with col4:
            st.metric(
                "95% CI",
                f"[{stats['ci_lower']*100:.1f}%, {stats['ci_upper']*100:.1f}%]",
                delta=f"Width: {(stats['ci_upper']-stats['ci_lower'])*100:.1f}%"
            )
        
        with col5:
            p_value = stats['chi2_p_value']
            st.metric(
                "Chi² p-value",
                f"{p_value:.4f}",
                delta="Not significant" if p_value > 0.05 else "Significant!"
            )
        
        # Additional insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="insight-box">
                <b>🎯 Convergence Status:</b><br>
                {'' if stats['diff'] < 0.01 else 'Not '}Converged (within 1%)<br>
                <b>Probability within 1%:</b> {stats['prob_within_1pct']:.1f}%<br>
                <b>Probability within 5%:</b> {stats['prob_within_5pct']:.1f}%
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="insight-box">
                <b>📈 Convergence Analysis:</b><br>
                <b>Std Error:</b> {stats['std_error']:.4f}<br>
                <b>Trend:</b> {'↓' if stats['trend'] < 0 else '↑'} {abs(stats['trend']):.6f}<br>
                <b>Recent Volatility:</b> {stats['recent_std']:.4f}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if stats['expected_flips_to_conv'] > 0 and stats['expected_flips_to_conv'] < 1e6:
                st.markdown(f"""
                <div class="insight-box">
                    <b>⏱️ Predictions:</b><br>
                    <b>Expected flips to converge:</b><br>
                    ~{stats['expected_flips_to_conv']:,.0f} more flips<br>
                    <b>Total expected:</b><br>
                    ~{stats['total_flips'] + stats['expected_flips_to_conv']:,.0f} flips
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="insight-box">
                    <b>⏱️ Status:</b><br>
                    {'Already converged!' if stats['diff'] < 0.01 else 'Converging slowly...'}<br>
                    <b>Keep simulating</b> to see convergence
                </div>
                """, unsafe_allow_html=True)
    
    # Create and display plots
    fig = create_advanced_plots(sim.history, stats, heads_weight)
    st.plotly_chart(fig, use_container_width=True)
    
    # Export data option
    if stats and len(sim.history) > 0:
        st.markdown("---")
        with st.expander("📥 Export Data", expanded=False):
            st.markdown("Download the simulation data for further analysis:")
            
            # Create DataFrame
            df = pd.DataFrame(sim.history)
            
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv,
                    file_name=f"coin_flip_data_{sim.total_flips}_flips.csv",
                    mime="text/csv"
                )
            with col2:
                st.markdown(f"**Data Points:** {len(df):,}")
                st.markdown(f"**File Size:** ~{len(csv) / 1024:.1f} KB")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📚 Understanding the Statistical Analysis
    
    | Metric | What it tells you |
    |--------|-------------------|
    | **Confidence Interval** | The range that contains the true proportion with 95% probability |
    | **Z-Score** | How many standard deviations the current proportion is from expected |
    | **Chi-Square Test** | Whether the observed distribution differs significantly from expected |
    | **Standard Error** | The precision of your estimate (smaller = better) |
    | **Convergence Rate** | How quickly the proportion is stabilizing |
    | **P(within 1%)** | Probability that the current proportion is within 1% of expected |
    """)

if __name__ == "__main__":
    main()