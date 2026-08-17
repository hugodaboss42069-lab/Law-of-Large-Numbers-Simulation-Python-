import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from scipy import stats
from scipy.special import comb
import math
import random
from datetime import datetime
from collections import Counter, defaultdict
import itertools

# Page configuration
st.set_page_config(
    page_title="Law of Large Numbers - Pro Edition",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #FFD700 0%, #FF6B6B 30%, #4ECDC4 60%, #a29bfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: shimmer 3s infinite;
        background-size: 200% 200%;
    }
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    .pro-badge {
        display: inline-block;
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #000;
        padding: 0.2rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0,0,0,0.5);
    }
    .stat-label {
        font-size: 0.85rem;
        color: #a8a8b3;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        margin-top: 0.3rem;
        background: linear-gradient(135deg, #4ECDC4 0%, #44a08d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-value.gold {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-value.red {
        background: linear-gradient(135deg, #FF6B6B 0%, #ee5a24 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-value.purple {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-value.green {
        background: linear-gradient(135deg, #00ff88 0%, #00b894 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .insight-box {
        background: linear-gradient(135deg, rgba(26,26,46,0.95) 0%, rgba(22,33,62,0.95) 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #4ECDC4;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.05);
    }
    .warning-box {
        border-left: 4px solid #FF6B6B;
    }
    .success-box {
        border-left: 4px solid #00ff88;
    }
    .info-box {
        border-left: 4px solid #4ECDC4;
    }
    .pattern-box {
        border-left: 4px solid #FFD700;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
    }
    .stButton > button:active {
        transform: scale(0.95);
    }
    .pattern-alert {
        background: linear-gradient(135deg, rgba(255,215,0,0.1), rgba(255,107,107,0.1));
        border: 1px solid rgba(255,215,0,0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .achievement {
        background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,184,148,0.1));
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ProCoinSimulator:
    def __init__(self):
        self.total_flips = 0
        self.heads = 0
        self.tails = 0
        self.history = []
        self.running = False
        self.max_flips = 5000000
        self.flip_results = []
        self.batch_results = []
        self.milestones_reached = set()
        
        # Pattern tracking
        self.current_streak = 0
        self.current_streak_type = None  # 'H' or 'T'
        self.longest_streak = 0
        self.longest_streak_type = None
        self.streak_history = []
        self.pattern_counts = defaultdict(int)
        self.pattern_lengths = [1, 2, 3, 4, 5]
        self.runs_data = []
        self.anomalies_detected = []
        
        # Statistical tracking
        self.chi_square_history = []
        self.runs_test_history = []
        self.autocorrelation_history = []
        
    def flip_coin(self, heads_weight, flip_force, bias_noise=0):
        """Simulate a coin flip with weight and force"""
        force_factor = flip_force / 10.0
        effective_weight = 0.5 + (heads_weight - 0.5) * (1 - force_factor * 0.7)
        effective_weight += bias_noise * 0.1
        noise_scale = 0.05 + (flip_force / 10.0) * 0.1
        noise = np.random.normal(0, noise_scale)
        final_prob = np.clip(effective_weight + noise, 0.01, 0.99)
        return 1 if np.random.random() < final_prob else 0
    
    def update_patterns(self, result):
        """Update pattern detection with new flip result"""
        result_char = 'H' if result == 1 else 'T'
        
        # Update streak tracking
        if self.current_streak_type == result_char:
            self.current_streak += 1
        else:
            if self.current_streak > 0:
                self.streak_history.append((self.total_flips, self.current_streak, self.current_streak_type))
                if self.current_streak > self.longest_streak:
                    self.longest_streak = self.current_streak
                    self.longest_streak_type = self.current_streak_type
            self.current_streak = 1
            self.current_streak_type = result_char
        
        # Update pattern counts (all lengths)
        if len(self.flip_results) >= 5:
            for length in self.pattern_lengths:
                if len(self.flip_results) >= length:
                    pattern = ''.join(['H' if x == 1 else 'T' for x in self.flip_results[-length:]])
                    self.pattern_counts[pattern] += 1
    
    def run_flips(self, num_flips, heads_weight, flip_force, bias_noise=0, batch_size=0):
        """Run multiple flips with pattern detection"""
        new_heads = 0
        new_tails = 0
        
        for i in range(num_flips):
            result = self.flip_coin(heads_weight, flip_force, bias_noise)
            self.flip_results.append(result)
            
            # Update patterns
            self.update_patterns(result)
            
            if result == 1:
                self.heads += 1
                new_heads += 1
            else:
                self.tails += 1
                new_tails += 1
            self.total_flips += 1
            
            # Check for significant streaks (anomalies)
            if self.current_streak >= 10:
                if len(self.anomalies_detected) == 0 or self.anomalies_detected[-1][0] < self.total_flips - 100:
                    self.anomalies_detected.append((
                        self.total_flips,
                        self.current_streak,
                        self.current_streak_type,
                        f"Streak of {self.current_streak} {self.current_streak_type}s! (Expected in random data)"
                    ))
            
            # Batch for CLT
            if batch_size > 0 and self.total_flips % batch_size == 0:
                batch_heads = sum(self.flip_results[-batch_size:])
                self.batch_results.append(batch_heads / batch_size)
            
            # Store history at intervals
            interval = max(1, num_flips // 200)
            if self.total_flips % interval == 0 or i == num_flips - 1:
                proportion = self.heads / self.total_flips
                std_error = math.sqrt((proportion * (1 - proportion)) / self.total_flips)
                self.history.append({
                    'flips': self.total_flips,
                    'proportion': proportion,
                    'std_error': std_error,
                    'heads': self.heads,
                    'tails': self.tails,
                    'timestamp': datetime.now()
                })
            
            # Milestone tracking
            if self.total_flips >= 10 and '10' not in self.milestones_reached:
                self.milestones_reached.add('10')
            elif self.total_flips >= 100 and '100' not in self.milestones_reached:
                self.milestones_reached.add('100')
            elif self.total_flips >= 1000 and '1000' not in self.milestones_reached:
                self.milestones_reached.add('1000')
            elif self.total_flips >= 10000 and '10000' not in self.milestones_reached:
                self.milestones_reached.add('10000')
            elif self.total_flips >= 100000 and '100000' not in self.milestones_reached:
                self.milestones_reached.add('100000')
            elif self.total_flips >= 1000000 and '1000000' not in self.milestones_reached:
                self.milestones_reached.add('1000000')
        
        return new_heads, new_tails
    
    def analyze_patterns(self, heads_weight):
        """Comprehensive pattern analysis"""
        if self.total_flips == 0:
            return None
        
        # Runs test (Wald-Wolfowitz)
        runs = 1
        for i in range(1, len(self.flip_results)):
            if self.flip_results[i] != self.flip_results[i-1]:
                runs += 1
        
        n1 = self.heads
        n2 = self.tails
        expected_runs = 1 + (2 * n1 * n2) / (n1 + n2) if (n1 + n2) > 0 else 0
        std_runs = math.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))) if (n1 + n2) > 1 else 0
        z_runs = (runs - expected_runs) / std_runs if std_runs > 0 else 0
        p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
        
        # Pattern frequency analysis
        pattern_analysis = {}
        for length in self.pattern_lengths:
            if length <= len(self.flip_results):
                all_patterns = [''.join(['H' if x == 1 else 'T' for x in self.flip_results[i:i+length]]) 
                               for i in range(len(self.flip_results) - length + 1)]
                pattern_counts = Counter(all_patterns)
                expected_count = len(all_patterns) / (2 ** length)
                
                # Chi-square for pattern distribution
                observed = [pattern_counts.get(p, 0) for p in [''.join(bits) for bits in itertools.product(['H', 'T'], repeat=length)]]
                expected = [expected_count] * (2 ** length)
                chi2_pattern, p_pattern = stats.chisquare(observed, expected)
                
                pattern_analysis[length] = {
                    'counts': pattern_counts,
                    'expected': expected_count,
                    'chi2': chi2_pattern,
                    'p_value': p_pattern,
                    'total_patterns': len(all_patterns)
                }
        
        return {
            'runs': runs,
            'expected_runs': expected_runs,
            'z_runs': z_runs,
            'p_runs': p_runs,
            'pattern_analysis': pattern_analysis,
            'longest_streak': self.longest_streak,
            'longest_streak_type': self.longest_streak_type,
            'total_streaks': len(self.streak_history),
            'anomalies': self.anomalies_detected,
            'pattern_counts': dict(self.pattern_counts)
        }

def create_pro_plots(simulator, heads_weight, pattern_analysis):
    """Create Pro Edition plots with pattern detection"""
    
    history = simulator.history
    if not history:
        fig = go.Figure()
        fig.add_annotation(text="🚀 Run the simulation to unlock Pro features!", showarrow=False, font=dict(size=20))
        return fig
    
    # Extract data
    flips = [h['flips'] for h in history]
    proportions = [h['proportion'] for h in history]
    std_errors = [h['std_error'] for h in history]
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            '🎯 Running Proportion with Confidence Intervals',
            '📊 Distribution & Convergence',
            '📈 Statistical Indicators',
            '🎲 Central Limit Theorem',
            '🔍 Pattern Detection - Streaks',
            '📊 Pattern Distribution',
            '🔄 Autocorrelation Analysis',
            '🏆 Achievements & Anomalies'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
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
            line=dict(color='#4ECDC4', width=3),
            hovertemplate='Flips: %{x:,}<br>Proportion: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Confidence interval
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
    
    fig.add_hline(
        y=heads_weight,
        line_dash="dash",
        line_color="#FF6B6B",
        annotation_text=f"Expected: {heads_weight*100:.1f}%",
        annotation_position="bottom right",
        row=1, col=1
    )
    
    fig.add_hrect(
        y0=heads_weight - 0.01,
        y1=heads_weight + 0.01,
        line_width=0,
        fillcolor="rgba(0,255,0,0.1)",
        row=1, col=1
    )
    
    fig.update_xaxes(title_text="Number of Flips", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Proportion", range=[0, 1], row=1, col=1)
    
    # Plot 2: Distribution
    if len(simulator.flip_results) > 0:
        window = min(100, len(simulator.flip_results))
        running_avg = []
        for i in range(window, len(simulator.flip_results) + 1):
            running_avg.append(sum(simulator.flip_results[i-window:i]) / window)
        
        fig.add_trace(
            go.Histogram(
                x=running_avg,
                nbinsx=20,
                name='Running Average',
                marker_color='#764ba2',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        if len(running_avg) > 10:
            mu = np.mean(running_avg)
            sigma = np.std(running_avg)
            x_norm = np.linspace(min(running_avg), max(running_avg), 100)
            y_norm = stats.norm.pdf(x_norm, mu, sigma) * len(running_avg) * (max(running_avg) - min(running_avg)) / 20
            
            fig.add_trace(
                go.Scatter(
                    x=x_norm,
                    y=y_norm,
                    mode='lines',
                    name='Normal Fit',
                    line=dict(color='#FF6B6B', width=2, dash='dash')
                ),
                row=1, col=2
            )
        
        fig.update_xaxes(title_text="Proportion (Window of 100)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    # Plot 3: Statistical Indicators
    if len(history) > 10:
        window_size = min(20, len(history))
        rolling_std = []
        for i in range(window_size, len(history)):
            subset = [h['proportion'] for h in history[i-window_size:i]]
            rolling_std.append(np.std(subset))
        
        fig.add_trace(
            go.Scatter(
                x=flips[window_size:],
                y=rolling_std,
                mode='lines',
                name='Rolling Std Dev',
                line=dict(color='#FFA500', width=2),
                fill='tozeroy',
                fillcolor='rgba(255,165,0,0.2)'
            ),
            row=2, col=1
        )
        
        fig.add_hline(
            y=0.01,
            line_dash="dash",
            line_color="#00ff88",
            annotation_text="Convergence Threshold",
            annotation_position="bottom right",
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Number of Flips", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Standard Deviation", type="log", row=2, col=1)
    
    # Plot 4: CLT
    if len(simulator.batch_results) > 1:
        fig.add_trace(
            go.Histogram(
                x=simulator.batch_results,
                nbinsx=30,
                name='Sample Means',
                marker_color='#6c5ce7',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        mu = np.mean(simulator.batch_results)
        sigma = np.std(simulator.batch_results)
        x_norm = np.linspace(min(simulator.batch_results), max(simulator.batch_results), 100)
        y_norm = stats.norm.pdf(x_norm, mu, sigma) * len(simulator.batch_results) * (max(simulator.batch_results) - min(simulator.batch_results)) / 30
        
        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                mode='lines',
                name='Normal Fit',
                line=dict(color='#FF6B6B', width=2)
            ),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Sample Mean", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    # Plot 5: Pattern Detection - Streaks
    if pattern_analysis and len(simulator.streak_history) > 0:
        streak_flips = [s[0] for s in simulator.streak_history]
        streak_lengths = [s[1] for s in simulator.streak_history]
        streak_types = [s[2] for s in simulator.streak_history]
        
        colors = ['#4ECDC4' if t == 'H' else '#FF6B6B' for t in streak_types]
        
        fig.add_trace(
            go.Scatter(
                x=streak_flips,
                y=streak_lengths,
                mode='markers',
                name='Streaks',
                marker=dict(
                    size=streak_lengths,
                    color=colors,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                hovertemplate='Flip: %{x:,}<br>Length: %{y}<br>Type: %{text}<extra></extra>',
                text=streak_types
            ),
            row=3, col=1
        )
        
        fig.add_hline(
            y=10,
            line_dash="dash",
            line_color="#FFD700",
            annotation_text="10+ Streak Alert",
            annotation_position="bottom right",
            row=3, col=1
        )
        
        fig.update_xaxes(title_text="Flip Number", row=3, col=1)
        fig.update_yaxes(title_text="Streak Length", row=3, col=1)
    
    # Plot 6: Pattern Distribution
    if pattern_analysis and pattern_analysis.get('pattern_analysis'):
        for length, analysis in pattern_analysis['pattern_analysis'].items():
            if len(analysis['counts']) <= 8:  # Only show smaller patterns
                patterns = list(analysis['counts'].keys())[:8]
                counts = list(analysis['counts'].values())[:8]
                
                fig.add_trace(
                    go.Bar(
                        x=patterns,
                        y=counts,
                        name=f'{length}-bit patterns',
                        marker_color='#a29bfe',
                        opacity=0.7
                    ),
                    row=3, col=2
                )
                break
    
    fig.update_xaxes(title_text="Pattern", row=3, col=2)
    fig.update_yaxes(title_text="Frequency", row=3, col=2)
    
    # Plot 7: Autocorrelation
    if len(simulator.flip_results) > 100:
        max_lag = 20
        autocorr = []
        for lag in range(1, max_lag + 1):
            corr = np.corrcoef(simulator.flip_results[:-lag], simulator.flip_results[lag:])[0, 1]
            autocorr.append(corr if not np.isnan(corr) else 0)
        
        fig.add_trace(
            go.Bar(
                x=list(range(1, max_lag + 1)),
                y=autocorr,
                name='Autocorrelation',
                marker_color='#FF6B6B',
                opacity=0.7
            ),
            row=4, col=1
        )
        
        # Significance bounds
        bound = 1.96 / math.sqrt(len(simulator.flip_results))
        fig.add_hrect(
            y0=-bound,
            y1=bound,
            line_width=0,
            fillcolor="rgba(0,255,0,0.1)",
            row=4, col=1
        )
        
        fig.update_xaxes(title_text="Lag", row=4, col=1)
        fig.update_yaxes(title_text="Correlation", row=4, col=1)
    
    # Plot 8: Achievements & Anomalies
    if pattern_analysis and pattern_analysis.get('anomalies'):
        anomaly_flips = [a[0] for a in pattern_analysis['anomalies']]
        anomaly_lengths = [a[1] for a in pattern_analysis['anomalies']]
        
        fig.add_trace(
            go.Scatter(
                x=anomaly_flips,
                y=anomaly_lengths,
                mode='markers+text',
                name='Anomalies',
                marker=dict(
                    size=20,
                    color='#FFD700',
                    symbol='star',
                    line=dict(width=2, color='white')
                ),
                text=['🔴' for _ in anomaly_flips],
                textposition='top center',
                hovertemplate='Flip: %{x:,}<br>Length: %{y}<extra>⚠️ Anomaly Detected!</extra>'
            ),
            row=4, col=2
        )
        
        fig.update_xaxes(title_text="Flip Number", row=4, col=2)
        fig.update_yaxes(title_text="Anomaly Severity", row=4, col=2)
    
    # Update layout
    fig.update_layout(
        height=1600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_dark',
        bargap=0.1,
        font=dict(family="Arial, sans-serif"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div style="text-align: center;">
        <span style="font-size: 3rem;">🏆</span>
        <div class="main-header">
            The Law of Large Numbers: Pro Edition
            <span class="pro-badge">PRO</span>
        </div>
        <div style="color: #a8a8b3; margin-bottom: 2rem;">
            🔍 Advanced Pattern Detection • 📊 Statistical Analysis • 🎯 5 Million Flips
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = ProCoinSimulator()
        st.session_state.running = False
        st.session_state.flips_per_step = 500
        st.session_state.show_clt = True
        st.session_state.pattern_analysis = None
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎮 Pro Controls")
        
        with st.expander("🎯 Coin Settings", expanded=True):
            heads_weight = st.slider(
                "Heads Probability",
                min_value=0.01,
                max_value=0.99,
                value=0.5,
                step=0.01
            )
            
            flip_force = st.slider(
                "Flip Force 💪",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="1=Light (predictable), 10=Forceful (random)"
            )
            
            bias_noise = st.slider(
                "Bias Noise 🌊",
                min_value=-0.5,
                max_value=0.5,
                value=0.0,
                step=0.01,
                help="Adds random bias to the coin"
            )
        
        with st.expander("⚡ Simulation Controls", expanded=True):
            flips_per_step = st.slider(
                "Flips per Step",
                min_value=10,
                max_value=50000,
                value=1000,
                step=100,
                help="Higher = faster simulation"
            )
            
            batch_size = st.number_input(
                "CLT Batch Size",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )
            
            max_flips = st.selectbox(
                "Max Flips",
                options=[1000, 10000, 100000, 500000, 1000000, 5000000],
                index=5,
                help="5 million flips for Pro analysis!"
            )
            st.session_state.simulator.max_flips = max_flips
        
        with st.expander("🔍 Pattern Detection", expanded=True):
            st.markdown("""
            **Active Pattern Detection:**
            - 🔴 Runs Analysis
            - 🟡 Streak Tracking
            - 🟢 Pattern Frequency
            - 🔵 Anomaly Detection
            - 🟣 Autocorrelation
            """)
            
            detect_streaks = st.checkbox("Detect Streaks (10+)", value=True)
            detect_patterns = st.checkbox("Detect Repeating Patterns", value=True)
            detect_anomalies = st.checkbox("Detect Anomalies", value=True)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            start_button = st.button("▶ Start", use_container_width=True)
        with col2:
            pause_button = st.button("⏸ Pause", use_container_width=True)
        with col3:
            reset_button = st.button("🔄 Reset", use_container_width=True)
        
        auto_run = st.checkbox("🤖 Auto-Run", value=True)
        
        # Quick presets
        st.markdown("---")
        st.markdown("### 🚀 Pro Presets")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 Fair Coin", use_container_width=True):
                heads_weight = 0.5
                flip_force = 5
                bias_noise = 0
                st.rerun()
        with col2:
            if st.button("🎲 Biased", use_container_width=True):
                heads_weight = 0.7
                flip_force = 3
                bias_noise = 0
                st.rerun()
    
    # Main content
    sim = st.session_state.simulator
    
    # Handle buttons
    if reset_button:
        sim = ProCoinSimulator()
        st.session_state.simulator = sim
        st.session_state.running = False
        st.session_state.pattern_analysis = None
        st.success("🔄 Reset complete! Ready for Pro analysis...")
        st.rerun()
    
    if start_button:
        st.session_state.running = True
    
    if pause_button:
        st.session_state.running = False
        st.info("⏸ Paused. Press 'Start' to continue.")
    
    # Run simulation
    if (auto_run or st.session_state.running) and sim.total_flips < sim.max_flips:
        sim.run_flips(flips_per_step, heads_weight, flip_force, bias_noise, batch_size)
        st.session_state.simulator = sim
        
        # Update pattern analysis
        st.session_state.pattern_analysis = sim.analyze_patterns(heads_weight)
        
        # Progress
        progress = min(sim.total_flips / sim.max_flips, 1.0)
        st.progress(progress)
        
        # Milestone display
        milestone_text = ""
        if sim.total_flips >= 10:
            milestone_text += "🔟 "
        if sim.total_flips >= 100:
            milestone_text += "💯 "
        if sim.total_flips >= 1000:
            milestone_text += "🔢 "
        if sim.total_flips >= 10000:
            milestone_text += "🎯 "
        if sim.total_flips >= 100000:
            milestone_text += "🏆 "
        if sim.total_flips >= 1000000:
            milestone_text += "👑 "
        
        st.caption(f"📊 Flips: {sim.total_flips:,} / {sim.max_flips:,}  {milestone_text}")
        
        if sim.total_flips < sim.max_flips and auto_run:
            time.sleep(0.01)
            st.rerun()
        elif sim.total_flips >= sim.max_flips:
            st.session_state.running = False
            st.balloons()
            st.success(f"🎉 Completed {sim.total_flips:,} flips! Pro analysis complete!")
    
    # Stats dashboard
    if sim.total_flips > 0:
        current_prop = sim.heads / sim.total_flips
        diff = abs(current_prop - heads_weight)
        std_error = math.sqrt((current_prop * (1 - current_prop)) / sim.total_flips)
        
        # Top row stats
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">🪙 Total Flips</div>
                <div class="stat-value gold">{sim.total_flips:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">📈 Heads</div>
                <div class="stat-value">{sim.heads:,} ({current_prop*100:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">📉 Tails</div>
                <div class="stat-value red">{sim.tails:,} ({(1-current_prop)*100:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">🎯 Expected</div>
                <div class="stat-value purple">{heads_weight*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            status_color = "✅" if diff < 0.01 else ("🟡" if diff < 0.03 else "🔴")
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">📊 Status</div>
                <div class="stat-value">{status_color} {diff*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Pattern detection dashboard
        if st.session_state.pattern_analysis:
            pa = st.session_state.pattern_analysis
            
            st.markdown("---")
            st.markdown("### 🔍 Pattern Detection & Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="insight-box pattern-box">
                    <b>🎯 Streak Analysis</b><br>
                    • Longest Streak: <b>{pa['longest_streak']} {pa['longest_streak_type']}</b><br>
                    • Total Streaks: {pa['total_streaks']:,}<br>
                    • Expected Max: ~{int(math.log2(sim.total_flips) + 1)}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                runs_status = "✅ Random" if pa['p_runs'] > 0.05 else "⚠️ Non-random"
                st.markdown(f"""
                <div class="insight-box pattern-box">
                    <b>📊 Runs Test</b><br>
                    • Runs: {pa['runs']:,}<br>
                    • Expected: {pa['expected_runs']:,.0f}<br>
                    • P-value: {pa['p_runs']:.4f} ({runs_status})
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                anomaly_count = len(pa['anomalies'])
                st.markdown(f"""
                <div class="insight-box pattern-box">
                    <b>⚠️ Anomalies Detected</b><br>
                    • Total Anomalies: {anomaly_count}<br>
                    • Latest: {pa['anomalies'][-1][1]} {pa['anomalies'][-1][2]}s at flip {pa['anomalies'][-1][0]:,}<br>
                    • Status: {'🟢 Normal' if anomaly_count < 10 else '🟡 Many detections'}
                </div>
                """, unsafe_allow_html=True)
            
            # Pattern frequency summary
            if pa['pattern_analysis']:
                st.markdown("#### 📊 Pattern Frequency Analysis")
                
                pattern_cols = st.columns(4)
                for idx, (length, analysis) in enumerate(pa['pattern_analysis'].items()):
                    if idx < 4 and len(analysis['counts']) <= 8:
                        with pattern_cols[idx]:
                            top_patterns = sorted(analysis['counts'].items(), key=lambda x: x[1], reverse=True)[:3]
                            st.markdown(f"""
                            <div class="insight-box" style="border-left-color: #a29bfe;">
                                <b>{length}-Bit Patterns</b><br>
                                {', '.join([f'{p}: {c:,}' for p, c in top_patterns])}<br>
                                <span style="color: #a8a8b3; font-size: 0.8rem;">P-value: {analysis['p_value']:.4f}</span>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Anomaly alerts
            if pa['anomalies']:
                st.markdown("#### 🔔 Recent Anomaly Alerts")
                recent_anomalies = pa['anomalies'][-5:]
                for anomaly in recent_anomalies:
                    st.markdown(f"""
                    <div class="pattern-alert">
                        ⚠️ <b>Streak of {anomaly[1]} {anomaly[2]}s</b> detected at flip {anomaly[0]:,}
                        <span style="color: #a8a8b3; font-size: 0.9rem;">— {anomaly[3]}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Achievements
            st.markdown("#### 🏆 Pro Achievements")
            
            achievements = []
            if sim.total_flips >= 1000:
                achievements.append("🎯 1,000 Flips Club")
            if sim.total_flips >= 10000:
                achievements.append("💎 10,000 Flips Club")
            if sim.total_flips >= 100000:
                achievements.append("🏆 100,000 Flips Club")
            if sim.total_flips >= 1000000:
                achievements.append("👑 1,000,000 Flips Club")
            if sim.total_flips >= 5000000:
                achievements.append("🌟 5,000,000 Flips Club (Pro!)")
            if pa['longest_streak'] >= 15:
                achievements.append("🔥 Streak Master (15+)")
            if pa['p_runs'] > 0.05:
                achievements.append("🎲 Randomness Expert")
            if len(pa['anomalies']) >= 5:
                achievements.append("🔍 Pattern Hunter")
            
            if achievements:
                cols = st.columns(3)
                for idx, achievement in enumerate(achievements[:6]):
                    with cols[idx % 3]:
                        st.markdown(f"""
                        <div class="achievement">
                            {achievement}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("🚀 Keep flipping to unlock achievements!")
    
    # Main plots
    fig = create_pro_plots(sim, heads_weight, st.session_state.pattern_analysis)
    st.plotly_chart(fig, use_container_width=True)
    
    # Export section
    with st.expander("📥 Pro Export & Analysis", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if len(sim.history) > 0:
                df = pd.DataFrame(sim.history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Pro Analysis CSV",
                    data=csv,
                    file_name=f"pro_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                st.caption(f"📁 {len(df):,} data points")
        
        with col2:
            if sim.total_flips > 0:
                st.markdown(f"""
                **📈 Pro Summary**
                - Total Flips: {sim.total_flips:,}
                - Longest Streak: {sim.longest_streak} {sim.longest_streak_type}s
                - Patterns Analyzed: {len(st.session_state.pattern_analysis['pattern_analysis']) if st.session_state.pattern_analysis else 0} lengths
                - Anomalies: {len(sim.anomalies_detected)}
                - Achievements: {len(achievements) if 'achievements' in locals() else 0}
                """)
    
    # Pro tips footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #a8a8b3; font-size: 0.9rem;">
        🏆 <b>Pro Edition Features:</b> Pattern Detection • Runs Analysis • Streak Tracking • Anomaly Detection • Autocorrelation • CLT Demo
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()