import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import deque

# Page configuration
st.set_page_config(
    page_title="Law of Large Numbers - Coin Flip Simulator",
    page_icon="🪙",
    layout="wide"
)

# Custom CSS for better styling
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
</style>
""", unsafe_allow_html=True)

class CoinFlipSimulator:
    def __init__(self):
        self.total_flips = 0
        self.heads = 0
        self.tails = 0
        self.history = []
        self.running = False
        self.max_flips = 1000000

    def flip_coin(self, heads_weight, flip_force):
        """Simulate a coin flip with weight and force"""
        # Force affects randomness (1-10 scale)
        force_factor = flip_force / 10.0

        # Higher force = more randomness (closer to 50/50)
        effective_weight = 0.5 + (heads_weight - 0.5) * (1 - force_factor * 0.7)

        # Add random noise based on force
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
            if result == 1:
                self.heads += 1
                new_heads += 1
            else:
                self.tails += 1
                new_tails += 1
            self.total_flips += 1

            # Store history at intervals
            if self.total_flips % max(1, num_flips // 100) == 0:
                proportion = self.heads / self.total_flips
                self.history.append((self.total_flips, proportion))

        return new_heads, new_tails

def create_plots(history, heads_weight, total_flips, heads, tails):
    """Create interactive Plotly figures"""

    # Extract data
    if history:
        flips, proportions = zip(*history)
    else:
        flips, proportions = [], []

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Running Proportion of Heads',
            'Convergence Status',
            'Distribution of Outcomes',
            'Convergence Speed'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Plot 1: Running proportion
    if flips:
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

        # Expected value line
        fig.add_hline(
            y=heads_weight,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Expected: {heads_weight*100:.1f}%",
            annotation_position="bottom right",
            row=1, col=1
        )

        # Convergence zone (±1%)
        fig.add_hrect(
            y0=heads_weight - 0.01,
            y1=heads_weight + 0.01,
            line_width=0,
            fillcolor="green",
            opacity=0.1,
            row=1, col=1
        )

        # Current proportion
        if total_flips > 0:
            current_prop = heads / total_flips
            fig.add_hline(
                y=current_prop,
                line_dash="dot",
                line_color="blue",
                annotation_text=f"Current: {current_prop*100:.2f}%",
                annotation_position="top right",
                row=1, col=1
            )

    fig.update_xaxes(title_text="Number of Flips", row=1, col=1)
    fig.update_yaxes(title_text="Proportion", range=[0, 1], row=1, col=1)

    # Plot 2: Convergence status (bar chart showing difference from expected)
    if total_flips > 0:
        current_prop = heads / total_flips
        diff = abs(current_prop - heads_weight)

        # Determine status
        if diff < 0.01:
            status = "Converged ✓"
            color = "#00ff00"
        elif diff < 0.03:
            status = "Approaching"
            color = "#ffa500"
        else:
            status = "Still converging"
            color = "#ff4444"

        fig.add_trace(
            go.Bar(
                x=['Difference from Expected'],
                y=[diff * 100],
                name='Difference',
                marker_color=color,
                text=[f'{diff*100:.2f}%'],
                textposition='outside',
            ),
            row=1, col=2
        )

        fig.add_hline(
            y=1,
            line_dash="dash",
            line_color="green",
            annotation_text="Convergence Zone (±1%)",
            annotation_position="top right",
            row=1, col=2
        )

        fig.update_yaxes(title_text="Difference (%)", range=[0, max(5, diff*100*1.5)], row=1, col=2)

    # Plot 3: Distribution histogram
    fig.add_trace(
        go.Bar(
            x=['Tails', 'Heads'],
            y=[tails, heads],
            name='Counts',
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f'{tails:,}', f'{heads:,}'],
            textposition='outside',
        ),
        row=2, col=1
    )

    # Expected counts
    if total_flips > 0:
        expected_heads = total_flips * heads_weight
        expected_tails = total_flips * (1 - heads_weight)

        fig.add_trace(
            go.Bar(
                x=['Tails', 'Heads'],
                y=[expected_tails, expected_heads],
                name='Expected',
                marker_color=['rgba(255,107,107,0.3)', 'rgba(78,205,196,0.3)'],
                text=[f'{expected_tails:,.0f}', f'{expected_heads:,.0f}'],
                textposition='outside',
            ),
            row=2, col=1
        )

    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Outcome", row=2, col=1)

    # Plot 4: Convergence speed (how quickly it converges over time)
    if len(flips) > 1:
        diffs = [abs(p - heads_weight) * 100 for p in proportions]
        fig.add_trace(
            go.Scatter(
                x=flips,
                y=diffs,
                mode='lines',
                name='Difference %',
                line=dict(color='#764ba2', width=2),
                fill='tozeroy',
                fillcolor='rgba(118, 75, 162, 0.2)'
            ),
            row=2, col=2
        )

        fig.add_hline(
            y=1,
            line_dash="dash",
            line_color="green",
            annotation_text="Convergence Zone",
            annotation_position="top right",
            row=2, col=2
        )

        fig.update_yaxes(
            title_text="Difference from Expected (%)",
            type="log" if total_flips > 1000 else "linear",
            row=2, col=2
        )
        fig.update_xaxes(title_text="Number of Flips", row=2, col=2)

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        template='plotly_dark',
        bargap=0.2,
    )

    return fig

def main():
    # Header
    st.markdown('<div class="main-header">🪙 Law of Large Numbers Simulator</div>', unsafe_allow_html=True)

    st.markdown("""
    This interactive simulation demonstrates how the proportion of heads converges to the expected probability 
    as the number of coin flips increases. Adjust the parameters below and watch the convergence in real-time!
    """)

    # Initialize session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = CoinFlipSimulator()
        st.session_state.running = False
        st.session_state.flips_per_step = 100

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
            max_value=10000,
            value=500,
            step=10,
            help="Higher values = faster simulation"
        )

        st.markdown("### Actions")
        col1, col2 = st.columns(2)

        with col1:
            start_button = st.button("▶ Start", use_container_width=True)
        with col2:
            reset_button = st.button("🔄 Reset", use_container_width=True)

        # Auto-run checkbox
        auto_run = st.checkbox("Auto-run", value=True)

        # Max flips
        max_flips = st.selectbox(
            "Max Flips",
            options=[1000, 10000, 100000, 500000, 1000000],
            index=2,
            help="Maximum number of flips to simulate"
        )
        st.session_state.simulator.max_flips = max_flips

        # Info box
        st.markdown("---")
        st.markdown("### 📊 Understanding the Simulation")
        st.info("""
        **Key Concepts:**
        - **Few flips (10-100):** Large fluctuations possible
        - **Many flips (10,000+):** Proportion stabilizes
        - **1 million flips:** Very close to expected value

        **Convergence Zone (±1%):** 
        When the proportion is within 1% of expected, the simulation has converged.
        """)

        # Stats summary
        st.markdown("---")
        st.markdown("### 📈 Current Stats")
        stats_placeholder = st.empty()

    # Main content area
    col1, col2, col3, col4 = st.columns(4)

    # Stats display
    sim = st.session_state.simulator

    if sim.total_flips > 0:
        current_prop = sim.heads / sim.total_flips
        diff = abs(current_prop - heads_weight) * 100

        # Determine convergence status
        if diff < 1:
            status = "✅ Converged!"
            status_color = "converged"
        elif diff < 3:
            status = "🔄 Approaching..."
            status_color = "converging"
        else:
            status = "⏳ Still converging"
            status_color = "far"

        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Flips</div>
                <div class="stat-value">{sim.total_flips:,}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #4ECDC4 0%, #44a08d 100%);">
                <div class="stat-label">Heads</div>
                <div class="stat-value">{sim.heads:,} ({current_prop*100:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #FF6B6B 0%, #ee5a24 100%);">
                <div class="stat-label">Tails</div>
                <div class="stat-value">{sim.tails:,} ({(1-current_prop)*100:.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <div class="stat-label">Status</div>
                <div class="stat-value {status_color}">{status}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        for col in [col1, col2, col3, col4]:
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">Waiting...</div>
                    <div class="stat-value">-</div>
                </div>
                """, unsafe_allow_html=True)

    # Sidebar stats update
    if sim.total_flips > 0:
        with stats_placeholder.container():
            st.markdown(f"""
            - **Total Flips:** {sim.total_flips:,}
            - **Heads:** {sim.heads:,} ({current_prop*100:.2f}%)
            - **Tails:** {sim.tails:,} ({(1-current_prop)*100:.2f}%)
            - **Expected Heads:** {heads_weight*100:.1f}%
            - **Difference:** {diff:.2f}%
            - **Flip Force:** {flip_force}/10
            """)

    # Handle buttons
    if reset_button:
        sim = CoinFlipSimulator()
        st.session_state.simulator = sim
        st.session_state.running = False
        st.rerun()

    if start_button:
        st.session_state.running = True

    # Run simulation
    if (auto_run or st.session_state.running) and sim.total_flips < sim.max_flips:
        # Run flips
        new_heads, new_tails = sim.run_flips(flips_per_step, heads_weight, flip_force)

        # Update session state
        st.session_state.simulator = sim

        # Progress bar
        progress = min(sim.total_flips / sim.max_flips, 1.0)
        st.progress(progress, text=f"Simulating... {sim.total_flips:,} / {sim.max_flips:,} flips")

        # Auto-rerun if not complete
        if sim.total_flips < sim.max_flips and auto_run:
            time.sleep(0.01)  # Small delay to prevent UI freezing
            st.rerun()
        elif sim.total_flips >= sim.max_flips:
            st.session_state.running = False
            st.balloons()
            st.success(f"✅ Completed {sim.total_flips:,} flips! The Law of Large Numbers is demonstrated!")

    # Create and display plots
    if sim.history:
        fig = create_plots(
            sim.history,
            heads_weight,
            sim.total_flips,
            sim.heads,
            sim.tails
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Press 'Start' or enable 'Auto-run' to begin the simulation!")

        # Show example of what the plot will look like
        st.markdown("### 📊 Preview")
        st.markdown("""
        The simulation will show:
        1. **Running Proportion** - How the proportion of heads changes over time
        2. **Convergence Status** - How close we are to the expected value
        3. **Distribution** - Current vs expected counts
        4. **Convergence Speed** - How quickly we're approaching the expected value
        """)

    # Footer with explanation
    st.markdown("---")
    st.markdown("""
    ### 📚 Understanding the Law of Large Numbers

    The **Law of Large Numbers (LLN)** states that as the number of trials increases, 
    the sample average converges to the expected value.

    **Key Observations:**
    - 🔴 **Small samples (10-100 flips):** Results can be far from expected
    - 🟡 **Medium samples (1,000-10,000 flips):** Results begin to stabilize
    - 🟢 **Large samples (100,000+ flips):** Results are very close to expected

    **The Flip Force Effect:**
    - **Low force (1-3):** The coin's bias dominates, results are more predictable
    - **Medium force (4-7):** Natural randomness, good demonstration of LLN
    - **High force (8-10):** Random factors dominate, tends toward 50/50
    """)

if __name__ == "__main__":
    main()