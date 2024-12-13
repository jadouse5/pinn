import streamlit as st
import torch
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import io

class ThreeBodyPINN(torch.nn.Module):
    def __init__(self, hidden_layers=[256, 256, 256, 256]):
        super().__init__()
        layers = []
        prev_dim = 1
        for h_dim in hidden_layers:
            layers.append(torch.nn.Linear(prev_dim, h_dim))
            layers.append(torch.nn.Tanh())
            prev_dim = h_dim
        layers.append(torch.nn.Linear(prev_dim, 6))
        self.network = torch.nn.Sequential(*layers)
        self.G = 6.67430e-11
        self.m1 = 1.0
        self.m2 = 1.0
        self.m3 = 1.0

    def forward(self, t):
        return self.network(t)

class TraditionalSolver:
    def __init__(self, G=6.67430e-11, m1=1.0, m2=1.0, m3=1.0):
        self.G = G
        self.m1, self.m2, self.m3 = m1, m2, m3
    
    def derivatives(self, state, t):
        x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = state
        
        # Compute distances
        r12 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        r13 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
        r23 = np.sqrt((x2 - x3)**2 + (y2 - y3)**2)
        
        # Compute accelerations
        ax1 = self.G * (self.m2 * (x2 - x1) / r12**3 + self.m3 * (x3 - x1) / r13**3)
        ay1 = self.G * (self.m2 * (y2 - y1) / r12**3 + self.m3 * (y3 - y1) / r13**3)
        
        ax2 = self.G * (self.m1 * (x1 - x2) / r12**3 + self.m3 * (x3 - x2) / r23**3)
        ay2 = self.G * (self.m1 * (y1 - y2) / r12**3 + self.m3 * (y3 - y2) / r23**3)
        
        ax3 = self.G * (self.m1 * (x1 - x3) / r13**3 + self.m2 * (x2 - x3) / r23**3)
        ay3 = self.G * (self.m1 * (y1 - y3) / r13**3 + self.m2 * (y2 - y3) / r23**3)
        
        return [vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3]

def load_model(uploaded_file):
    try:
        model = ThreeBodyPINN()
        # Load model bytes into buffer
        bytes_data = uploaded_file.getvalue()
        buffer = io.BytesIO(bytes_data)
        # Load state dict with proper settings
        state_dict = torch.load(buffer, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def compare_methods(pinn_model, t_range, n_points=1000):
    # Time points
    t_points = np.linspace(t_range[0], t_range[1], n_points)
    
    # PINN predictions
    t_tensor = torch.tensor(t_points, dtype=torch.float32).reshape(-1, 1)
    start_time = time.time()
    with torch.no_grad():
        pinn_positions = pinn_model(t_tensor).numpy()
    pinn_time = time.time() - start_time
    
    # Traditional method
    solver = TraditionalSolver(G=pinn_model.G, m1=pinn_model.m1, 
                             m2=pinn_model.m2, m3=pinn_model.m3)
    initial_conditions = np.concatenate([pinn_positions[0], np.zeros(6)])
    
    start_time = time.time()
    try:
        trad_solution = odeint(solver.derivatives, initial_conditions, t_points)
        trad_positions = trad_solution[:, :6]
        trad_time = time.time() - start_time
        trad_success = True
    except Exception as e:
        st.warning(f"Traditional method failed: {str(e)}")
        trad_positions = np.nan * np.ones_like(pinn_positions)
        trad_time = time.time() - start_time
        trad_success = False
    
    return {
        't': t_points,
        'pinn': pinn_positions,
        'traditional': trad_positions,
        'computation_time': {'pinn': pinn_time, 'traditional': trad_time},
        'trad_success': trad_success
    }

def create_comparison_plot(results):
    fig = make_subplots(rows=2, cols=2,
                       specs=[[{"colspan": 2}, None],
                             [{"type": "scatter"}, {"type": "scatter"}]],
                       subplot_titles=('Trajectories Comparison',
                                     'Position Differences', 'Method Stability'))
    
    # Trajectories
    colors = ['red', 'blue', 'green']
    for i in range(3):
        # PINN trajectories
        fig.add_trace(
            go.Scatter(x=results['pinn'][:, i*2], y=results['pinn'][:, i*2+1],
                      mode='lines', name=f'Body {i+1} (PINN)',
                      line=dict(color=colors[i], width=2)),
            row=1, col=1
        )
        
        if results['trad_success']:
            fig.add_trace(
                go.Scatter(x=results['traditional'][:, i*2], 
                          y=results['traditional'][:, i*2+1],
                          mode='lines', name=f'Body {i+1} (Traditional)',
                          line=dict(color=colors[i], dash='dash', width=2)),
                row=1, col=1
            )
    
    # Position differences
    if results['trad_success']:
        for i in range(3):
            diff = np.sqrt(
                (results['pinn'][:, i*2] - results['traditional'][:, i*2])**2 +
                (results['pinn'][:, i*2+1] - results['traditional'][:, i*2+1])**2
            )
            fig.add_trace(
                go.Scatter(x=results['t'], y=diff,
                          name=f'Body {i+1} Difference',
                          line=dict(color=colors[i])),
                row=2, col=1
            )
    
    # Method stability
    for i in range(3):
        pinn_stability = np.sqrt(
            np.gradient(results['pinn'][:, i*2])**2 +
            np.gradient(results['pinn'][:, i*2+1])**2
        )
        fig.add_trace(
            go.Scatter(x=results['t'], y=pinn_stability,
                      name=f'Body {i+1} Stability (PINN)',
                      line=dict(color=colors[i])),
            row=2, col=2
        )
        
        if results['trad_success']:
            trad_stability = np.sqrt(
                np.gradient(results['traditional'][:, i*2])**2 +
                np.gradient(results['traditional'][:, i*2+1])**2
            )
            fig.add_trace(
                go.Scatter(x=results['t'], y=trad_stability,
                          name=f'Body {i+1} Stability (Traditional)',
                          line=dict(color=colors[i], dash='dash')),
                row=2, col=2
            )
    
    fig.update_layout(height=800, showlegend=True,
                     title_text="PINN vs Traditional Method Comparison")
    
    return fig

def main():
    st.set_page_config(page_title="Three-Body Comparison", layout="wide")
    
    st.title("Three-Body Problem: PINN vs Traditional Methods")
    st.write("""
    Compare the performance of Physics-Informed Neural Networks (PINNs) with traditional
    numerical methods for solving the three-body problem over long time periods.
    """)
    
    # Model loading
    st.sidebar.header("Model Configuration")
    uploaded_model = st.sidebar.file_uploader("Upload trained PINN model (.pth)", 
                                            type='pth')
    
    if uploaded_model is None:
        st.warning("Please upload a trained model to continue")
        return
    
    # Load model
    model = load_model(uploaded_model)
    if model is None:
        return
    
    # Time range selection
    st.sidebar.header("Time Range")
    t_start = st.sidebar.number_input("Start Time", value=0.0)
    t_end = st.sidebar.number_input("End Time", value=100.0, min_value=1.0, 
                                   max_value=1000.0)
    n_points = st.sidebar.slider("Number of Points", 100, 2000, 1000)
    
    # Compare methods
    if st.sidebar.button("Run Comparison"):
        with st.spinner("Running comparison..."):
            results = compare_methods(model, [t_start, t_end], n_points)
            
            # Create comparison plot
            fig = create_comparison_plot(results)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show computation times
            st.write("### Computation Times")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("PINN Time", f"{results['computation_time']['pinn']:.3f}s")
            with col2:
                st.metric("Traditional Time", 
                         f"{results['computation_time']['traditional']:.3f}s")
            
            if not results['trad_success']:
                st.warning("""
                Traditional method failed to compute the full trajectory.
                This often happens due to numerical instabilities in long-term predictions.
                """)

if __name__ == "__main__":
    main()
