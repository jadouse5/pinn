import streamlit as st
import torch
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

class TraditionalSolver:
    """Traditional numerical solver using scipy's odeint"""
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

def compare_methods(pinn_model, t_range, n_points=1000):
    """Compare PINN and traditional methods"""
    # Time points
    t_points = np.linspace(t_range[0], t_range[1], n_points)
    
    # PINN predictions
    t_tensor = torch.tensor(t_points, dtype=torch.float32).reshape(-1, 1)
    start_time = time.time()
    with torch.no_grad():
        pinn_positions = pinn_model(t_tensor).numpy()
    pinn_time = time.time() - start_time
    
    # Traditional method
    solver = TraditionalSolver(G=pinn_model.G, m1=pinn_model.m1, m2=pinn_model.m2, m3=pinn_model.m3)
    initial_conditions = np.concatenate([pinn_positions[0], np.zeros(6)])  # positions and velocities
    
    start_time = time.time()
    try:
        trad_solution = odeint(solver.derivatives, initial_conditions, t_points)
        trad_positions = trad_solution[:, :6]
        trad_time = time.time() - start_time
        trad_success = True
    except:
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
    """Create interactive comparison plot"""
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
        
        # Traditional method trajectories
        if results['trad_success']:
            fig.add_trace(
                go.Scatter(x=results['traditional'][:, i*2], y=results['traditional'][:, i*2+1],
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
        # PINN stability
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
        
        # Traditional method stability
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
    
    # Update layout
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
    uploaded_model = st.sidebar.file_uploader("Upload trained PINN model (.pth)", type='pth')
    
    if uploaded_model is not None:
        model = ThreeBodyPINN()
        model.load_state_dict(torch.load(uploaded_model, weights_only=True))
        model.eval()
        
        # Time range selection
        st.sidebar.header("Time Range")
        t_start = st.sidebar.number_input("Start Time", value=0.0)
        t_end = st.sidebar.number_input("End Time", value=100.0, max_value=1000.0)
        n_points = st.sidebar.slider("
