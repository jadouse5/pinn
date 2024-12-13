import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from datetime import datetime

# Three-Body PINN Model
class ThreeBodyPINN(torch.nn.Module):
    def __init__(self, hidden_layers=[128, 128, 128, 128]):
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

# Predefined scenarios
class ThreeBodyScenarios:
    @staticmethod
    def get_scenarios():
        return {
            'Figure Eight': {
                'masses': [1.0, 1.0, 1.0],
                'G': 1.0,
                'description': 'Three equal masses orbiting in a figure-8 pattern'
            },
            'Sun-Jupiter-Asteroid': {
                'masses': [1.0, 0.000954786, 1e-10],
                'G': 39.5,
                'description': 'Sun-Jupiter-Asteroid system showing orbital resonances'
            },
            'Earth-Moon-Satellite': {
                'masses': [1.0, 0.0123, 1e-15],
                'G': 6.67430e-11,
                'description': 'Earth-Moon-Satellite system showing Lagrange points'
            },
            'Lagrange Equilibrium': {
                'masses': [1.0, 1.0, 1.0],
                'G': 1.0,
                'description': 'Stable equilateral triangle configuration'
            },
            'Custom': {
                'masses': [1.0, 1.0, 1.0],
                'G': 1.0,
                'description': 'Define your own masses and gravitational constant'
            }
        }

def predict_trajectories(model, t_range, n_points=1000):
    """Get predictions from the PINN model"""
    t_points = np.linspace(t_range[0], t_range[1], n_points)
    t = torch.tensor(t_points, dtype=torch.float32).reshape(-1, 1)
    
    with torch.no_grad():
        positions = model(t).numpy()
    
    return t_points, positions

def create_interactive_plot(t, positions, masses):
    """Create interactive Plotly visualization"""
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=2,
                       specs=[[{"colspan": 2}, None],
                             [{"type": "scatter"}, {"type": "scatter"}]],
                       subplot_titles=('Trajectories', 'Distances Between Bodies', 
                                     'Relative Velocities'))
    
    # Colors for each body
    colors = ['red', 'blue', 'green']
    names = [f'Body {i+1} (m={masses[i]:.2e})' for i in range(3)]
    
    # Add trajectories
    for i in range(3):
        fig.add_trace(
            go.Scatter(x=positions[:, i*2], y=positions[:, i*2+1],
                      mode='lines+markers',
                      name=names[i],
                      line=dict(color=colors[i]),
                      marker=dict(color=colors[i], size=8)),
            row=1, col=1
        )
    
    # Calculate and plot distances between bodies
    r12 = np.sqrt((positions[:, 0] - positions[:, 2])**2 + 
                  (positions[:, 1] - positions[:, 3])**2)
    r13 = np.sqrt((positions[:, 0] - positions[:, 4])**2 + 
                  (positions[:, 1] - positions[:, 5])**2)
    r23 = np.sqrt((positions[:, 2] - positions[:, 4])**2 + 
                  (positions[:, 3] - positions[:, 5])**2)
    
    fig.add_trace(
        go.Scatter(x=t, y=r12, name='Distance 1-2', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=r13, name='Distance 1-3', line=dict(color='orange')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=r23, name='Distance 2-3', line=dict(color='brown')),
        row=2, col=1
    )
    
    # Calculate and plot relative velocities
    dt = t[1] - t[0]
    vel_x = np.gradient(positions, dt, axis=0)
    vel_mag = np.sqrt(vel_x[:, ::2]**2 + vel_x[:, 1::2]**2)
    
    for i in range(3):
        fig.add_trace(
            go.Scatter(x=t, y=vel_mag[:, i], 
                      name=f'Velocity {i+1}',
                      line=dict(color=colors[i])),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Three-Body System Analysis",
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="X Position", row=1, col=1)
    fig.update_yaxes(title_text="Y Position", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Distance", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_yaxes(title_text="Velocity Magnitude", row=2, col=2)
    
    return fig

def main():
    st.set_page_config(page_title="Three-Body Explorer", layout="wide")
    
    st.title("Three-Body System Explorer")
    st.write("""
    Explore different three-body configurations using Physics-Informed Neural Networks.
    Visualize trajectories, distances, and velocities for various scenarios.
    """)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Model loading
    uploaded_model = st.sidebar.file_uploader("Upload trained PINN model (.pth)", type='pth')
    if uploaded_model is None:
        st.warning("Please upload a trained model to continue")
        return
    
    # Load model
    model = ThreeBodyPINN()
    model.load_state_dict(torch.load(uploaded_model, map_location=torch.device('cpu')))
    model.eval()
    
    # Scenario selection
    scenarios = ThreeBodyScenarios.get_scenarios()
    scenario_name = st.sidebar.selectbox("Select Scenario", list(scenarios.keys()))
    scenario = scenarios[scenario_name]
    
    st.sidebar.markdown(f"**Description:** {scenario['description']}")
    
    # Time parameters
    st.sidebar.header("Time Parameters")
    t_start = st.sidebar.number_input("Start Time", value=0.0)
    t_end = st.sidebar.number_input("End Time", value=10.0)
    n_points = st.sidebar.slider("Number of Points", 100, 2000, 1000)
    
    # Mass configuration
    st.sidebar.header("Mass Configuration")
    if scenario_name == 'Custom':
        m1 = st.sidebar.number_input("Mass 1", value=1.0, format="%.8f")
        m2 = st.sidebar.number_input("Mass 2", value=1.0, format="%.8f")
        m3 = st.sidebar.number_input("Mass 3", value=1.0, format="%.8f")
        G = st.sidebar.number_input("Gravitational Constant", value=1.0, format="%.8f")
    else:
        m1, m2, m3 = scenario['masses']
        G = scenario['G']
        st.sidebar.write(f"Mass 1: {m1:.2e}")
        st.sidebar.write(f"Mass 2: {m2:.2e}")
        st.sidebar.write(f"Mass 3: {m3:.2e}")
        st.sidebar.write(f"G: {G:.2e}")
    
    # Update model parameters
    model.m1, model.m2, model.m3 = m1, m2, m3
    model.G = G
    
    # Run simulation button
    if st.sidebar.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            # Get predictions
            t, positions = predict_trajectories(model, [t_start, t_end], n_points)
            
            # Create interactive plot
            fig = create_interactive_plot(t, positions, [m1, m2, m3])
            st.plotly_chart(fig, use_container_width=True)
            
            # Save results
            results_df = pd.DataFrame({
                'Time': np.repeat(t, 3),
                'Body': ['Body 1'] * len(t) + ['Body 2'] * len(t) + ['Body 3'] * len(t),
                'X': np.concatenate([positions[:, 0], positions[:, 2], positions[:, 4]]),
                'Y': np.concatenate([positions[:, 1], positions[:, 3], positions[:, 5]])
            })
            
            # Download results
            st.download_button(
                "Download Results CSV",
                results_df.to_csv(index=False).encode('utf-8'),
                f"three_body_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                key='download-csv'
            )

if __name__ == "__main__":
    main()
