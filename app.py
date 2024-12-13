import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime


class CelestialBodyPINN(torch.nn.Module):
    def __init__(self, scenario_params=None):
        super().__init__()
        # Standard architecture to match trained model
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 6)
        )
        
        # Initialize physics parameters
        if scenario_params is None:
            self.G = 6.67430e-20
            self.m1 = 5.972e24
            self.m2 = 7.342e22
            self.m3 = 6.39e23
        else:
            self.G = scenario_params['G']
            self.m1 = scenario_params['masses'][0]
            self.m2 = scenario_params['masses'][1]
            self.m3 = scenario_params['masses'][2]

    def forward(self, t):
        return self.network(t)

def load_model_safely(uploaded_file, scenario_params):
    """Safely load the model with error handling"""
    try:
        # Create model instance
        model = CelestialBodyPINN(scenario_params)
        
        # Read file bytes
        bytes_data = uploaded_file.getvalue()
        buffer = io.BytesIO(bytes_data)
        
        # Load state dict
        state_dict = torch.load(buffer, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, None
    except Exception as e:
        return None, str(e)

def create_orbit_plot(positions, scenario, time_points):
    """Create interactive orbital plot using plotly"""
    
    fig = make_subplots(rows=2, cols=2,
                       specs=[[{"colspan": 2}, None],
                             [{"type": "scatter"}, {"type": "scatter"}]],
                       subplot_titles=('Orbital Trajectories',
                                     'Distances Between Bodies', 
                                     'Orbital Velocities'))
    
    colors = ['blue', 'gray', 'red']
    
    # Plot trajectories
    for i, body in enumerate(scenario['bodies']):
        fig.add_trace(
            go.Scatter(x=positions[:, i*2], y=positions[:, i*2+1],
                      mode='lines+markers',
                      name=body,
                      line=dict(color=colors[i], width=2),
                      marker=dict(size=8)),
            row=1, col=1
        )
    
    # Plot distances between bodies
    r12 = np.sqrt((positions[:, 0] - positions[:, 2])**2 + 
                  (positions[:, 1] - positions[:, 3])**2)
    r13 = np.sqrt((positions[:, 0] - positions[:, 4])**2 + 
                  (positions[:, 1] - positions[:, 5])**2)
    r23 = np.sqrt((positions[:, 2] - positions[:, 4])**2 + 
                  (positions[:, 3] - positions[:, 5])**2)
    
    fig.add_trace(
        go.Scatter(x=time_points, y=r12, name=f'Distance {scenario["bodies"][0]}-{scenario["bodies"][1]}'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_points, y=r13, name=f'Distance {scenario["bodies"][0]}-{scenario["bodies"][2]}'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_points, y=r23, name=f'Distance {scenario["bodies"][1]}-{scenario["bodies"][2]}'),
        row=2, col=1
    )
    
    # Calculate and plot velocities
    dt = time_points[1] - time_points[0]
    velocities = np.gradient(positions, dt, axis=0)
    vel_mag = np.sqrt(velocities[:, ::2]**2 + velocities[:, 1::2]**2)
    
    for i, body in enumerate(scenario['bodies']):
        fig.add_trace(
            go.Scatter(x=time_points, y=vel_mag[:, i],
                      name=f'{body} Velocity'),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"Three-Body System: {', '.join(scenario['bodies'])}",
        showlegend=True,
    )
    
    if scenario['display_scale'] == 'log':
        fig.update_yaxes(type="log", row=2, col=1)
    
    return fig

def main():
    st.set_page_config(page_title="Celestial Body Simulator", layout="wide")
    
    st.title("Celestial Body Orbital Simulator")
    st.write("""
    Explore different three-body celestial systems using Physics-Informed Neural Networks.
    Select a predefined scenario or create your own!
    """)
    
    # Initialize session state if needed
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    # Predefined scenarios
    scenarios = {
        'Earth-Moon-Mars': {
            'bodies': ['Earth', 'Moon', 'Mars'],
            'masses': [5.972e24, 7.342e22, 6.39e23],
            'distances': [0, 384400, 225e6],
            'velocities': [29.78, 1.022, 24.077],
            'G': 6.67430e-20,
            'time_scale': 365*24*3600,
            'display_scale': 'log',
            'description': 'Classical Earth-Moon-Mars system'
        },
        'Sun-Earth-Jupiter': {
            'bodies': ['Sun', 'Earth', 'Jupiter'],
            'masses': [1.989e30, 5.972e24, 1.898e27],
            'distances': [0, 149.6e6, 778.5e6],
            'velocities': [0, 29.78, 13.07],
            'G': 6.67430e-20,
            'time_scale': 365*24*3600*12,
            'display_scale': 'log',
            'description': 'Major solar system bodies'
        },
        'Earth-Moon-Satellite': {
            'bodies': ['Earth', 'Moon', 'Satellite'],
            'masses': [5.972e24, 7.342e22, 1000],
            'distances': [0, 384400, 42164],
            'velocities': [0, 1.022, 3.075],
            'G': 6.67430e-20,
            'time_scale': 28*24*3600,
            'display_scale': 'linear',
            'description': 'Earth-Moon system with geostationary satellite'
        }
    }
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Scenario selection
    scenario_name = st.sidebar.selectbox("Select Scenario", list(scenarios.keys()))
    scenario = scenarios[scenario_name]
    
    st.sidebar.markdown(f"**Description:** {scenario['description']}")
    
    # Model loading
    uploaded_model = st.sidebar.file_uploader(
        "Upload trained PINN model (.pth)", 
        type='pth',
        help="Upload your trained PINN model"
    )
    
    if uploaded_model is not None:
        # Load model
        model, error = load_model_safely(uploaded_model, scenario)
        
        if error:
            st.error(f"Error loading model: {error}")
            return
        
        st.session_state.model = model
        
        # Simulation parameters
        st.sidebar.header("Simulation Parameters")
        t_end = st.sidebar.number_input(
            "Simulation Time (seconds)", 
            value=float(scenario['time_scale']),
            format="%.2e"
        )
        n_points = st.sidebar.slider("Number of Points", 100, 2000, 1000)
        
        # Run simulation
        if st.sidebar.button("Run Simulation"):
            with st.spinner("Running simulation..."):
                try:
                    # Generate time points
                    t_points = np.linspace(0, t_end, n_points)
                    t = torch.tensor(t_points, dtype=torch.float32).reshape(-1, 1)
                    
                    # Get predictions
                    with torch.no_grad():
                        positions = model(t).numpy()
                    
                    # Create plots
                    fig = create_orbit_plot(positions, scenario, t_points)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display system information
                    st.write("### System Information")
                    col1, col2, col3 = st.columns(3)
                    
                    for i, (body, col) in enumerate(zip(scenario['bodies'], [col1, col2, col3])):
                        with col:
                            st.write(f"**{body}**")
                            st.write(f"Mass: {scenario['masses'][i]:.2e} kg")
                            st.write(f"Distance: {scenario['distances'][i]:.2e} km")
                            if i > 0:
                                st.write(f"Velocity: {scenario['velocities'][i]:.2e} km/s")
                    
                except Exception as e:
                    st.error(f"Error during simulation: {str(e)}")
    else:
        st.warning("Please upload a trained model to continue")

if __name__ == "__main__":
    main()
