import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import io

class CelestialBodyPINN(torch.nn.Module):
    def __init__(self, scenario_params=None):
        super().__init__()
        
        # Network layers to match the trained model exactly
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, 256),      # network.0
            torch.nn.Tanh(),              # network.1
            torch.nn.Linear(256, 256),    # network.2
            torch.nn.Tanh(),              # network.3
            torch.nn.Linear(256, 256),    # network.4
            torch.nn.Tanh(),              # network.5
            torch.nn.Linear(256, 256),    # network.6
            torch.nn.Tanh(),              # network.7
            torch.nn.Linear(256, 6),      # network.8 (directly to output)
            # Removed the extra layers that were causing the mismatch
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
        
        # Load state dict
        state_dict = torch.load(uploaded_file, map_location=torch.device('cpu'))
        
        # Create new state dict with correct keys
        new_state_dict = {}
        layer_map = {
            'network.0': 'network.0',  # Input layer
            'network.2': 'network.2',  # First hidden layer
            'network.4': 'network.4',  # Second hidden layer
            'network.6': 'network.6',  # Third hidden layer
            'network.8': 'network.8'   # Output layer
        }
        
        for old_key in state_dict:
            if old_key in layer_map:
                new_key = layer_map[old_key]
                new_state_dict[new_key + '.weight'] = state_dict[old_key + '.weight']
                new_state_dict[new_key + '.bias'] = state_dict[old_key + '.bias']
        
        # Load the modified state dict
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        # Verify model loaded correctly
        st.success("Model loaded successfully!")
        return model, None
    except Exception as e:
        return None, str(e)
        
def create_orbit_plot(positions, scenario, time_points):
    """Create interactive orbital plot with rotating bodies"""
    
    fig = make_subplots(rows=2, cols=2,
                       specs=[[{"colspan": 2}, None],
                             [{"type": "scatter"}, {"type": "scatter"}]],
                       subplot_titles=('Orbital Trajectories',
                                     'Distances Between Bodies', 
                                     'Orbital Velocities'))
    
    # Define colors and sizes for bodies
    body_colors = ['blue', 'gray', 'red']
    body_sizes = [20, 15, 15]  # Relative sizes of bodies
    
    # Create animation frames
    frames = []
    for i in range(len(time_points)):
        frame_data = []
        
        # Add trajectory traces for each body
        for j in range(3):
            # Trajectory up to current time
            frame_data.append(
                go.Scatter(
                    x=positions[:i+1, j*2],
                    y=positions[:i+1, j*2+1],
                    mode='lines',
                    line=dict(color=body_colors[j], width=1),
                    opacity=0.5,
                    showlegend=False
                )
            )
            
            # Current position with rotating marker
            frame_data.append(
                go.Scatter(
                    x=[positions[i, j*2]],
                    y=[positions[i, j*2+1]],
                    mode='markers',
                    marker=dict(
                        size=body_sizes[j],
                        color=body_colors[j],
                        symbol='circle',
                        line=dict(color='white', width=1)
                    ),
                    name=scenario['bodies'][j]
                )
            )
        
        # Add distance plot
        r12 = np.sqrt((positions[:i+1, 0] - positions[:i+1, 2])**2 + 
                      (positions[:i+1, 1] - positions[:i+1, 3])**2)
        r13 = np.sqrt((positions[:i+1, 0] - positions[:i+1, 4])**2 + 
                      (positions[:i+1, 1] - positions[:i+1, 5])**2)
        r23 = np.sqrt((positions[:i+1, 2] - positions[:i+1, 4])**2 + 
                      (positions[:i+1, 3] - positions[:i+1, 5])**2)
                
        # For distance plots
        frame_data.append(
            go.Scatter(
                x=time_points[:i+1], 
                y=r12, 
                name=f'Distance {scenario["bodies"][0]}-{scenario["bodies"][1]}',
                line=dict(color='purple'),
                xaxis='x2',  # Use xaxis2 for the second row
                yaxis='y2'   # Use yaxis2 for the second row
            )
        )
        frame_data.append(
            go.Scatter(
                x=time_points[:i+1], 
                y=r13, 
                name=f'Distance {scenario["bodies"][0]}-{scenario["bodies"][2]}',
                line=dict(color='orange'),
                xaxis='x2',
                yaxis='y2'
            )
        )
            frame_data.append(
                go.Scatter(
                    x=time_points[:i+1], 
                    y=r23, 
                    name=f'Distance {scenario["bodies"][1]}-{scenario["bodies"][2]}',
                    line=dict(color='green'),
                    xaxis='x2',
                    yaxis='y2'
                )
            )
        
        # For velocity plots
        for j in range(3):
            frame_data.append(
                go.Scatter(
                    x=time_points[:i+1], 
                    y=vel_mag[:, j],
                    name=f'{scenario["bodies"][j]} Velocity',
                    line=dict(color=body_colors[j]),
                    xaxis='x3',  # Use xaxis3 for the third subplot
                    yaxis='y3'   # Use yaxis3 for the third subplot
                )
            )        

# For velocity plots
        for j in range(3):
            frame_data.append(
                go.Scatter(
                    x=time_points[:i+1], 
                    y=vel_mag[:, j],
                    name=f'{scenario["bodies"][j]} Velocity',
                    line=dict(color=body_colors[j]),
                    xaxis='x3',  # Use xaxis3 for the third subplot
                    yaxis='y3'   # Use yaxis3 for the third subplot
                )
            )
    # Initial empty plot
    for j in range(3):
        # Trajectory trace
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='lines',
                      line=dict(color=body_colors[j], width=1),
                      opacity=0.5,
                      showlegend=False),
            row=1, col=1
        )
        # Body position
        fig.add_trace(
            go.Scatter(x=[], y=[], mode='markers',
                      marker=dict(size=body_sizes[j], color=body_colors[j],
                                symbol='circle', line=dict(color='white', width=1)),
                      name=scenario['bodies'][j]),
            row=1, col=1
        )
    
    # Add distance traces
    for name, color in zip(['1-2', '1-3', '2-3'], ['purple', 'orange', 'green']):
        fig.add_trace(
            go.Scatter(x=[], y=[], name=f'Distance {name}',
                      line=dict(color=color)),
            row=2, col=1
        )
    
    # Add velocity traces
    for j, body in enumerate(scenario['bodies']):
        fig.add_trace(
            go.Scatter(x=[], y=[], name=f'{body} Velocity',
                      line=dict(color=body_colors[j])),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"Three-Body System: {', '.join(scenario['bodies'])}",
        showlegend=True,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 50, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Time: ', 'suffix': ' s'},
            'steps': [
                {
                    'args': [[f'frame{k}'], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': f'{time_points[k]:.1f}',
                    'method': 'animate'
                }
                for k in range(0, len(time_points), max(1, len(time_points)//10))
            ]
        }]
    )
    
    # Update axes
    fig.update_xaxes(title_text="X Position (km)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (km)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Distance (km)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Velocity (km/s)", row=2, col=2)
    
    if scenario['display_scale'] == 'log':
        fig.update_yaxes(type="log", row=2, col=1)
    
    # Add frames
    fig.frames = frames
    
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
