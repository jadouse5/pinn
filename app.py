import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
import time

# ---------------------------------------
# Classes and Functions
# ---------------------------------------

class TraditionalSolver:
    def __init__(self, G=6.67430e-20, m1=1.0, m2=1.0, m3=1.0):
        self.G = G
        self.m1, self.m2, self.m3 = m1, m2, m3
    
    def derivatives(self, state, t):
        # First 6 values are positions, next 6 are velocities
        positions = state[:6]
        velocities = state[6:]
        
        x1, y1 = positions[0], positions[1]
        x2, y2 = positions[2], positions[3]
        x3, y3 = positions[4], positions[5]
        
        # Calculate distances
        r12 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2) + 1e-10
        r13 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2) + 1e-10
        r23 = np.sqrt((x2 - x3)**2 + (y2 - y3)**2) + 1e-10
        
        # Accelerations
        ax1 = self.G * (self.m2 * (x2 - x1) / r12**3 + self.m3 * (x3 - x1) / r13**3)
        ay1 = self.G * (self.m2 * (y2 - y1) / r12**3 + self.m3 * (y3 - y1) / r13**3)
        
        ax2 = self.G * (self.m1 * (x1 - x2) / r12**3 + self.m3 * (x3 - x2) / r23**3)
        ay2 = self.G * (self.m1 * (y1 - y2) / r12**3 + self.m3 * (y3 - y2) / r23**3)
        
        ax3 = self.G * (self.m1 * (x1 - x3) / r13**3 + self.m2 * (x2 - x3) / r23**3)
        ay3 = self.G * (self.m1 * (y1 - y3) / r13**3 + self.m2 * (y2 - y3) / r23**3)
        
        return np.array([*velocities, ax1, ay1, ax2, ay2, ax3, ay3])
        

class CelestialBodyPINN(torch.nn.Module):
    def __init__(self, scenario_params=None):
        super().__init__()
        
        # Neural network architecture must match the trained model
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 6)
        )
        
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
        model = CelestialBodyPINN(scenario_params)
        state_dict = torch.load(uploaded_file, map_location=torch.device('cpu'))
        
        # Attempt to load state dict directly
        # If it doesn't match, try adjusting keys
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # The code snippet below attempts to map keys
            new_state_dict = {}
            layer_map = {
                'network.0': 'network.0',
                'network.2': 'network.2',
                'network.4': 'network.4',
                'network.6': 'network.6',
                'network.8': 'network.8'
            }
            for old_key in list(state_dict.keys()):
                for layer_key in layer_map:
                    if old_key.startswith(layer_key):
                        new_key = old_key.replace(layer_key, layer_map[layer_key])
                        new_state_dict[new_key] = state_dict[old_key]
                        break
            
            model.load_state_dict(new_state_dict, strict=False)
        
        model.eval()
        st.success("Model loaded successfully!")
        return model, None
    except Exception as e:
        return None, str(e)

def create_orbit_plot(positions, scenario, time_points):
    """
    Create a 3-body orbital plot with similar style to the two-body code.
    The plot includes:
    - Orbital motion (top row)
    - Distance from center (bottom left)
    - Angular position (bottom right)
    """
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None],
               [{"type": "scatter"}, {"type": "scatter"}]],
        subplot_titles=('Orbital Motion',
                        'Distance from Origin',
                        'Angular Position')
    )
    
    # Extract positions for each body
    body1_x = positions[:, 0]
    body1_y = positions[:, 1]
    body2_x = positions[:, 2]
    body2_y = positions[:, 3]
    body3_x = positions[:, 4]
    body3_y = positions[:, 5]
    
    # Add body trajectories
    fig.add_trace(
        go.Scatter(x=body1_x, y=body1_y,
                   mode='lines+markers',
                   name=scenario['bodies'][0],
                   line=dict(color='blue'),
                   marker=dict(size=10)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=body2_x, y=body2_y,
                   mode='lines+markers',
                   name=scenario['bodies'][1],
                   line=dict(color='red'),
                   marker=dict(size=8)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=body3_x, y=body3_y,
                   mode='lines+markers',
                   name=scenario['bodies'][2],
                   line=dict(color='green'),
                   marker=dict(size=8)),
        row=1, col=1
    )
    
    # Plot the center (0,0) for reference
    fig.add_trace(
        go.Scatter(x=[0], y=[0],
                   mode='markers',
                   name='Reference Point (0,0)',
                   marker=dict(size=5, color='black')),
        row=1, col=1
    )
    
    # Distance from origin
    r1 = np.sqrt(body1_x**2 + body1_y**2)
    r2 = np.sqrt(body2_x**2 + body2_y**2)
    r3 = np.sqrt(body3_x**2 + body3_y**2)
    
    fig.add_trace(
        go.Scatter(x=time_points, y=r1, 
                   name=f"{scenario['bodies'][0]} Distance",
                   line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_points, y=r2, 
                   name=f"{scenario['bodies'][1]} Distance",
                   line=dict(color='red')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_points, y=r3, 
                   name=f"{scenario['bodies'][2]} Distance",
                   line=dict(color='green')),
        row=2, col=1
    )
    
    # Angular positions
    theta1 = np.arctan2(body1_y, body1_x)
    theta2 = np.arctan2(body2_y, body2_x)
    theta3 = np.arctan2(body3_y, body3_x)
    
    fig.add_trace(
        go.Scatter(x=time_points, y=theta1, 
                   name=f"{scenario['bodies'][0]} Angle",
                   line=dict(color='blue')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=time_points, y=theta2, 
                   name=f"{scenario['bodies'][1]} Angle",
                   line=dict(color='red')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=time_points, y=theta3, 
                   name=f"{scenario['bodies'][2]} Angle",
                   line=dict(color='green')),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Three-Body System: {', '.join(scenario['bodies'])}",
        hovermode='closest'
    )
    
    fig.update_xaxes(title_text="X Position (km)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (km)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Distance (km)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Angle (rad)", row=2, col=2)
    
    return fig

def create_comparison_plot(positions_pinn, positions_trad, scenario, time_points):
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"colspan": 2}, None]],
        subplot_titles=('PINN Method', 'Traditional Method', 'Deviation Between Methods'),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    colors = ['blue', 'red', 'green']
    body_sizes = [10, 8, 8]

    # PINN trajectories
    for j in range(3):
        fig.add_trace(
            go.Scatter(
                x=positions_pinn[:, j*2],
                y=positions_pinn[:, j*2+1],
                mode='lines+markers',
                name=f'{scenario["bodies"][j]} (PINN)',
                line=dict(color=colors[j], width=2),
                marker=dict(size=body_sizes[j], symbol='circle')
            ),
            row=1, col=1
        )

    # Traditional method trajectories
    for j in range(3):
        fig.add_trace(
            go.Scatter(
                x=positions_trad[:, j*2],
                y=positions_trad[:, j*2+1],
                mode='lines+markers',
                name=f'{scenario["bodies"][j]} (Traditional)',
                line=dict(color=colors[j], width=2, dash='dot'),
                marker=dict(size=body_sizes[j], symbol='square')
            ),
            row=1, col=2
        )

    deviation = np.sqrt(np.sum((positions_pinn - positions_trad)**2, axis=1))
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=deviation,
            mode='lines',
            name='Method Deviation',
            line=dict(color='black', width=2)
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='rgb(240, 240, 240)',
        title_text=f"PINN vs Traditional Method Comparison: {', '.join(scenario['bodies'])}",
    )

    for i in range(2):
        fig.update_xaxes(title_text="X Position (km)", row=1, col=i+1)
        fig.update_yaxes(title_text="Y Position (km)", row=1, col=i+1)
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Deviation (km)", row=2, col=1)

    return fig


# ---------------------------------------
# Main Application
# ---------------------------------------

def main():
    st.set_page_config(page_title="Three-Body Orbital Simulator", layout="wide")
    st.title("Celestial Body Orbital Simulator (Three-Body with PINN)")
    st.write("""
    Explore different three-body celestial systems using a trained Physics-Informed Neural Network model.
    Upload your trained PINN model and simulate the three-body problem.
    """)

    # Initialize session state for the model
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

    st.sidebar.header("Configuration")
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
        model, error = load_model_safely(uploaded_model, scenario)
        if error:
            st.error(f"Error loading model: {error}")
            return
        st.session_state.model = model

        # Time controls
        st.sidebar.header("Time Settings")
        time_unit = st.sidebar.selectbox("Time Unit", ["Days", "Months", "Years"])
        time_value = st.sidebar.number_input(f"Number of {time_unit}", min_value=1, max_value=1000, value=1)
        time_multipliers = {
            "Days": 24 * 3600,
            "Months": 30 * 24 * 3600,
            "Years": 365 * 24 * 3600
        }
        t_end = time_value * time_multipliers[time_unit]
        
        n_points = st.sidebar.slider("Number of Points", 100, 2000, 1000)
        
        show_comparison = st.sidebar.checkbox("Compare with Traditional Method")
        if show_comparison:
            st.sidebar.warning("Traditional method may show chaotic behavior over long periods")

        # Run simulation
        if st.sidebar.button("Run Simulation"):
            with st.spinner("Running simulation..."):
                try:
                    t_points = np.linspace(0, t_end, n_points)
                    t = torch.tensor(t_points, dtype=torch.float32).reshape(-1, 1)
                    
                    # Get PINN predictions
                    with torch.no_grad():
                        positions_pinn = st.session_state.model(t).numpy()

                    if show_comparison:
                        trad_solver = TraditionalSolver(
                            G=scenario['G'],
                            m1=scenario['masses'][0],
                            m2=scenario['masses'][1],
                            m3=scenario['masses'][2]
                        )
                        
                        # Initial conditions from PINN's first output
                        initial_positions = positions_pinn[0]
                        # Assign initial velocities from scenario (approximation)
                        initial_velocities = np.zeros(6)
                        for i in range(3):
                            if i > 0:
                                v = scenario['velocities'][i]
                                # Assign velocity along y-axis for simplicity
                                initial_velocities[i*2] = 0
                                initial_velocities[i*2+1] = v
                        
                        initial_state = np.concatenate([initial_positions, initial_velocities])
                        
                        # Solve traditionally with odeint
                        t_dense = np.linspace(0, t_end, n_points*10)
                        trad_solution = odeint(trad_solver.derivatives, initial_state, t_dense)
                        indices = np.linspace(0, len(t_dense)-1, n_points, dtype=int)
                        positions_trad = trad_solution[indices, :6]
                        
                        # Comparison plot
                        fig = create_comparison_plot(positions_pinn, positions_trad, scenario, t_points)
                    else:
                        # Single orbit plot
                        fig = create_orbit_plot(positions_pinn, scenario, t_points)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display system info
                    st.write("### System Information")
                    col1, col2, col3 = st.columns(3)
                    
                    for i, (body, col) in enumerate(zip(scenario['bodies'], [col1, col2, col3])):
                        with col:
                            st.write(f"**{body}**")
                            st.write(f"Mass: {scenario['masses'][i]:.2e} kg")
                            st.write(f"Distance: {scenario['distances'][i]:.2e} km")
                            if i > 0:
                                st.write(f"Velocity: {scenario['velocities'][i]:.2e} km/s")
                    
                    if show_comparison:
                        deviation = np.sqrt(np.sum((positions_pinn - positions_trad)**2, axis=1))
                        chaos_time_index = np.where(deviation > 1e3)[0]
                        chaos_time = t_points[chaos_time_index[0]] if len(chaos_time_index) > 0 else None
                        
                        st.write("### Chaos Analysis")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Maximum Deviation", f"{deviation.max():.2e} km")
                        with col2:
                            if chaos_time is not None:
                                st.metric("Time to Chaos", f"{chaos_time/3600/24:.1f} days")
                            else:
                                st.metric("Time to Chaos", "Not reached")
                
                except Exception as e:
                    st.error(f"Error during simulation: {str(e)}")
                    st.error("Full error:", exc_info=True)
    else:
        st.warning("Please upload a trained model to continue")

if __name__ == "__main__":
    main()
