import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
import time
from datetime import datetime
import io

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
        r12 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2) + 1e-10  # Add small number to avoid division by zero
        r13 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2) + 1e-10
        r23 = np.sqrt((x2 - x3)**2 + (y2 - y3)**2) + 1e-10
        
        # Calculate accelerations
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
    """Create interactive orbital plot"""
    
    fig = make_subplots(rows=2, cols=2,
                       specs=[[{"colspan": 2}, None],
                             [{"type": "scatter"}, {"type": "scatter"}]],
                       subplot_titles=('Orbital Motion',
                                     'Distance from Center', 
                                     'Angular Position'))
    
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
    
    # Distance from center for each body
    r1 = np.sqrt(body1_x**2 + body1_y**2)
    r2 = np.sqrt(body2_x**2 + body2_y**2)
    r3 = np.sqrt(body3_x**2 + body3_y**2)
    
    # Plot distances
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
    
    # Angular position for each body
    theta1 = np.arctan2(body1_y, body1_x)
    theta2 = np.arctan2(body2_y, body2_x)
    theta3 = np.arctan2(body3_y, body3_x)
    
    # Plot angular positions
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

    # Colors and sizes
    colors = ['blue', 'gray', 'red']
    body_sizes = [20, 15, 15]

    # Add PINN trajectories (left plot)
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

    # Add traditional method trajectories (right plot)
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

    # Calculate and plot deviation
    deviation = np.sqrt(np.sum((positions_pinn - positions_trad)**2, axis=1))
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=deviation,
            mode='lines',
            name='Method Deviation',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='rgb(240, 240, 240)',
        title_text=f"PINN vs Traditional Method Comparison: {', '.join(scenario['bodies'])}",
    )

    # Update axes
    for i in range(2):
        fig.update_xaxes(title_text="X Position (km)", row=1, col=i+1, showgrid=True, gridwidth=1, gridcolor='white')
        fig.update_yaxes(title_text="Y Position (km)", row=1, col=i+1, showgrid=True, gridwidth=1, gridcolor='white')
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Deviation (km)", row=2, col=1)

    return fig
                        
def main():
    try:
        st.set_page_config(page_title="Celestial Body Simulator", layout="wide")
        
        st.title("Celestial Body Orbital Simulator")
        st.write("""
        Explore different three-body celestial systems using Physics-Informed Neural Networks.
        Select a predefined scenario or create your own!
        """)

        # Initialize session state
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
            
            # Time controls
            st.sidebar.header("Time Settings")
            time_unit = st.sidebar.selectbox(
                "Time Unit", 
                ["Days", "Months", "Years"]
            )
            
            time_value = st.sidebar.number_input(
                f"Number of {time_unit}", 
                min_value=1, 
                max_value=1000, 
                value=1
            )
            
            # Convert to seconds based on unit
            time_multipliers = {
                "Days": 24 * 3600,
                "Months": 30 * 24 * 3600,
                "Years": 365 * 24 * 3600
            }
            t_end = time_value * time_multipliers[time_unit]
            
            n_points = st.sidebar.slider("Number of Points", 100, 2000, 1000)
            
            # Comparison settings
            show_comparison = st.sidebar.checkbox("Compare with Traditional Method")
            
            if show_comparison:
                st.sidebar.warning("Traditional method may show chaotic behavior over long periods")
            
            # Run simulation button
            if st.sidebar.button("Run Simulation"):
                with st.spinner("Running simulation..."):
                    try:
                        # Generate time points
                        t_points = np.linspace(0, t_end, n_points)
                        t = torch.tensor(t_points, dtype=torch.float32).reshape(-1, 1)
                        
                        # Get PINN predictions
                        with torch.no_grad():
                            positions_pinn = model(t).numpy()
                        
                        if show_comparison:
                            # Get traditional method predictions
                            trad_solver = TraditionalSolver(
                                G=scenario['G'],
                                m1=scenario['masses'][0],
                                m2=scenario['masses'][1],
                                m3=scenario['masses'][2]
                            )
                            
                            # Set up initial conditions with proper velocities
                            initial_positions = positions_pinn[0]
                            initial_velocities = np.zeros(6)  # Initialize velocities
                            for i in range(3):
                                if i > 0:  # Skip first body (usually central body)
                                    v = scenario['velocities'][i]
                                    initial_velocities[i*2] = 0  # vx
                                    initial_velocities[i*2+1] = v  # vy
                            
                            initial_state = np.concatenate([initial_positions, initial_velocities])
                            
                            # Solve using odeint with smaller time steps for stability
                            t_dense = np.linspace(0, t_end, n_points * 10)  # Use more points for integration
                            trad_solution = odeint(trad_solver.derivatives, initial_state, t_dense)
                            
                            # Downsample the solution to match PINN points
                            indices = np.linspace(0, len(t_dense)-1, n_points, dtype=int)
                            positions_trad = trad_solution[indices, :6]  # Take only positions
                            
                            # Create comparison plot
                            fig = create_comparison_plot(positions_pinn, positions_trad, scenario, t_points)
                        else:
                            # Create single plot
                            fig = create_orbit_plot(positions_pinn, scenario, t_points)
                        
                        # Display plot
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
                        
                        # Add chaos analysis if comparison is enabled
                        if show_comparison:
                            deviation = np.sqrt(np.sum((positions_pinn - positions_trad)**2, axis=1))
                            chaos_time = t_points[np.where(deviation > 1e3)[0][0]] if np.any(deviation > 1e3) else None
                            
                            st.write("### Chaos Analysis")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Maximum Deviation", f"{deviation.max():.2e} km")
                            with col2:
                                if chaos_time:
                                    st.metric("Time to Chaos", f"{chaos_time/3600/24:.1f} days")
                                else:
                                    st.metric("Time to Chaos", "Not reached")
                        
                    except Exception as e:
                        st.error(f"Error during simulation: {str(e)}")
                        st.error("Full error:", exc_info=True)
        else:
            st.warning("Please upload a trained model to continue")
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Full error:", exc_info=True)


if __name__ == "__main__":
    main()
