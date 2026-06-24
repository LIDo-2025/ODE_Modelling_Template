import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

class SimpleCICR2D:
    """
    Simplified 2D CICR model with Ca (X) and IP3 (Y)
    """
    
    def __init__(self, L=128, dt=0.05, dx=1.0, D_x=0.0, D_y=0.0):
        # Grid parameters
        self.L = L
        self.dt = dt
        self.dx = dx
        
        # Diffusion coefficients
        self.D_x = D_x  # Calcium diffusion
        self.D_y = D_y  # IP3 diffusion
        
        # Model parameters
        self.a  = 0.33
        self.m2 = 20
        self.m3 = 23
        self.ka = 0.8
        self.k  = 0.8
        self.k1 = 0.8
        
        # Unified stimulation parameters
        self.stim_amplitude = 0.0      # Amplitude of stimulation
        self.stim_frequency = 0.0      # Frequency of stimulation  
        self.stim_center = (L//2, L//2) # Center of stimulation
        self.stim_radius = 5           # Radius of stimulation area
        self.stim_phase_shift = 0.0    # Phase shift for spiral induction (0 = target patterns)
        self.time = 0.0                # Internal time counter
        
        # Initialize fields
        self.X = np.ones((L, L)) * 0.4  # Calcium
        self.Y = np.ones((L, L)) * 2.7  # IP3
        
        # Laplacian kernel for diffusion
        self.laplacian_kernel = np.array([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]]) / (dx**2)
        
        # Phase map (calculated when phase_shift is set)
        self.phase_map = None
    
    def set_stimulation(self, amplitude=1.0, frequency=0.11, radius=3, phase_shift=0.0, center=None):
        """
        Unified method to set stimulation parameters
        - phase_shift = 0: target patterns (symmetric)
        - phase_shift > 0: spiral patterns (asymmetric)
        """
        self.stim_amplitude = amplitude
        self.stim_frequency = frequency
        self.stim_radius = radius
        self.stim_phase_shift = phase_shift
        
        if center is not None:
            self.stim_center = center
            
        self.time = 0.0
        
        # Calculate phase map if phase_shift is used
        if phase_shift != 0:
            self._calculate_phase_map()
        
        pattern_type = "target" if phase_shift == 0 else "spiral"
        print(f"Stimulation: {pattern_type} patterns, amp={amplitude}, freq={frequency}, "
              f"radius={radius}, phase_shift={phase_shift:.2f}π")
    
    def _calculate_phase_map(self):
        """Calculate phase shift map for spiral induction"""
        center_x, center_y = self.stim_center
        y, x = np.ogrid[-center_y:self.L-center_y, -center_x:self.L-center_x]
        distance = np.sqrt(x*x + y*y)
        angle = np.arctan2(y, x)
        
        # Create phase shift map: phase increases continuously with angle
        self.phase_map = np.zeros((self.L, self.L))
        stim_mask = distance <= self.stim_radius
        self.phase_map[stim_mask] = angle[stim_mask] * self.stim_phase_shift
    
    def apply_stimulation(self):
        """Apply the appropriate stimulation based on parameters"""
        if self.stim_amplitude <= 0:
            return
            
        center_x, center_y = self.stim_center
        y, x = np.ogrid[-center_y:self.L-center_y, -center_x:self.L-center_x]
        distance = np.sqrt(x*x + y*y)
        stim_mask = distance <= self.stim_radius
        
        # Create stimulation array of the same shape as the grid
        stimulation = np.zeros((self.L, self.L))
        
        if self.stim_phase_shift != 0 and self.phase_map is not None:
            # Phased stimulation for spirals
            stimulation[stim_mask] = self.stim_amplitude * np.sin(
                2 * np.pi * self.stim_frequency * self.time + self.phase_map[stim_mask]
            )
        else:
            # Symmetric stimulation for target patterns
            stimulation_value = self.stim_amplitude * np.sin(
                2 * np.pi * self.stim_frequency * self.time
            )
            stimulation[stim_mask] = stimulation_value
        
        # Apply to calcium field
        self.X[stim_mask] += stimulation[stim_mask] * self.dt
        self.time += self.dt
    
    def reaction_terms(self, X, Y):
        """Calculate reaction terms for the simplified model"""
        dXdt = (self.a - self.m2*X/(1+X) + 
                (self.m3*Y/(self.k1+Y))*X**2/(self.ka+X**2) + Y - self.k*X)
        dYdt = (self.m2*X/(1+X) - 
                (self.m3*Y/(self.k1+Y))*X**2/(self.ka+X**2) - Y)
        
        return dXdt, dYdt
    
    def diffusion_step(self, field, D):
        """Apply diffusion using convolution"""
        return D * ndimage.convolve(field, self.laplacian_kernel, mode='reflect')
    
    def step(self):
        """Perform one integration step"""
        # Calculate reaction terms
        dX_react, dY_react = self.reaction_terms(self.X, self.Y)
        
        # Add diffusion
        dX_diff = self.diffusion_step(self.X, self.D_x)
        dY_diff = self.diffusion_step(self.Y, self.D_y)
        
        # Update concentrations
        self.X += self.dt * (dX_react + dX_diff)
        self.Y += self.dt * (dY_react + dY_diff)
        
        # Apply stimulation
        self.apply_stimulation()
        
        # Ensure non-negative concentrations
        self.X = np.maximum(self.X, 0)
        self.Y = np.maximum(self.Y, 0)
    
    def add_initial_noise(self, x_noise=0.1, y_noise=0.05):
        """Add noise to initial conditions"""
        self.X += np.random.normal(0, x_noise, (self.L, self.L))
        self.Y += np.random.normal(0, y_noise, (self.L, self.L))


def run_simulation(stim_type="target", amplitude=1.5, frequency=0.1, radius=5, phase_shift=0.0, 
                   L=128, D_x=0.0, D_y=0.0, time_steps=5000):
    """
    Unified simulation runner
    
    Parameters:
    - stim_type: "target" or "spiral" 
    - amplitude: stimulation amplitude
    - frequency: stimulation frequency
    - radius: stimulation radius
    - phase_shift: phase shift for spirals
    """
    
    # Create model
    model = SimpleCICR2D(L=L, D_x=D_x, D_y=D_y)
    model.add_initial_noise()
    
    # Set stimulation based on type
    if stim_type == "target":
        model.set_stimulation(
            amplitude=amplitude,
            frequency=frequency,
            radius=radius,
            phase_shift=0.0  # Force no phase shift for targets
        )
        print("Running TARGET pattern simulation...")
    elif stim_type == "spiral":
        model.set_stimulation(
            amplitude=amplitude,
            frequency=frequency,
            radius=radius,
            phase_shift=phase_shift
        )
        print("Running SPIRAL pattern simulation...")
    else:
        raise ValueError("stim_type must be 'target' or 'spiral'")
    
    # Set up visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Fixed color ranges
    X_min, X_max = 0, 0.8
    Y_min, Y_max = 2, 3.0
    
    # Create plots
    im1 = ax1.imshow(model.X, cmap='hot', origin='lower', vmin=X_min, vmax=X_max)
    ax1.set_title('Calcium (X)')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(model.Y, cmap='viridis', origin='lower', vmin=Y_min, vmax=Y_max)
    ax2.set_title('IP3 (Y)')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    # Run simulation
    for step in range(time_steps):
        model.step()
        
        if step % 5 == 0:
            im1.set_data(model.X)
            im2.set_data(model.Y)
            ax1.set_title(f'Calcium (X) - Step {step}')
            ax2.set_title(f'IP3 (Y) - Step {step}')
            plt.pause(0.002)
            
        if step % 500 == 0:
            print(f"Step {step}, X range: [{model.X.min():.2f}, {model.X.max():.2f}]")
    
    plt.show()
    
    return model


# Run
if __name__ == "__main__":
    print("2D CICR Model - without or with Stimulation")
    
    run_simulation(stim_type="spiral", amplitude=1.0, frequency=0.11, 
                          radius=3, phase_shift=0.0*np.pi, D_x=0.5, D_y=0.0) # target: amplitude=1.0, frequency=0.11, radius=3, phase_shift=0.3

