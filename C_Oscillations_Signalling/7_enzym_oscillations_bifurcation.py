import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

def model(t, variables, a1, b1, a2, b2, k_max, K_m, k_i, n, m, q):
    """Coupled system with feedback inhibition"""
    S, P = variables
    
    enzymatic_rate = (k_max * S**n) / (K_m**m + S**m) / (1 + k_i * P**q)
    
    dSdt = a1 - b1 * S - enzymatic_rate
    dPdt = a2 - b2 * P + enzymatic_rate
    
    return [dSdt, dPdt]


def find_asymptotic_extrema(a1_value, model_func, initial_conditions, 
                            params_dict, t_transient=500, t_analysis=200):
    """
    Simulate the system and extract min/max values in asymptotic regime.
    
    This simplified version just takes the min/max of the time series,
    which is more robust than peak detection for bifurcation diagrams.
    """
    t_total = t_transient + t_analysis
    
    params = (a1_value, params_dict['b1'], params_dict['a2'], params_dict['b2'],
              params_dict['k_max'], params_dict['K_m'], params_dict['k_i'],
              params_dict['n'], params_dict['m'], params_dict['q'])
    
    # Solve ODE
    sol = solve_ivp(model_func, [0, t_total], initial_conditions, 
                    args=params, method='LSODA', dense_output=True,
                    rtol=1e-8, atol=1e-10)
    
    # Evaluate in asymptotic regime
    t_eval = np.linspace(t_transient, t_total, 2000)
    solution = sol.sol(t_eval)
    
    S_asymptotic = solution[0, :]
    P_asymptotic = solution[1, :]
    
    # Simply take min and max of the time series
    S_min = np.min(S_asymptotic)
    S_max = np.max(S_asymptotic)
    P_min = np.min(P_asymptotic)
    P_max = np.max(P_asymptotic)
    
    return S_min, S_max, P_min, P_max


def create_bifurcation_diagram(a1_range, model_func, initial_conditions, 
                               params_dict, n_points=100, t_transient=500,
                               t_analysis=200):
    """Create bifurcation diagram data."""
    
    a1_values = np.linspace(a1_range[0], a1_range[1], n_points)
    
    S_mins = []
    S_maxs = []
    P_mins = []
    P_maxs = []
    
    # Use continuation
    current_ic = initial_conditions.copy()
    
    print("Computing bifurcation diagram...")
    print(f"Scanning a1 from {a1_range[0]:.4f} to {a1_range[1]:.4f}")
    print(f"Number of points: {n_points}")
    print("-" * 60)
    
    for i, a1 in enumerate(a1_values):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Progress: {i+1:3d}/{n_points} | a1 = {a1:.6f}")
        
        try:
            S_min, S_max, P_min, P_max = find_asymptotic_extrema(
                a1, model_func, current_ic, params_dict,
                t_transient=t_transient, t_analysis=t_analysis
            )
            
            S_mins.append(S_min)
            S_maxs.append(S_max)
            P_mins.append(P_min)
            P_maxs.append(P_max)
            
            # Update initial conditions for continuation
            current_ic = [(S_min + S_max)/2, (P_min + P_max)/2]
            
        except Exception as e:
            print(f"Warning: Failed at a1 = {a1:.6f}: {e}")
            # Use previous values
            if len(S_mins) > 0:
                S_mins.append(S_mins[-1])
                S_maxs.append(S_maxs[-1])
                P_mins.append(P_mins[-1])
                P_maxs.append(P_maxs[-1])
            else:
                S_mins.append(np.nan)
                S_maxs.append(np.nan)
                P_mins.append(np.nan)
                P_maxs.append(np.nan)
    
    print("-" * 60)
    print("Bifurcation diagram computation complete!")
    
    return (np.array(a1_values), 
            np.array(S_mins), np.array(S_maxs),
            np.array(P_mins), np.array(P_maxs))


def plot_bifurcation_diagram(a1_values, S_mins, S_maxs, P_mins, P_maxs, 
                             save_filename='bifurcation_diagram_corrected.png',
                             oscillation_range=None):
    """
    Plot the bifurcation diagram with BOTH min and max clearly visible.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Highlight oscillation region if provided
    if oscillation_range is not None:
        ax1.axvspan(oscillation_range[0], oscillation_range[1], 
                   alpha=0.15, color='yellow', label='Expected oscillations')
        ax2.axvspan(oscillation_range[0], oscillation_range[1], 
                   alpha=0.15, color='yellow', label='Expected oscillations')
    
    # Plot substrate bifurcation diagram
    # Plot minima in BLUE and maxima in RED
    ax1.plot(a1_values, S_mins, 'o', color='blue', markersize=4, 
             alpha=0.6, label='S min', markeredgewidth=0)
    ax1.plot(a1_values, S_maxs, 'o', color='red', markersize=4, 
             alpha=0.6, label='S max', markeredgewidth=0)
    
    ax1.set_xlabel('Parameter a1', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Substrate (S)', fontsize=13, fontweight='bold')
    ax1.set_title('Bifurcation Diagram - Substrate Concentration', 
                  fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='best')
    ax1.tick_params(labelsize=11)
    
    # Plot product bifurcation diagram
    ax2.plot(a1_values, P_mins, 'o', color='blue', markersize=4, 
             alpha=0.6, label='P min', markeredgewidth=0)
    ax2.plot(a1_values, P_maxs, 'o', color='red', markersize=4, 
             alpha=0.6, label='P max', markeredgewidth=0)
    
    ax2.set_xlabel('Parameter a1', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Product (P)', fontsize=13, fontweight='bold')
    ax2.set_title('Bifurcation Diagram - Product Concentration', 
                  fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11, loc='best')
    ax2.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as: {save_filename}")
    
    return fig


def analyze_oscillations(a1_values, S_mins, S_maxs, P_mins, P_maxs):
    """Analyze where oscillations occur."""
    
    S_amplitude = S_maxs - S_mins
    P_amplitude = P_maxs - P_mins
    
    # Threshold for considering it an oscillation
    threshold = 0.01
    
    S_oscillating = S_amplitude > threshold
    P_oscillating = P_amplitude > threshold
    
    print("\n" + "="*70)
    print("OSCILLATION ANALYSIS")
    print("="*70)
    
    # Find oscillatory regions
    if np.any(S_oscillating):
        osc_indices = np.where(S_oscillating)[0]
        a1_osc_start = a1_values[osc_indices[0]]
        a1_osc_end = a1_values[osc_indices[-1]]
        print(f"\nSubstrate oscillations detected:")
        print(f"  Range: a1 ∈ [{a1_osc_start:.4f}, {a1_osc_end:.4f}]")
        print(f"  Number of oscillating points: {len(osc_indices)}/{len(a1_values)}")
        
        # Max amplitude
        max_amp_idx = np.argmax(S_amplitude)
        print(f"  Maximum amplitude: {S_amplitude[max_amp_idx]:.4f} at a1 = {a1_values[max_amp_idx]:.4f}")
    else:
        print("\nNo substrate oscillations detected (all steady state)")
    
    if np.any(P_oscillating):
        osc_indices = np.where(P_oscillating)[0]
        a1_osc_start = a1_values[osc_indices[0]]
        a1_osc_end = a1_values[osc_indices[-1]]
        print(f"\nProduct oscillations detected:")
        print(f"  Range: a1 ∈ [{a1_osc_start:.4f}, {a1_osc_end:.4f}]")
        print(f"  Number of oscillating points: {len(osc_indices)}/{len(a1_values)}")
        
        # Max amplitude
        max_amp_idx = np.argmax(P_amplitude)
        print(f"  Maximum amplitude: {P_amplitude[max_amp_idx]:.4f} at a1 = {a1_values[max_amp_idx]:.4f}")
    else:
        print("\nNo product oscillations detected (all steady state)")
    
    print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("CORRECTED BIFURCATION ANALYSIS")
    print("Enzymatic Model with Feedback Inhibition")
    print("="*70)
    
    # Model parameters
    S_0 = 2.5
    P_0 = 0.6
    initial_conditions = [S_0, P_0]
    
    params = {
        'b1': 0.18,
        'b2': 0.05,
        'a2': 0.02,
        'k_max': 25.0,
        'K_m': 0.7,
        'k_i': 0.06,
        'n': 1,
        'm': 3,
        'q': 2.8
    }
    
    # Bifurcation parameter range
    a1_min = 0.63
    a1_max = 0.72
    n_points = 150
    
    # Simulation parameters
    t_transient = 500
    t_analysis = 200
    
    print(f"\nParameters:")
    print(f"  a1 range: [{a1_min}, {a1_max}]")
    print(f"  Points: {n_points}")
    print(f"  Initial conditions: S0={S_0}, P0={P_0}")
    print()
    
    # Create bifurcation diagram
    a1_vals, S_mins, S_maxs, P_mins, P_maxs = create_bifurcation_diagram(
        [a1_min, a1_max],
        model,
        initial_conditions,
        params,
        n_points=n_points,
        t_transient=t_transient,
        t_analysis=t_analysis
    )
    
    # Plot
    plot_img = plot_bifurcation_diagram(a1_vals, S_mins, S_maxs, P_mins, P_maxs,
                            save_filename='bifurcation_diagram_oscillations.png',
                            oscillation_range=[0.63, 0.72])
    
    # Analyze
    analyze_oscillations(a1_vals, S_mins, S_maxs, P_mins, P_maxs)
    
    # Print some sample values
    print("\nSample values:")
    print("a1      | S_min  | S_max  | P_min  | P_max  ")
    print("-" * 55)
    for i in [0, 10, 50, 100, 149]:
        print(f"{a1_vals[i]:.4f} | {S_mins[i]:.4f} | {S_maxs[i]:.4f} | {P_mins[i]:.4f} | {P_maxs[i]:.4f}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    plt.show()

    plot_img.savefig("/Users/geroldbaier/Documents/GitHub/ODE_Modelling_Template/C_Oscillations_Signalling/oscillator_bifurcations.png")
