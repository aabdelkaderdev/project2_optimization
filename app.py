import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from optimization import create_custom_function, newton_method, steepest_descent_fixed, steepest_descent_optimal

st.set_page_config(page_title="Optimization Algorithms Visualizer", layout="wide")

st.title("Optimization Algorithms Visualizer")
st.markdown("### By: **Abdelrahman Abdelkader** | ID: **20100254**")

st.sidebar.header("Problem Setup")

st.sidebar.markdown("### Objective Function Definition")
expr_str = st.sidebar.text_input("Enter mathematical expression for f(x, y)", value="x^2 + y^2")
st.sidebar.caption("Use 'x' and 'y', e.g., 'x^2 * sin(y) - 2x'")
try:
    obj_fn = create_custom_function(expr_str)
except Exception as e:
    st.sidebar.error(f"Error parsing function: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Algorithm Parameters")

default_x, default_y = 1.0, 1.0

start_x = st.sidebar.number_input("Starting X", value=float(default_x), step=0.1)
start_y = st.sidebar.number_input("Starting Y", value=float(default_y), step=0.1)
x0 = np.array([start_x, start_y])

algorithm = st.sidebar.selectbox("Select Optimization Algorithm", [
    "Newton's Method", 
    "Steepest Descent (Fixed Step)", 
    "Steepest Descent (Optimal Step)",
    "Compare All Three"
])

max_iter = st.sidebar.number_input("Max Iterations", min_value=1, max_value=2000, value=100)
fixed_alpha = st.sidebar.number_input("Fixed Step Size (α)", value=0.01, step=0.01) if "Fixed" in algorithm else None
tol = 1e-5

# Run Algorithms
results = {}

if algorithm == "Newton's Method" or algorithm == "Compare All Three":
    results["Newton"] = newton_method(obj_fn, x0, max_iter, tol)
if algorithm == "Steepest Descent (Fixed Step)" or algorithm == "Compare All Three":
    results["SD (Fixed)"] = steepest_descent_fixed(obj_fn, x0, alpha=fixed_alpha if fixed_alpha else 0.01, max_iter=max_iter, tol=tol)
if algorithm == "Steepest Descent (Optimal Step)" or algorithm == "Compare All Three":
    results["SD (Optimal)"] = steepest_descent_optimal(obj_fn, x0, max_iter, tol)

# Create tabs for different views
tab_visual, tab_analysis, tab_adv = st.tabs(["Visualization", "Comparative Analysis", "Advanced Details"])

with tab_visual:
    st.header(f"Visualizing $f(x, y) = {expr_str}$")
    
    # Generate Contour Plot capabilities
    pad = 1.0
    all_x = [x0[0]]
    all_y = [x0[1]]
    for path in results.values():
        all_x.extend(path[:, 0])
        all_y.extend(path[:, 1])
        
    x_min, x_max = min(np.nanmin(all_x)-pad, obj_fn.domain_x[0]), max(np.nanmax(all_x)+pad, obj_fn.domain_x[1])
    y_min, y_max = min(np.nanmin(all_y)-pad, obj_fn.domain_y[0]), max(np.nanmax(all_y)+pad, obj_fn.domain_y[1])
    
    # Cap limits for blowing up cases
    x_min, x_max = max(x_min, obj_fn.domain_x[0]-10), min(x_max, obj_fn.domain_x[1]+10)
    y_min, y_max = max(y_min, obj_fn.domain_y[0]-10), min(y_max, obj_fn.domain_y[1]+10)
    
    X_vals = np.linspace(x_min, x_max, 200)
    Y_vals = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(X_vals, Y_vals)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = obj_fn.f(np.array([X[i,j], Y[i,j]]))
            
    fig = go.Figure()
    
    # Cap Z values for better visual resolution of contours if function blows up
    z_min_clip = np.nanmin(Z)
    z_max_clip = np.nanpercentile(Z, 80)
    Z_clipped = np.clip(Z, z_min_clip, z_max_clip)

    fig.add_trace(go.Contour(
        x=X_vals, 
        y=Y_vals, 
        z=Z_clipped, 
        colorscale='Viridis', 
        opacity=0.7,
        ncontours=30,
        showscale=False
    ))
    
    # True minima
    for mx, my in obj_fn.min_points:
        fig.add_trace(go.Scatter(x=[mx], y=[my], mode='markers', marker=dict(color='yellow', symbol='star', size=15, line=dict(color='black', width=1)), name='True Minimum'))
    
    colors = {"Newton": "white", "SD (Fixed)": "orange", "SD (Optimal)": "cyan"}
    
    if algorithm != "Compare All Three":
        path = list(results.values())[0]
        alg_name = list(results.keys())[0]
        st.subheader(f"Path taken by {alg_name}")
        
        step = st.slider("Select Iteration Step (Animation slider)", 0, max(0, len(path)-1), max(0, len(path)-1))
        sub_path = path[:step+1]
        
        fig.add_trace(go.Scatter(x=sub_path[:, 0], y=sub_path[:, 1], mode='lines+markers', 
                                 line=dict(color=colors[alg_name], width=3),
                                 marker=dict(size=8, color=colors[alg_name], line=dict(color='black', width=1)),
                                 name=alg_name))
        
        # Highlight current point
        if len(sub_path) > 0:
            fig.add_trace(go.Scatter(x=[sub_path[-1, 0]], y=[sub_path[-1, 1]], mode='markers', 
                                     marker=dict(size=14, color='red', symbol='diamond-open', line=dict(color='red', width=3)), name='Current Pt'))
                                     
        st.plotly_chart(fig, use_container_width=True)
        
        if len(sub_path) > 0:
            st.write(f"**Current Point:** ({sub_path[-1, 0]:.4f}, {sub_path[-1, 1]:.4f}) | **Function Value:** {obj_fn.f(sub_path[-1]):.6f} | **Grad Norm:** {np.linalg.norm(obj_fn.grad(sub_path[-1])):.6f}")
        
    else:
        st.subheader("Comparison of Paths")
        for alg, path in results.items():
            fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode='lines+markers', 
                                     line=dict(color=colors[alg], width=2),
                                     marker=dict(size=5, line=dict(color='black', width=0.5)),
                                     name=alg))
        st.plotly_chart(fig, use_container_width=True)


with tab_analysis:
    st.header("Comparative Analysis")
    
    if algorithm == "Compare All Three":
        metrics = []
        for alg, path in results.items():
            final_pt = path[-1] if len(path) > 0 else x0
            distances = [np.linalg.norm((final_pt[0]-mx, final_pt[1]-my)) for mx, my in obj_fn.min_points]
            min_dist = min(distances) if distances else np.nan
            
            metrics.append({
                "Algorithm": alg,
                "Iterations": max(0, len(path) - 1), 
                "Final X": final_pt[0],
                "Final Y": final_pt[1],
                "Final f(x,y)": obj_fn.f(final_pt),
                "Distance to True Min": min_dist
            })
            
        df = pd.DataFrame(metrics)
        st.dataframe(df.style.highlight_min(subset=["Iterations", "Distance to True Min"], color='lightgreen'))
        
        st.markdown("### Observations on Speed and Stability")
        st.markdown('''
        - **Newton's Method** is extremely fast (often 1 iteration for quadratics) but can be unstable if the starting point is far from a minimum and the Hessian becomes non-positive definite.
        - **Steepest Descent (Fixed $\\alpha$)** is highly dependent on $\\alpha$. If $\\alpha$ is too small, it converges very slowly. If it's too large, it may diverge entirely. It tends to exhibit a prominent zig-zag pattern around narrow valleys (e.g., Rosenbrock).
        - **Steepest Descent (Optimal $\\alpha$)** calculates the best step length at each iteration using a line search. It often reaches the minimum in far fewer iterations than a fixed step size. It still exhibits zig-zagging because consecutive search directions are mathematically orthogonal in optimal steepest descent.
        ''')
    else:
        st.info("Select 'Compare All Three' from the sidebar to view full analytical metrics across all algorithms simultaneously.")
        path = list(results.values())[0]
        
        st.write("### Iterative Tracking (Step-by-Step Updates)")
        df_iters = pd.DataFrame({
            "Iteration": range(len(path)),
            "X": path[:, 0],
            "Y": path[:, 1],
            "f(X, Y)": [obj_fn.f(p) for p in path],
            "Gradient Norm": [np.linalg.norm(obj_fn.grad(p)) for p in path]
        })
        st.dataframe(df_iters)

with tab_adv:
    st.header("Advanced Analysis")
    st.markdown("""
    ### 1. Higher-Degree Functions
    This app includes a **Degree 4 (Quartic)** function with multiple local minima. Finding the optimal step size on higher-degree polynomials can sometimes involve multiple roots along the line search direction.

    ### 2. Starting Point Sensitivity 
    Test the **Himmelblau** or **Quartic** function with different initial coordinate points ($x_0$).
    You will observe that these algorithms converge to completely different local minima depending on which "basin of attraction" the initial point lies in. 

    ### 3. Parameter Impact: Initial Point vs. Learning Rate
    - The final convergent location is **heavily influenced by the starting point**. Gradient descent generally cannot escape a local minimum once inside its basin.
    - Conversely, an improperly sized **fixed $\\alpha$** can cause the algorithm to overshoot its basin and land in another basin's territory, making the result chaotic and effectively jumping the valleys.
    """)
