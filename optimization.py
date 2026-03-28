import numpy as np
import scipy.optimize as opt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

class ObjectiveFunction:
    def __init__(self, name, f, grad, hessian, domain_x, domain_y, min_points):
        self.name = name
        self.f = f
        self.grad = grad
        self.hessian = hessian
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.min_points = min_points  # list of (x,y) tuples

# 1. Quadratic Function: f(x,y) = x^2 + 2y^2
def quad_f(x): return x[0]**2 + 2*x[1]**2
def quad_g(x): return np.array([2*x[0], 4*x[1]])
def quad_h(x): return np.array([[2, 0], [0, 4]])
quadratic = ObjectiveFunction(
    "Quadratic ($f(x,y)=x^2+2y^2$)",
    quad_f, quad_g, quad_h,
    domain_x=(-5, 5), domain_y=(-5, 5),
    min_points=[(0, 0)]
)

# 2. Rosenbrock Function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
def rosen_f(x): return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
def rosen_g(x): return np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
def rosen_h(x): return np.array([
    [2 - 400*(x[1]-x[0]**2) + 800*x[0]**2, -400*x[0]],
    [-400*x[0], 200]
])
rosenbrock = ObjectiveFunction(
    "Rosenbrock",
    rosen_f, rosen_g, rosen_h,
    domain_x=(-2, 2.5), domain_y=(-1, 3.5),
    min_points=[(1, 1)]
)

# 3. Himmelblau Function
def himm_f(x): return (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2
def himm_g(x_v): 
    x, y = x_v[0], x_v[1]
    return np.array([
        4*x*(x**2+y-11) + 2*(x+y**2-7),
        2*(x**2+y-11) + 4*y*(x+y**2-7)
    ])
def himm_h(x_v):
    x, y = x_v[0], x_v[1]
    return np.array([
        [12*x**2 + 4*y - 42, 4*x + 4*y],
        [4*x + 4*y, 4*x + 12*y**2 - 26]
    ])
himmelblau = ObjectiveFunction(
    "Himmelblau",
    himm_f, himm_g, himm_h,
    domain_x=(-5, 5), domain_y=(-5, 5),
    min_points=[(3.0, 2.0), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)]
)

# 4. Quartic (Degree 4) Function: f(x,y) = x^4 - 4x^2 + y^2 + 2y
def quart_f(x_v): x, y = x_v[0], x_v[1]; return x**4 - 4*x**2 + y**2 + 2*y
def quart_g(x_v): x, y = x_v[0], x_v[1]; return np.array([4*x**3 - 8*x, 2*y + 2])
def quart_h(x_v): x, y = x_v[0], x_v[1]; return np.array([[12*x**2 - 8, 0], [0, 2]])
quartic = ObjectiveFunction(
    "Degree 4 Quartic ($f(x,y)=x^4-4x^2+y^2+2y$)",
    quart_f, quart_g, quart_h,
    domain_x=(-3, 3), domain_y=(-4, 2),
    min_points=[(np.sqrt(2), -1), (-np.sqrt(2), -1)]
)

FUNCTIONS = {
    "Quadratic": quadratic,
    "Rosenbrock": rosenbrock,
    "Himmelblau": himmelblau,
    "Degree 4 (Quartic)": quartic
}

def create_custom_function(expr_str, x_range=(-5, 5), y_range=(-5, 5)):
    x, y = sp.symbols('x y')
    # Transformations for convenience: allow x^2 -> x**2, 2x -> 2*x
    transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
    try:
        expr = parse_expr(expr_str, transformations=transformations)
    except Exception as e:
        raise ValueError(f"Could not parse expression: {e}")
        
    grad_x = sp.diff(expr, x)
    grad_y = sp.diff(expr, y)
    
    hess_xx = sp.diff(grad_x, x)
    hess_xy = sp.diff(grad_x, y)
    hess_yx = sp.diff(grad_y, x)
    hess_yy = sp.diff(grad_y, y)
    
    # Lambdify with numpy
    f_lam = sp.lambdify((x, y), expr, 'numpy')
    gx_lam = sp.lambdify((x, y), grad_x, 'numpy')
    gy_lam = sp.lambdify((x, y), grad_y, 'numpy')
    
    hxx_lam = sp.lambdify((x, y), hess_xx, 'numpy')
    hxy_lam = sp.lambdify((x, y), hess_xy, 'numpy')
    hyx_lam = sp.lambdify((x, y), hess_yx, 'numpy')
    hyy_lam = sp.lambdify((x, y), hess_yy, 'numpy')
    
    def f_wrap(v):
        res = f_lam(v[0], v[1])
        # Force float return for 0D arrays or python scalars, but keep array shape for meshgrids
        if np.isscalar(res) or (isinstance(res, np.ndarray) and res.ndim == 0):
            return float(res)
        if isinstance(res, np.ndarray):
            return res
        return np.ones_like(v[0], dtype=float) * res
        
    def grad_wrap(v):
        gx = gx_lam(v[0], v[1])
        gy = gy_lam(v[0], v[1])
        # If expression didn't have x or y, it might return a constant 0, broadcast if necessary
        return np.array([gx, gy], dtype=float)
        
    def hess_wrap(v):
        return np.array([
            [hxx_lam(v[0], v[1]), hxy_lam(v[0], v[1])],
            [hyx_lam(v[0], v[1]), hyy_lam(v[0], v[1])]
        ], dtype=float)
        
    return ObjectiveFunction(
        f"Custom: {expr_str}",
        f_wrap, grad_wrap, hess_wrap,
        domain_x=x_range, domain_y=y_range,
        min_points=[]
    )

def newton_method(obj_fn, x0, max_iter=100, tol=1e-6):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(max_iter):
        g = obj_fn.grad(x)
        if np.linalg.norm(g) < tol:
            break
        H = obj_fn.hessian(x)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse or add epsilon for singular matrix
            step = np.linalg.lstsq(H + np.eye(len(x))*1e-4, g, rcond=None)[0]
        
        x = x - step
        path.append(x.copy())
    return np.array(path)

def steepest_descent_fixed(obj_fn, x0, alpha=0.1, max_iter=100, tol=1e-6):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(max_iter):
        g = obj_fn.grad(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - alpha * g
        path.append(x.copy())
    return np.array(path)

def steepest_descent_optimal(obj_fn, x0, max_iter=100, tol=1e-6):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(max_iter):
        g = obj_fn.grad(x)
        if np.linalg.norm(g) < tol:
            break
        
        # Line search strategy using scalar minimization
        def line_obj(alpha):
            return obj_fn.f(x - alpha * g)
        
        # Optimization bounded usually between 0 and a small constant, sometimes might need larger
        res = opt.minimize_scalar(line_obj, bounds=(0, 2.0), method='bounded')
        if res.success:
            alpha = res.x
        else:
            alpha = 0.05  # Safe fallback if line search fails
            
        x = x - alpha * g
        path.append(x.copy())
    return np.array(path)
