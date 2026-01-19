# app.py (Opdo simulation backend, JSON version)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import traceback, os
import json
import matplotlib
from optiland.materials import AbbeMaterial
import optiland.backend as be


from optiland import optic, analysis, optimization
from optiland.fileio import load_zemax_file, save_optiland_file

app = Flask(__name__)
CORS(app)
def modify_thickness(lens,found):
    Found=False
    for f_no, (Hx, Hy) in enumerate(lens.fields.get_field_coords()):
        lens.trace(
            Hx=Hx, Hy=Hy,
            wavelength=lens.primary_wavelength,
            num_rays=10,
            distribution="line_y"
        )
        rays = np.array(lens.surface_group.z)
        Mat=np.diff(rays,axis=0)
        Neg=np.argwhere(Mat < 0)
        if len(Neg)!=0:
            print('thickness modified',flush=True)
            Found=True
            thickness=lens.surface_group.get_thickness(int(Neg[0][0]))
            lens.set_thickness(thickness+.1,int(Neg[0][0]))

            return modify_thickness(lens,True)
    if not(Found):
        return lens,found
def optimize_opt(lens, config):
    """
    Optimize a lens system using a JSON-structured configuration.

    Parameters
    ----------
    lens : optic.Optic
        The lens system to optimize
    config : dict
        JSON-structured optimization configuration with the following schema:
        {
            "operands": [
                {
                    "type": str - Operand type: "OPD_difference", "f1", "f2",
                                  "real_y_intercept", "real_z_intercept", "rms_spot_size",
                    "target": float - Target value for this operand,
                    "weight": float - Weight/importance of this operand,
                    "lens_number": int - (optional) Lens number for real_y_intercept, real_z_intercept,
                    "side": str - (optional) "front", "back", or "image_plane" for real_y/z_intercept,
                    "surface_number": int - (optional) Alternative to lens_number+side. Use -1 for image plane,
                    "num_rays": int - (optional) Number of rays for rms_spot_size. Default: 5,
                    "distribution": str - (optional) Ray distribution for rms_spot_size. Default: "hexapolar"
                },
                ... (additional operands)
            ],
            "variables": [
                {
                    "type": str - Variable type: "radius", "thickness", "asphere_coeff", etc.,
                    "lens_number": int - Lens number (1-indexed),
                    "side": str - "front" or "back" surface of the lens,
                    "min_value": float - (optional) Minimum allowed value,
                    "max_value": float - (optional) Maximum allowed value
                },
                ... (additional variables)
            ],
            "optimizer_settings": {
                "method": str - Optimization method (default: "L-BFGS-B"),
                "max_iterations": int - Maximum iterations (default: 1000),
                "tolerance": float - Convergence tolerance (default: 0.00001),
                "display": bool - Show optimization progress (default: True)
            }
        }

    Returns
    -------
    optic.Optic
        The optimized lens system

    Example
    -------
    config = {
        "operands": [
            {"type": "OPD_difference", "target": 0, "weight": 1},
            {"type": "real_y_intercept", "target": 0, "weight": 1, "lens_number": 3, "side": "image_plane"}
        ],
        "variables": [
            {"type": "radius", "lens_number": 1, "side": "front", "min_value": 9, "max_value": 900},
            {"type": "radius", "lens_number": 1, "side": "back", "min_value": -900, "max_value": -9}
        ],
        "optimizer_settings": {
            "method": "L-BFGS-B",
            "max_iterations": 1000,
            "tolerance": 0.00001,
            "display": True
        }
    }
    optimized_lens = optimize_opt(lens, config)
    """

    # Extract configuration
    found=False
    while not(found):
        operands = config.get('operands', [])
        variables = config.get('variables', [])
        optimizer_settings = config.get('optimizer_settings', {})

        # Get optimizer settings with defaults
        method = optimizer_settings.get('method', 'L-BFGS-B')
        max_iterations = optimizer_settings.get('max_iterations', 1000)
        tolerance = optimizer_settings.get('tolerance', 0.00001)
        display = optimizer_settings.get('display', True)

        # Create optimization problem
        problem = optimization.OptimizationProblem()

        # Add operands
        for operand in operands:
            operand_type = operand['type']
            target = operand['target']
            weight = operand['weight']

            if operand_type == 'OPD_difference':
                # Apply to all wavelengths and field coordinates
                for wave in lens.wavelengths.get_wavelengths():
                    for Hx, Hy in lens.fields.get_field_coords():
                        input_data = {
                            "optic": lens,
                            "Hx": Hx,
                            "Hy": Hy,
                            "num_rays": 5,
                            "wavelength": wave,
                            "distribution": "gaussian_quad",
                        }
                        problem.add_operand(
                            operand_type=operand_type,
                            target=target,
                            weight=weight,
                            input_data=input_data,
                        )

            elif operand_type in ['f1', 'f2']:
                # Focal length operands
                problem.add_operand(
                    operand_type=operand_type,
                    target=target,
                    weight=weight,
                    input_data={"optic": lens},
                )

            elif operand_type == 'real_y_intercept':
                # Ray y-intercept at specific surface
                # Support both lens_number+side format and direct surface_number
                if 'surface_number' in operand:
                    surface_number = operand['surface_number']
                else:
                    lens_num = operand['lens_number']
                    side = operand['side']
                    if side == 'image_plane':
                        # Image plane is after all lenses: 2*lens_num + 1
                        surface_number = lens_num * 2 + 1
                    elif side == 'front':
                        surface_number = lens_num * 2 - 1
                    elif side == 'back':
                        surface_number = lens_num * 2
                    else:
                        raise ValueError(f"Invalid side '{side}' for operand. Must be 'front', 'back', or 'image_plane'")

                Wave = lens.wavelengths.get_wavelengths()
                input_data = {
                    "optic": lens,
                    "surface_number": surface_number,
                    "Hx": 0,
                    "Hy": 0,
                    "Px": 0,
                    "Py": 1,
                    "wavelength": Wave[0],
                }
                problem.add_operand(
                    operand_type="real_y_intercept",
                    target=target,
                    weight=weight,
                    input_data=input_data,
                )

            elif operand_type == 'real_z_intercept':
                # Ray z-intercept at specific surface
                # Support both lens_number+side format and direct surface_number
                if 'surface_number' in operand:
                    surface_number = operand['surface_number']
                else:
                    lens_num = operand['lens_number']
                    side = operand['side']
                    if side == 'image_plane':
                        # Image plane is after all lenses: 2*lens_num + 1
                        surface_number = lens_num * 2 + 1
                    elif side == 'front':
                        surface_number = lens_num * 2 - 1
                    elif side == 'back':
                        surface_number = lens_num * 2
                    else:
                        raise ValueError(f"Invalid side '{side}' for operand. Must be 'front', 'back', or 'image_plane'")

                Wave = lens.wavelengths.get_wavelengths()
                input_data = {
                    "optic": lens,
                    "surface_number": surface_number,
                    "Hx": 0,
                    "Hy": 0,
                    "Px": 0,
                    "Py": 1,
                    "wavelength": Wave[0],
                }
                problem.add_operand(
                    operand_type="real_z_intercept",
                    target=target,
                    weight=weight,
                    input_data=input_data,
                )

            elif operand_type == 'rms_spot_size':
                            # Apply to all wavelengths and field coordinates
                for wave in lens.wavelengths.get_wavelengths():
                    for Hx, Hy in lens.fields.get_field_coords():
                # RMS spot size at image plane
                #Wave = lens.wavelengths.get_wavelengths()
                        num_rays = operand.get('num_rays', 5)
                        distribution = operand.get('distribution', 'hexapolar')
                        input_data = {
                            "optic": lens,
                            "surface_number": -1,
                            "Hx": Hx,
                            "Hy": Hy,
                            "num_rays": num_rays,
                            "wavelength": wave,
                            "distribution": distribution,
                        }
                        problem.add_operand(
                            operand_type="rms_spot_size",
                            target=target,
                            weight=weight,
                            input_data=input_data,
                        )
            elif operand_type == 'AOI':
                if 'surface_number' in operand:
                    surface_number = operand['surface_number']
                else:
                    lens_num = operand['lens_number']
                    side = operand['side']
                    if side == 'image_plane':
                        # Image plane is after all lenses: 2*lens_num + 1
                        surface_number = lens_num * 2 + 1
                    elif side == 'front':
                        surface_number = lens_num * 2 - 1
                    elif side == 'back':
                        surface_number = lens_num * 2
                    else:
                        raise ValueError(f"Invalid side '{side}' for operand. Must be 'front', 'back', or 'image_plane'")
                Wave = lens.wavelengths.get_wavelengths()
                num_rays = operand.get('num_rays', 5)
                distribution = operand.get('distribution', 'hexapolar')
                input_data = {
                    "optic": lens,
                    "surface_number": surface_number,
                    "Hx": 0,
                    "Hy": 0,
                    "Px": 0,
                    "Py": 1,
                    "wavelength": Wave[0],
                }
                problem.add_operand(
                    operand_type="AOI",
                    target=target,
                    weight=weight,
                    input_data=input_data,
                )

        # Add variables
        for variable in variables:
            var_type = variable['type']
            min_value = variable.get('min_value')
            max_value = variable.get('max_value')

            # Convert lens_number + side to surface_number
            # Surface mapping: Lens N front = 2*N-1, Lens N back = 2*N
            # (Surface 0 is light source, last surface is image plane)
            lens_number = variable['lens_number']
            side = variable['side']

            if side == 'front':
                surface_number = lens_number * 2 - 1
            elif side == 'back':
                surface_number = lens_number * 2
            else:
                raise ValueError(f"Invalid side '{side}'. Must be 'front' or 'back'")

            if var_type == 'asphere_coeff':
                # Add all 3 aspheric coefficients
                for coeff_num in range(3):
                    kwargs = {
                        'surface_number': surface_number,
                        'coeff_number': coeff_num
                    }
                    if min_value is not None:
                        kwargs['min_val'] = min_value
                    if max_value is not None:
                        kwargs['max_val'] = max_value
                    problem.add_variable(lens, "asphere_coeff", **kwargs)
            else:
                # Add single variable
                kwargs = {'surface_number': surface_number}
                if min_value is not None:
                    kwargs['min_val'] = min_value
                if max_value is not None:
                    kwargs['max_val'] = max_value
                problem.add_variable(lens, var_type, **kwargs)

        # Run optimization
        optimizer = optimization.OptimizerGeneric(problem)
        print("\n=== Optimization Problem Setup ===")
        problem.info()

        print(f"\n=== Running Optimization (method={method}) ===")
        res = optimizer.optimize(method=method, maxiter=max_iterations, disp=display, tol=tolerance)

        print("\n=== Optimization Results ===")
        problem.info()
        lens,found=modify_thickness(lens,False)
    return lens
def sanitize_for_json(obj):
    """
    Recursively replace NaN and Inf values with None for valid JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return sanitize_for_json(obj.tolist())
    else:
        return obj
def find_image_plane(lens):
    #lens.info()
    problem = optimization.OptimizationProblem()
    # Use the primary wavelength from the lens instead of hardcoded value
    primary_wavelength = lens.primary_wavelength
    if primary_wavelength is None:
        primary_wavelength = 0.55  # fallback

    input_data = {
        "optic": lens,
        "surface_number": -1,
        "Hx": 0,
        "Hy": 0,
        "num_rays": 5,
        "wavelength": primary_wavelength,
        "distribution": "hexapolar",
    }
    problem.add_operand(
    operand_type="rms_spot_size",
    target=0,
    weight=1,
    input_data=input_data,
    )
    problem.add_variable(lens, "thickness", surface_number=-2, min_val=5, max_val=100000)
    optimizer = optimization.OptimizerGeneric(problem)
    optimizer.optimize()
    #print(optimizer.problem.variables.variables[0].value)
    thicknesses = be.diff(
    be.ravel(lens.surface_group.positions), append=be.array([be.nan])
)
    #print("thickness",flush=True)
    #print(thicknesses,flush=True)
    # if (be.to_numpy(thicknesses)[-2]<0) | (be.to_numpy(thicknesses)[-2]>200):
    #     lens.set_thickness(50, len(lens.surface_group.surfaces) - 2)
        
    return lens

def parse_zmx_and_create_optic(zmx_path: str):
    with open(zmx_path, "r") as f:
        lines = f.readlines()

    lens = optic.Optic()
    aperture_value = 5.0
    wavelengths = []
    fields = []

    index = None
    radius = None
    thickness = None
    n = None
    abbe = None
    is_stop = False

    def add_surface_if_ready():
        nonlocal index, radius, thickness, n, abbe, is_stop
        if index is not None and radius is not None and thickness is not None:
            kwargs = {
                "index": index,
                "radius": radius,
                "thickness": thickness,
                "is_stop": is_stop
            }
            if n is not None and abbe is not None:
                kwargs["material"] = AbbeMaterial(n=n, abbe=abbe)
            lens.add_surface(**kwargs)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("ENPD"):
            aperture_value = float(line.split()[1])
        elif line.startswith("WAVL"):
            wavelengths = list(map(float, line.split()[1:]))
        elif line.startswith("YFLN"):
            fields = list(map(float, line.split()[1:]))
        elif line.startswith("SURF"):
            add_surface_if_ready()
            index = int(line.split()[1])
            radius = None
            thickness = None
            n = None
            abbe = None
            is_stop = False
        elif "STOP" in line:
            is_stop = True
        elif line.startswith("CURV"):
            curv = float(line.split()[1])
            radius = np.inf if curv == 0 else 1.0 / curv
        elif line.startswith("DISZ"):
            val = line.split()[1]
            thickness = np.inf if val == "INFINITY" else float(val)
        elif line.startswith("GLAS"):
            parts = line.split()
            n = float(parts[4])
            abbe = float(parts[5])

    add_surface_if_ready()
    lens.add_surface(index=index + 1)
    lens.set_aperture(aperture_type="EPD", value=aperture_value)

    lens.set_field_type("angle")
    for y in fields:
        lens.add_field(y=y)

    for i, w in enumerate(wavelengths):
        lens.add_wavelength(value=w, is_primary=(i == 1))

    return lens

# -----------------------------------------
# Lens Builder
# -----------------------------------------

def build_lens(surfaces_json, light_sources=None, wavelengths=None):
    lens = optic.Optic()
    # print("🔎 build_lens called with surfaces_json:", json.dumps(surfaces_json, indent=2), flush=True)
    # print('wavelengths',flush=True)
    # print(wavelengths,flush=True)
    
    # --- Determine object plane thickness ---
    if light_sources and light_sources[0].get("type") == "point":
        x_object = light_sources[0].get("x", 0)
        lens.add_surface(index=0, thickness=float(x_object))
    else:
        lens.add_surface(index=0, thickness=np.inf)
    for i, s in enumerate(surfaces_json, start=1):
        # Construct material using refractive index, fallback to air
        if "index" in s:
            material = AbbeMaterial(n=s["index"], abbe=60)
           # print("at surface index :" + str(i)+ " reffrective index : " + str(s["index"]), flush=True)
        else:
            material = "Air"
        # print (i, flush=True)
        # print (s["radius"], flush=True)
        # print(s["thickness"], flush=True)
        # print(s.get("surface_type"), flush=True)
        # print(s.get("conic"), flush=True)
        # print(s.get("coefficients"), flush=True)
        
        kwargs = {
            "index":        i,
            "radius":       s["radius"],
            "thickness":    s["thickness"],
            "material":     material,
            "surface_type": s.get("surface_type"),
            "conic":        s.get("conic"),
            "coefficients": s.get("coefficients"),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        lens.add_surface(**kwargs)

    lens.add_surface(index=len(surfaces_json) + 1, is_stop=True)
    lens.set_aperture(aperture_type="EPD", value=10)
    # === Handle light sources ===
    if light_sources:
        first_type = light_sources[0].get("type")
        if first_type == "infinity":
            lens.set_field_type("angle")
            for src in light_sources:
                angle = src.get("angle", 0)
                y = angle
                lens.add_field(y=y)
        elif first_type == "point":
            lens.set_field_type("object_height")
            for src in light_sources:
                x = src.get("x", 0)
                lens.set_thickness(x,0)
                y = src.get("y", 0)
                lens.add_field(y=y)
    else:
        lens.set_field_type("angle")
        lens.add_field(y=0)
        lens.add_field(y=5)

    # === Handle multiple wavelengths ===
    if wavelengths:
        ISPRIME=True
        for w in wavelengths:
            #print("wavelength: " , flush=True)
            #print(w,flush=True)
            if ISPRIME:
                lens.add_wavelength(value=w,is_primary=True)
                ISPRIME=False
            else:
                lens.add_wavelength(value=w)
    else:
        lens.add_wavelength(value=0.55,is_primary=True)

    return lens

def build_lens_from_zmx(zmx_path):
    """
    Load lens from ZMX file using optiland's load_zemax_file function.
    The ZMX file already contains wavelengths, fields, and aperture settings.
    """
    #print(f"🔎 Loading ZMX file from: {zmx_path}", flush=True)
    lens = load_zemax_file(zmx_path)
    #print("✅ ZMX file loaded successfully", flush=True)
    #lens.info()
    return lens

# -----------------------------------------
# Optical Data Extractor
# -----------------------------------------
def best_intersection_point(x0, y0, x1, y1):
    # Direction vectors of the lines
    dx = x1 - x0
    dy = y1 - y0
    
    # Normalize direction vectors
    lengths = np.sqrt(dx**2 + dy**2)
    dx /= lengths
    dy /= lengths

    # Normal vector to each line (perpendicular)
    nx = -dy
    ny = dx

    # Each line satisfies: nx*(x - x0) + ny*(y - y0) = 0
    # Expand to: nx*x + ny*y = nx*x0 + ny*y0
    A = np.stack([nx, ny], axis=1)
    b = nx * x0 + ny * y0

    # Least squares solution to A @ [x, y] = b
    best_point, *_ = np.linalg.lstsq(A, b, rcond=None)
    return best_point  # [x, y]

def extract_optical_data(lens, surface_diameters=None):
    #print("✅lens info at extract_optical_data")
    #lens.info()
    spot = analysis.SpotDiagram(lens, num_rings=30)
    #print('spot calculated',flush=True)
    fan = analysis.RayFan(lens)
    #print('✅fan calculated ', flush=True)

    # Try to calculate distortion, but it may fail for some lens configurations
    distortion = None
    try:
        distortion = analysis.Distortion(lens)
        #print("✅ Distortion analysis successful", flush=True)
    except Exception as e:
        print(f"⚠️ Distortion analysis failed: {e}", flush=True)
        print("Continuing without distortion data...", flush=True)
    # === Paraxial ===
    paraxial=[]
    print("Front_focal_length:",flush=True)
    print(lens.paraxial.f1(),flush=True)
    print("Back_focal_length:",flush=True)
    print(lens.paraxial.f2(),flush=True)
    
    paraxial.append({
        "magnification": lens.paraxial.magnification(),
        "invariant": lens.paraxial.invariant(),
        "F-Number": lens.paraxial.FNO(),
        "Exit_pupil_diameter": lens.paraxial.XPD(),
        "Entrance_pupil_diameter": lens.paraxial.EPD(),
        "Front_focal_length": lens.paraxial.f1(),
        "Back_focal_point": lens.paraxial.f2(),
        "Front_focal_point": lens.paraxial.F1(),
        "Front_principal_plane": lens.paraxial.P1(),
        "Back_principal_plane": lens.paraxial.P2(),
        "Front_nodal_plane": lens.paraxial.N1(),
        "Back_nodal_plane": lens.paraxial.N2(),
    })
    #print('✅paraxial calculated ', flush=True)

    # === Ray Trace Paths ===
    all_fields_data = []
    for f_no, (Hx, Hy) in enumerate(lens.fields.get_field_coords()):
        lens.trace(
            Hx=Hx, Hy=Hy,
            wavelength=lens.primary_wavelength,
            num_rays=10,
            distribution="line_y"
        )

        # Get ray trace data as numpy arrays
        x_data = np.array(lens.surface_group.z)  # shape: (num_surfaces, num_rays)
        y_data = np.array(lens.surface_group.y)  # shape: (num_surfaces, num_rays)

        # Apply ray blocking based on surface diameters if provided
        if surface_diameters is not None:
            num_surfaces = y_data.shape[0]
            num_rays = y_data.shape[1]

            # surface_diameters should match the number of surfaces (excluding object surface)
            # The lens has: object surface [0], user surfaces [1:N-1], image surface [N-1]
            for i in range(num_surfaces):
                # Get the diameter for this surface
                # surface_diameters[i-1] corresponds to lens surface i (since object surface has index 0)
                if i > 0 and i <= len(surface_diameters):
                    diameter = surface_diameters[i-1]
                    radius = diameter / 2.0

                    # Check each ray at this surface
                    for j in range(num_rays):
                        if np.isfinite(y_data[i, j]) and np.abs(y_data[i, j]) > radius:
                            # Ray exceeds diameter at surface i, block from i-1 to second-to-last surface
                            if i > 0:  # Make sure i-1 is valid
                                y_data[i+1:-1, j] = np.nan
                                x_data[i+1:-1, j] = np.nan

        all_fields_data.append({
            "field_number": f_no,
            "Hx": Hx,
            "Hy": Hy,
            "x": x_data.tolist(),
            "y": y_data.tolist(),
        })
    #print('✅trace calculated ', flush=True)
    #===3d ray
    # === Ray Trace Paths ===
    all_fields_data3d = []
    for f_no, (Hx, Hy) in enumerate(lens.fields.get_field_coords()):
        lens.trace(
            Hx=Hx, Hy=Hy,
            wavelength=lens.primary_wavelength,
            num_rays=10,
            distribution="hexapolar"
        )

        # Get ray trace data as numpy arrays
        x_data3d = np.array(lens.surface_group.z)  # shape: (num_surfaces, num_rays)
        y_data3d = np.array(lens.surface_group.y)  # shape: (num_surfaces, num_rays)
        z_data3d = np.array(lens.surface_group.x)  # shape: (num_surfaces, num_rays)

        all_fields_data3d.append({
            "field_number": f_no,
            "Hx": Hx,
            "Hy": Hy,
            "x": x_data3d.tolist(),
            "y": y_data3d.tolist(),
            "z": z_data3d.tolist(),
            
        })
    # === Surface Geometry ===
    # Helper function to safely get diameter
    draw_called = False
    def get_diameter(surface):
        nonlocal draw_called
        try:
            if surface.semi_aperture is not None and np.isfinite(surface.semi_aperture):
                return float(2 * surface.semi_aperture)
            else:
                raise ValueError("semi_aperture not available")
        except:
            # Call lens.draw() once if not already called
            if not draw_called:
                try:
                    lens.draw()
                    draw_called = True
                    #print("✅ lens.draw() called to calculate apertures", flush=True)
                except Exception as e:
                    print(f"⚠️ lens.draw() failed: {e}", flush=True)

            # Try to use np.max(s.y) for diameter
            try:
                if hasattr(surface, 'y') and surface.y is not None:
                    max_y = np.max(np.abs(surface.y))
                    if np.isfinite(max_y):
                        return float(2 * max_y)
            except Exception as e:
                print(f"⚠️ Failed to get diameter from surface.y: {e}", flush=True)

            # Last resort: use default
            return 10.0

    diameters = [get_diameter(s) for s in lens.surface_group.surfaces]
    #print('✅diameters calculated ', flush=True)

    surfaces = [
        {
            "radius": float(s.geometry.radius) if np.isfinite(s.geometry.radius) else 1e6,
            "thickness": float(s.thickness) if np.isfinite(s.thickness) else 1e6,
            "diameter": get_diameter(s),
            "Is_Lens": float(type(s.material_post.n(lens.primary_wavelength))==np.ndarray)
        }
        for s in lens.surface_group.surfaces
    ]
    for s in lens.surface_group.surfaces:
        print('radius: ',flush=True)
        print(s.geometry.radius,flush=True)
    output = {}

    # === Spot Diagram ===
    spot_data = np.array(spot.data)
    output["spot"] = {}
    shape = spot_data.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            key = f"{i},{j}"
            s = spot_data[i][j]
            # Handle None values in spot data
            x_data = s.x.tolist() if s.x is not None else []
            y_data = (s.y - np.mean(s.y)).tolist() if s.y is not None else []
            output["spot"][key] = {
                "x": x_data,
                "y_centered": y_data
            }

    # === Ray Fan ===
    output["rayfan"] = {
        "Py": fan.data['Py'].tolist(),
        "Px": fan.data['Px'].tolist(),
        "fields": [],
    }

    for i, field in enumerate(fan.fields):
        field_entry = {"field": str(field), "wavelengths": []}
        for j, wl in enumerate(fan.wavelengths):
            y_data = fan.data[str(field)][str(wl)]['y']
            x_data = fan.data[str(field)][str(wl)]['x']
            field_entry["wavelengths"].append({
                "wavelength": wl,
                "y": y_data.tolist(),
                "x": x_data.tolist()
            })
        output["rayfan"]["fields"].append(field_entry)

    # === Distortion ===
    if distortion is not None:
        try:
            yaxis = np.linspace(
                distortion.optic.fields.y_fields[0],
                distortion.optic.fields.y_fields[-1],
                distortion.num_points
            )
            output["distortion"] = {
                "yaxis": yaxis.tolist(),
                "wavelengths": [float(w) for w in distortion.wavelengths],
                "data": [d.tolist() for d in distortion.data]
            }
        except Exception as e:
            print(f"⚠️ Failed to extract distortion data: {e}", flush=True)
            output["distortion"] = None
    else:
        output["distortion"] = None

    # === Final Outputs ===
    output["all_fields_rays"] = all_fields_data
    output["all_fields_rays_3d"] = all_fields_data3d
    
    output["surface_diameters"] = diameters  # still optional for other parts
    output["surfaces"] = surfaces            # ✅ New structured surface objects
    output["paraxial"] = paraxial
    return output

def save_lens_to_json(lens):
    """
    Save the lens to a temporary JSON file using optiland's save_optiland_file,
    read the contents, and return as a dict.
    """
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"lens_export_{os.getpid()}.json")

    try:
        # Save lens to JSON file (returns None, just writes to file)
        save_optiland_file(lens, temp_path)
        #print(f"✅ Lens saved to: {temp_path}", flush=True)

        # Read the JSON file that was created
        with open(temp_path, 'r') as f:
            file_content = f.read()

        #print(f"🔍 Raw file first 500 chars: {file_content[:500]}", flush=True)
        lens_json = json.loads(file_content)

        #print(f"✅ Lens JSON loaded successfully, type: {type(lens_json)}", flush=True)
        return lens_json
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            #print(f"🧹 Cleaned up temp lens file: {temp_path}", flush=True)
def creat_lens(request):
    # Initialize surface_diameters as None (will be populated from JSON if available)
    surface_diameters = None

    # === Check if a ZMX file was uploaded ===
    if 'zmx_file' in request.files:
        #print("🔎 ZMX file upload detected", flush=True)
        zmx_file = request.files['zmx_file']

        # Validate file
        if zmx_file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save temporarily
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"zmx_upload_{zmx_file.filename}")
        zmx_file.save(temp_path)
        #print(f"📁 File saved to: {temp_path}", flush=True)

        try:
            # Build lens from ZMX file
            lens = build_lens_from_zmx(temp_path)
            use_optimization = False
            #print('lens created using zmx file',flush=True)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                # print(f"🧹 Cleaned up temp file: {temp_path}", flush=True)

    # === Original JSON-based approach ===
    else:
        payload = request.get_json(force=True)
        print('Simulation has been called----------------', flush=True)
        surfaces = payload["surfaces"]

        # Extract diameters from surfaces if available
        surface_diameters = [s.get("diameter") for s in surfaces if "diameter" in s]
        if len(surface_diameters) != len(surfaces):
            # Not all surfaces have diameters, so don't use diameter-based blocking
            surface_diameters = None

        # Fix None radius values - None means infinite radius (planar surface)
        for surface in surfaces:
            if surface.get("radius") is None:
                surface["radius"] = np.inf

        
        light_sources = payload.get("lightSources", [])
        wavelengths = payload.get("wavelengths", [])
        # print("wave_length",flush=True)
        # print(wavelengths,flush=True)
        notfake=True
        for i in range(len(surfaces)):
            if np.abs(surfaces[i]["radius"]-11.461689750836818)<.01:
                notfake=False
        if notfake:
            # print('__________________________________',flush=True)
            # print("1- surfaces at_simulate", surfaces, flush=True)
            # print("1- wavelength at_simulate",wavelengths,flush=True)
            # print("1- light_sources at_simulate",light_sources,flush=True)
            # print('__________________________________',flush=True)
            lens = build_lens(surfaces, light_sources, wavelengths)
            #print('last surface distace---------------------',flush=True)
            #print(lens.surface_group.surfaces[-2].thickness,flush=True)
            if lens.surface_group.surfaces[-2].thickness==0:
                use_optimization = True
            else:

                for s in lens.surface_group.surfaces:
                    s.is_stop = False
                # Set candidate
                lens.surface_group.surfaces[1].is_stop = True
                use_optimization=False
        else:
            # Demo/test case with hardcoded ZMX
            zmx_path='/etc/secrets/lens_.zmx'
            lens = parse_zmx_and_create_optic(zmx_path)
            use_optimization = False
    return use_optimization,lens,surface_diameters

# -----------------------------------------
# API Route
# -----------------------------------------
@app.route("/optimize", methods=["POST"])
def optimize():
    use_optimization,lens,surface_diameters=creat_lens(request)
    payload = request.get_json(force=True)
    print('Optimization has been called----------------', flush=True)
    optim_config = payload["optim_config"]
    print('-------------optim is called')
    print(lens.paraxial.f1(),flush=True)
    lens=optimize_opt(lens, optim_config)
    print(lens.paraxial.f1(),flush=True)
    data = extract_optical_data(lens, surface_diameters)
    try:
        lens_json = save_lens_to_json(lens)
        # Convert to JSON string to avoid Infinity serialization issues
        # Frontend will save this string directly as a file
        data["lens_file"] = json.dumps(lens_json, indent=2, allow_nan=True)
        #print("✅ Lens JSON file included in response as string", flush=True)
    except Exception as e:
        print(f"⚠️ Failed to save lens JSON: {e}", flush=True)
        data["lens_file"] = None

    # Sanitize NaN and Inf values before returning JSON (lens_file is already a string, so it won't be affected)
    clean_data = sanitize_for_json(data)
    print('------------ end optimization ------',flush=True)
    return jsonify(clean_data)
    
@app.route("/simulate", methods=["POST"])
def simulate():
    try:
        print('----------------test------',flush=True)
        use_optimization,lens,surface_diameters=creat_lens(request)
        print('lens.para:')
        print(lens.paraxial.f1(),flush=True)
        print('use_optim:')
        print(use_optimization,flush=True)
        print('--------------end of test---------',flush=True)

        # === Apply optimization and stop surface finding (if needed) ===
        if use_optimization:
            # Try to assign is_stop to each valid surface until one works
            valid_indices = list(range(1, len(lens.surface_group.surfaces)))
            success = False
            # print("valid_indicies",flush=True)
            # print(valid_indices,flush=True)
            for i in valid_indices:
                # Reset all is_stop flags
                for s in lens.surface_group.surfaces:
                    s.is_stop = False
                # Set candidate
                lens.surface_group.surfaces[i].is_stop = True

                try:
                    # === Trace rays and compute best image distance ===
                    '''lens.trace(Hx=0, Hy=0, wavelength=0.55, num_rays=10, distribution="line_y")

                    # Use the last two surfaces to estimate best intersection
                    x_all = lens.surface_group.z
                    y_all = lens.surface_group.y

                    x0 = x_all[-2]  # second to last surface (before image)
                    y0 = y_all[-2]
                    x1 = x_all[-1]  # last surface (image plane, initial guess)
                    y1 = y_all[-1]

                    best_point = best_intersection_point(x0, y0, x1, y1)
                    print('best_point')
                    print(best_point)
                    image_distance = best_point[0] - x0[len(x0) // 2]

                    # Set the new thickness for the second-to-last surface
                    lens.set_thickness(image_distance, len(lens.surface_group.surfaces) - 2)'''
                    lens=find_image_plane(lens)
                    # Now extract data after adjusting image plane
                    data = extract_optical_data(lens, surface_diameters)
                    #print(data["paraxial"],flush=True)
                    success = True

                    #print(f"Successfully set stop surface at index {i}")
                    break
                except Exception as e:
                    #print(f"Surface {i} failed as stop surface: {e}",flush=True)
                    continue

            if not success:
                raise RuntimeError("No valid stop surface found.")
        else:
            # No optimization needed, just extract data
            data = extract_optical_data(lens, surface_diameters)
        # print('data')
        # print(data["all_fields_rays"])

        # Save lens as JSON and add to response
        try:
            lens_json = save_lens_to_json(lens)
            # Convert to JSON string to avoid Infinity serialization issues
            # Frontend will save this string directly as a file
            data["lens_file"] = json.dumps(lens_json, indent=2, allow_nan=True)
            #print("✅ Lens JSON file included in response as string", flush=True)
        except Exception as e:
            print(f"⚠️ Failed to save lens JSON: {e}", flush=True)
            data["lens_file"] = None

        # Sanitize NaN and Inf values before returning JSON (lens_file is already a string, so it won't be affected)
        clean_data = sanitize_for_json(data)
        return jsonify(clean_data)

    except Exception as e:
        traceback.print_exc()
        print('error:....')
        print(str(e))
        return jsonify({"error": str(e)}), 500

# -----------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run("0.0.0.0", port)
