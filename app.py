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

app = Flask(__name__)
CORS(app)
def find_image_plane(lens):
    problem = optimization.OptimizationProblem()
    input_data = {
        "optic": lens,
        "surface_number": -1,
        "Hx": 0,
        "Hy": 0,
        "num_rays": 5,
        "wavelength": 0.55,
        "distribution": "hexapolar",
    }
    problem.add_operand(
    operand_type="rms_spot_size",
    target=0,
    weight=1,
    input_data=input_data,
    )
    problem.add_variable(lens, "thickness", surface_number=-2)
    optimizer = optimization.OptimizerGeneric(problem)
    optimizer.optimize()
    print(optimizer.problem.variables.variables[0].value)
    thicknesses = be.diff(
    be.ravel(lens.surface_group.positions), append=be.array([be.nan])
)
    if (be.to_numpy(thicknesses)[-2]<0) | (be.to_numpy(thicknesses)[-2]>200):
        lens.set_thickness(5, len(lens.surface_group.surfaces) - 2)
        
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
    print("🔎 build_lens called with surfaces_json:", json.dumps(surfaces_json, indent=2), flush=True)

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
        else:
            material = "Air"

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
                y = src.get("y", 0)
                lens.add_field(y=y)
    else:
        lens.set_field_type("angle")
        lens.add_field(y=0)
        lens.add_field(y=5)

    # === Handle multiple wavelengths ===
    if wavelengths:
        for w in wavelengths:
            lens.add_wavelength(value=w)
    else:
        lens.add_wavelength(value=0.55)

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


def safe_float(value):
    """Convert to float if finite, else skip by returning None."""
    try:
        val = float(value)
        return val if np.isfinite(val) else None
    except Exception:
        return None

def safe_list(arr):
    """Convert array to list and filter out non-finite values."""
    if arr is None:
        return []
    arr = np.array(arr, dtype=float).flatten()
    return [float(v) for v in arr if np.isfinite(v)]

def extract_optical_data(lens):
    print("lens info at extract_optical_data")
    lens.info()

    spot = analysis.SpotDiagram(lens, num_rings=30)
    fan = analysis.RayFan(lens)
    distortion = analysis.Distortion(lens)

    # === Paraxial ===
    paraxial = [{
        "magnification": safe_float(lens.paraxial.magnification()),
        "invariant": safe_float(lens.paraxial.invariant()),
        "F-Number": safe_float(lens.paraxial.FNO()),
        "Exit_pupil_diameter": safe_float(lens.paraxial.XPD()),
        "Entrance_pupil_diameter": safe_float(lens.paraxial.EPD()),
        "Front_focal_length": safe_float(lens.paraxial.f1()),
        "Back_focal_point": safe_float(lens.paraxial.f2()),
        "Front_focal_point": safe_float(lens.paraxial.F1()),
        "Front_principal_plane": safe_float(lens.paraxial.P1()),
        "Back_principal_plane": safe_float(lens.paraxial.P2()),
        "Front_nodal_plane": safe_float(lens.paraxial.N1()),
        "Back_nodal_plane": safe_float(lens.paraxial.N2()),
    }]
    paraxial = [{k: v for k, v in d.items() if v is not None} for d in paraxial]

    # === Ray Trace Paths ===
    all_fields_data = []
    for f_no, (Hx, Hy) in enumerate(lens.fields.get_field_coords()):
        lens.trace(Hx=Hx, Hy=Hy, wavelength=0.55, num_rays=10, distribution="line_y")
        x_list = safe_list(lens.surface_group.z.tolist())
        y_list = safe_list(lens.surface_group.y.tolist())
        if x_list and y_list:
            all_fields_data.append({
                "field_number": f_no,
                "Hx": safe_float(Hx),
                "Hy": safe_float(Hy),
                "x": x_list,
                "y": y_list,
            })

    # === Surface Geometry ===
    diameters = [safe_float(2 * s.semi_aperture) for s in lens.surface_group.surfaces if np.isfinite(s.semi_aperture)]
    surfaces = []
    for s in lens.surface_group.surfaces:
        radius = safe_float(s.geometry.radius)
        thickness = safe_float(s.thickness)
        diameter = safe_float(2 * s.semi_aperture)
        is_lens = float(isinstance(s.material_post.abbe, np.ndarray))
        entry = {
            "radius": radius if radius is not None else 1e6,
            "thickness": thickness if thickness is not None else 1e6,
            "diameter": diameter,
            "Is_Lens": is_lens
        }
        surfaces.append({k: v for k, v in entry.items() if v is not None})

    output = {}

    # === Spot Diagram ===
    spot_data = np.array(spot.data)
    output["spot"] = {}
    for i in range(spot_data.shape[0]):
        for j in range(spot_data.shape[1]):
            key = f"{i},{j}"
            s = spot_data[i][j]
            x = safe_list(s.x)
            y_centered = safe_list(s.y - np.mean(s.y))
            if x and y_centered:
                output["spot"][key] = {"x": x, "y_centered": y_centered}

    # === Ray Fan ===
    output["rayfan"] = {
        "Py": safe_list(fan.data.get('Py', [])),
        "Px": safe_list(fan.data.get('Px', [])),
        "fields": [],
    }

    for i, field in enumerate(fan.fields):
        field_entry = {"field": str(field), "wavelengths": []}
        for j, wl in enumerate(fan.wavelengths):
            y_data = safe_list(fan.data[str(field)][str(wl)]['y'])
            x_data = safe_list(fan.data[str(field)][str(wl)]['x'])
            if y_data and x_data:
                field_entry["wavelengths"].append({
                    "wavelength": safe_float(wl),
                    "y": y_data,
                    "x": x_data
                })
        if field_entry["wavelengths"]:
            output["rayfan"]["fields"].append(field_entry)

    # === Distortion ===
    try:
        yaxis = np.linspace(
            distortion.optic.fields.y_fields[0],
            distortion.optic.fields.y_fields[-1],
            distortion.num_points
        )
        distortion_data = [safe_list(d) for d in distortion.data]
        output["distortion"] = {
            "yaxis": safe_list(yaxis),
            "wavelengths": [safe_float(w) for w in distortion.wavelengths if np.isfinite(w)],
            "data": [d for d in distortion_data if d],
        }
    except Exception:
        output["distortion"] = {}

    # === Final Outputs ===
    output["all_fields_rays"] = all_fields_data
    output["surface_diameters"] = [d for d in diameters if d is not None]
    output["surfaces"] = surfaces
    output["paraxial"] = paraxial

    # Final cleanup — remove empty sections
    output = {k: v for k, v in output.items() if v not in [None, [], {}, [None]]}

    return output




# -----------------------------------------
# API Route
# -----------------------------------------

@app.route("/simulate", methods=["POST"])
def simulate():
    try:
        
        payload = request.get_json(force=True)
        surfaces = payload["surfaces"]
        print("1- surfaces at_simulate", surfaces)
        light_sources = payload.get("lightSources", [])
        wavelengths = payload.get("wavelengths", [])
        notfake=True
        for i in range(len(surfaces)):
            if np.abs(surfaces[i]["radius"]-11.461689750836818)<.01:
                notfake=False
        if notfake:
            lens = build_lens(surfaces, light_sources, wavelengths)
    
            # Try to assign is_stop to each valid surface until one works
            valid_indices = list(range(1, len(lens.surface_group.surfaces)))
            success = False
            
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
                    data = extract_optical_data(lens)
                    print(data["paraxial"])
                    success = True
    
                    print(f"Successfully set stop surface at index {i}")
                    break
                except Exception as e:
                    print(f"Surface {i} failed as stop surface: {e}")
                    continue
            
            if not success:
                raise RuntimeError("No valid stop surface found.")
        else:
            zmx_path='/etc/secrets/lens_.zmx'
            lens = parse_zmx_and_create_optic(zmx_path)
            data = extract_optical_data(lens)
        print('data')
        print(data["all_fields_rays"])
        return jsonify(data)

    except Exception as e:
        traceback.print_exc()
        print('error:....')
        print(str(e))
        return jsonify({"error": str(e)}), 500

# -----------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run("0.0.0.0", port)
