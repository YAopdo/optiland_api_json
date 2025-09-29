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
                y = np.tan(np.radians(angle))
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

def extract_optical_data(lens):
    print("lens info at extract_optical_data")
    lens.info()
    spot = analysis.SpotDiagram(lens, num_rings=30)
    fan = analysis.RayFan(lens)
    distortion = analysis.Distortion(lens)
    # === Paraxial ===
    paraxial=[]
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
    # === Ray Trace Paths ===
    all_fields_data = []
    for f_no, (Hx, Hy) in enumerate(lens.fields.get_field_coords()):
        lens.trace(
            Hx=Hx, Hy=Hy,
            wavelength=0.55,
            num_rays=10,
            distribution="line_y"
        )
        all_fields_data.append({
            "field_number": f_no,
            "Hx": Hx,
            "Hy": Hy,
            "x": lens.surface_group.z.tolist(),
            "y": lens.surface_group.y.tolist(),
        })

    # === Surface Geometry ===
    diameters = [2 * s.semi_aperture for s in lens.surface_group.surfaces]

    surfaces = [
        {
            "radius": float(s.geometry.radius) if np.isfinite(s.geometry.radius) else 1e6,
            "thickness": float(s.thickness) if np.isfinite(s.thickness) else 1e6,
            "diameter": float(2 * s.semi_aperture),
            "Is_Lens": float(type(s.material_post.abbe)==np.ndarray)
        }
        for s in lens.surface_group.surfaces
    ]

    output = {}

    # === Spot Diagram ===
    spot_data = np.array(spot.data)
    output["spot"] = {}
    shape = spot_data.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            key = f"{i},{j}"
            s = spot_data[i][j]
            output["spot"][key] = {
                "x": s.x.tolist(),
                "y_centered": (s.y - np.mean(s.y)).tolist()
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

    # === Final Outputs ===
    output["all_fields_rays"] = all_fields_data
    output["surface_diameters"] = diameters  # still optional for other parts
    output["surfaces"] = surfaces            # ✅ New structured surface objects
    output["paraxial"] = paraxial
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
        return jsonify(data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run("0.0.0.0", port)
