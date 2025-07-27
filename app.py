# app.py (Opdo simulation backend, JSON version)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import traceback, os
import json
import matplotlib

from optiland import optic, analysis

app = Flask(__name__)
CORS(app)

# -----------------------------------------
# Lens Builder
# -----------------------------------------

def build_lens(surfaces_json):
    lens = optic.Optic()
    lens.add_surface(index=0, thickness=np.inf)  # Object plane

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
    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)
    lens.add_field(y=5)
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
    lens.info()
    spot = analysis.SpotDiagram(lens, num_rings=30)
    fan = analysis.RayFan(lens)
    distortion = analysis.Distortion(lens)

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
            "diameter": float(2 * s.semi_aperture)
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

    return output



# -----------------------------------------
# API Route
# -----------------------------------------

@app.route("/simulate", methods=["POST"])
def simulate():
    try:
        payload = request.get_json(force=True)
        surfaces = payload["surfaces"]

        lens = build_lens(surfaces)
        
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
                lens.trace(Hx=0, Hy=0, wavelength=0.55, num_rays=10, distribution="line_y")
                
                # Use the last two surfaces to estimate best intersection
                x_all = lens.surface_group.z
                y_all = lens.surface_group.y
                
                x0 = x_all[-2]  # second to last surface (before image)
                y0 = y_all[-2]
                x1 = x_all[-1]  # last surface (image plane, initial guess)
                y1 = y_all[-1]
                
                best_point = best_intersection_point(x0, y0, x1, y1)
                image_distance = best_point[0] - x0[len(x0) // 2]
                
                # Set the new thickness for the second-to-last surface
                lens.set_thickness(image_distance, len(lens.surface_group.surfaces) - 2)
                
                # Now extract data after adjusting image plane
                data = extract_optical_data(lens)
                success = True

                print(f"Successfully set stop surface at index {i}")
                break
            except Exception as e:
                print(f"Surface {i} failed as stop surface: {e}")
                continue
        
        if not success:
            raise RuntimeError("No valid stop surface found.")
        

        return jsonify(data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run("0.0.0.0", port)
