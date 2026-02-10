# app.py (Opdo simulation backend, JSON version)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import traceback, os
import json
from optiland.materials import AbbeMaterial
import optiland.backend as be

import math
from optiland import optic, analysis, optimization
from optiland.fileio import load_zemax_file, save_optiland_file
from optiland.tolerancing.perturbation import DistributionSampler

from optiland.tolerancing import RangeSampler, SensitivityAnalysis, Tolerancing
from optiland.tolerancing.monte_carlo import MonteCarlo

app = Flask(__name__)
CORS(app)


from typing import Any, Dict, List, Optional, Sequence, Tuple


def df_to_columnar_payload(df, max_rows=None):
    if max_rows is not None:
        df = df.head(max_rows)

    clean = df.replace([np.inf, -np.inf], np.nan)

    # Convert to python-native scalars + JSON-friendly None for NaN
    data = {}
    for c in clean.columns:
        col = clean[c].astype(object).where(~clean[c].isna(), None).tolist()
        data[str(c)] = col

    schema = {
        "columns": [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns],
        "n_rows": int(len(df)),
    }

    return {"schema": schema, "data": data}

def fix_lens_geometry_from_summary(
    lens,
    summary: dict,
    safety_margin: float = 1e-3,
):
    """
    Fix lens geometry based on geometry_summary_from_request output.

    Rules:
    - If lens edge <= 0  → increase lens thickness
    - If gap <= 0        → increase air gap thickness
    - If everything OK   → lens is returned unchanged

    Thickness index convention:
      i=1 → first lens edge
      i=2 → first gap
      i=3 → second lens edge
      i=4 → second gap
      ...
    """

    lens_edges = summary.get("Lense edge", [])
    gaps       = summary.get("gaps", [])

    thickness_updates = []

    # --- lens edges (glass thickness) ---
    for k, edge in enumerate(lens_edges):
        if edge <= 0:
            delta = abs(edge) + safety_margin
            thickness_index = 2 * k + 1   # 1, 3, 5, ...
            thickness_updates.append((thickness_index, delta))

    # --- gaps (air thickness) ---
    for k, gap in enumerate(gaps):
        if gap <= 0:
            delta = abs(gap) + safety_margin
            thickness_index = 2 * k + 2   # 2, 4, 6, ...
            thickness_updates.append((thickness_index, delta))

    # --- apply fixes ---
    if not thickness_updates:
        return lens,False  # everything fine

    for i, delta in thickness_updates:
        old_t = lens.surface_group.get_thickness(i)
        new_t = old_t + delta
        lens.set_thickness(new_t, i)

    return lens,True


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)   # converts numpy scalars / 0d arrays to python float too
    except Exception:
        return None

def build_surfaces_for_geometry_summary(lens, surface_diameters: Sequence[Optional[float]],wl_um: float = 0.555) -> List[Dict[str, Any]]:
     
    """
    surface_diameters ALWAYS excludes object & image planes.
    lens surfaces include them, so we iterate i=1..n_total-2 and use diameters[i-1].
    """
    sg = lens.surface_group
    s_objs = sg.surfaces
    n_total = len(s_objs)

    n_used = n_total - 2
    if len(surface_diameters) != n_used:
        raise ValueError(
            f"surface_diameters must have length n_total-2 ({n_used}) because it excludes "
            f"object & image planes, got {len(surface_diameters)} while lens has {n_total} surfaces."
        )

    out: List[Dict[str, Any]] = []

    # exclude object plane (0) and image plane (n_total-1)
    for i in range(1, n_total - 1):
        s = s_objs[i]
        geom = s.geometry

        radius = _to_float(getattr(geom, "radius", None))
        conic  = _to_float(getattr(geom, "k", 0.0)) or 0.0

        coeffs_raw = getattr(geom, "coefficients", None)
        coefficients = [] if coeffs_raw is None else [float(c) for c in coeffs_raw]

        thickness = _to_float(sg.get_thickness(i)) or 0.0
        diameter  = _to_float(surface_diameters[i - 1])  # <-- shift because diameters exclude first/last

        s_dict: Dict[str, Any] = {
            "radius": radius,
            "thickness": thickness,
            "diameter": diameter,
            "conic": conic,
            "coefficients": coefficients,
        }

        # "index" presence => glass after surface i
        try:
            n_post = float(s.material_post.n(wl_um))
            if math.isfinite(n_post) and n_post > 1.0005:
                s_dict["index"] = n_post  # value unused; key presence is what your summary function checks
        except Exception:
            pass

        out.append(s_dict)

    return out
def geometry_summary_from_request(surfaces, n_pts: int = 600) -> Dict[str, List[float]]:
    """
    Minimal output:
      {"Lense edge": [edge_t_lens0, edge_t_lens1, ...],
       "gaps":       [min_gap_0_1,  min_gap_1_2,  ...],
       "lens_passed": [bool0, bool1, ...]}

    lens_passed[i] is True if lens_edges[i] > 0, otherwise False.

    Assumptions aligned with your backend builder:
      - surfaces[i] containing "index" means the medium AFTER surface i is glass
      - A "lens element" is (front=i, back=i+1) with center thickness = surfaces[i]["thickness"]
      - Adjacent-lens gap is between back of lens i and front of lens i+1
      - If diameters exist for all surfaces, overlap radius uses min(diameters); otherwise best-effort.

    Uses even-asphere sag:
      z = r^2/(R*(1 + sqrt(1 - (1+k) r^2/R^2))) + sum(Ci * r^(2i))
      x(r) = vertex_x + z(r)
    """

    

    # Normalize radii (None -> inf)
    for s in surfaces:
        if s.get("radius") is None:
            s["radius"] = float("inf")

    # Diameters (optional)
    #diam_list = [s.get("diameter", None) for s in surfaces]
    diam_list       = [s.get("diameter") for s in surfaces if "diameter" in s]
    have_all_diam = all(d is not None for d in diam_list)

    def _coeff_list(v: Any) -> List[float]:
        if v is None:
            return []
        return [float(x) for x in v]

    def sag_even_asphere(R: float, k: float, coeffs: Sequence[float], r: float) -> float:
        if math.isinf(R):
            z_base = 0.0
        else:
            arg = 1.0 - (1.0 + k) * (r * r) / (R * R)
            if arg < 0:
                return float("nan")
            denom = R * (1.0 + math.sqrt(arg))
            if denom == 0:
                return float("nan")
            z_base = (r * r) / denom

        z_poly = 0.0
        for i, Ci in enumerate(coeffs, start=1):
            z_poly += Ci * (r ** (2 * i))
        return z_base + z_poly

    def surface_x(vertex_x: float, s: Dict[str, Any], r: float) -> float:
        R = float(s.get("radius", float("inf")))
        k = float(s.get("conic", 0.0) if s.get("conic") is not None else 0.0)
        coeffs = _coeff_list(s.get("coefficients"))
        return vertex_x + sag_even_asphere(R, k, coeffs, r)

    # Vertex x positions for each surface (global coordinates)
    vertex_x = [0.0]
    for s in surfaces:
        t = s.get("thickness", 0.0)
        if t is None:
            t = 0.0
        if t == float("inf") or t == np.inf:
            vertex_x.append(vertex_x[-1] + float("inf"))
        else:
            vertex_x.append(vertex_x[-1] + float(t))

    # Detect lens elements as (front_i, back_i) pairs
    lens_pairs = []
    i = 0
    while i < len(surfaces) - 1:
        if "index" in surfaces[i]:
            lens_pairs.append((i, i + 1))
            i = i + 1
        i += 1

    # --- Edge thickness per lens element ---
    lens_edges: List[float] = []
    for (front_i, back_i) in lens_pairs:
        t_glass = float(surfaces[front_i].get("thickness", 0.0))

        # Choose diameter for edge eval
        if have_all_diam:
            D = float(min(diam_list[front_i], diam_list[back_i]))
        else:
            d1 = surfaces[front_i].get("diameter")
            d2 = surfaces[back_i].get("diameter")
            if d1 is None and d2 is None:
                lens_edges.append(float("nan"))
                continue
            D = float(min([d for d in (d1, d2) if d is not None]))

        r_edge = D / 2.0

        # Local coordinates for edge thickness: front vertex at 0, back vertex at t_glass
        x_front_edge = surface_x(0.0, surfaces[front_i], r_edge)
        x_back_edge  = surface_x(t_glass, surfaces[back_i], r_edge)

        lens_edges.append(float(x_back_edge - x_front_edge))

    # --- Minimum gap between adjacent lenses ---
    gaps: List[float] = []

    def min_gap_between_surfaces(s1_idx: int, s2_idx: int, r_max: float) -> float:
        rr = np.linspace(0.0, r_max, n_pts)
        x1 = np.array([surface_x(vertex_x[s1_idx], surfaces[s1_idx], float(r)) for r in rr])
        x2 = np.array([surface_x(vertex_x[s2_idx], surfaces[s2_idx], float(r)) for r in rr])
        valid = np.isfinite(x1) & np.isfinite(x2)
        if not np.any(valid):
            return float("nan")
        d = x2[valid] - x1[valid]
        return float(np.min(d))

    for p in range(len(lens_pairs) - 1):
        back_i = lens_pairs[p][1]
        front_j = lens_pairs[p + 1][0]

        # shared radius region
        if have_all_diam:
            r_max = 0.5 * float(min(diam_list[back_i], diam_list[front_j]))
        else:
            d_back = surfaces[back_i].get("diameter")
            d_front = surfaces[front_j].get("diameter")
            if d_back is None and d_front is None:
                gaps.append(float("nan"))
                continue
            if d_back is None:
                r_max = 0.5 * float(d_front)
            elif d_front is None:
                r_max = 0.5 * float(d_back)
            else:
                r_max = 0.5 * float(min(d_back, d_front))

        gaps.append(min_gap_between_surfaces(back_i, front_j, r_max))


    # Create lens_passed array: True if lens_edges[i] > 0, False otherwise
    lens_passed = [edge > 0 for edge in lens_edges]

    return {"Lense edge": lens_edges, "gaps": gaps, "lens_passed": lens_passed}

def normalize_asphere_coefficients(optic):
    for surf in optic.surface_group.surfaces:
        geom = getattr(surf, "geometry", None)
        if geom is None or not hasattr(geom, "coefficients"):
            continue

        coeffs = geom.coefficients

        # None -> empty list
        if coeffs is None:
            geom.coefficients = []
            continue

        # If it's already a Python list/tuple, keep as list (ensure floats)
        if isinstance(coeffs, (list, tuple)):
            geom.coefficients = [float(x) for x in coeffs]
            continue

        # If it's a numpy array (or numpy-like), convert to list of floats
        if isinstance(coeffs, np.ndarray):
            if coeffs.size == 0:
                geom.coefficients = []
            else:
                geom.coefficients = [float(x) for x in coeffs.reshape(-1)]
            continue

        # Fallback: try to coerce anything else to numpy
        arr = np.asarray(coeffs).reshape(-1)
        geom.coefficients = [] if arr.size == 0 else [float(x) for x in arr]

def tolerancing(request):
    print('------------tolerancing is running----------',flush=True)
    payload = request.get_json(force=True)
    use_optimization,lens,surface_diameters=creat_lens(request)
    print('-----lens created for tolerancing----',flush=True)
    lens.info()
    optim_config = payload["Tolerancing_config"]
    # Extract configuration

    operands = optim_config.get('operands', [])
    variables = optim_config.get('variables', [])
    optimizer_settings = optim_config.get('optimizer_settings', {})
    print('operands for tolerancing:',flush=True)
    print(operands,flush=True)
    print('variables for tolerancing:',flush=True)
    print(variables,flush=True)

    tolerancing = Tolerancing(lens)
    # Add variables
    for variable in variables:
        var_type = variable['type']
        loc = variable.get('loc')
        scale = variable.get('scale')

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
        print('var added:...',flush=True)
        print('loc,scale,var_type,surface_number: ',loc,scale,scale,var_type,surface_number,flush=True)
        sampler = DistributionSampler("normal", loc=loc, scale=scale)
        tolerancing.add_perturbation(var_type, sampler, surface_number=surface_number)
    for operand in operands:
        operand_type = operand['type']
        target = operand.get('target',0)
        weight = operand.get('weight',1)
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
                    tolerancing.add_operand(
                        operand_type=operand_type,
                        target=target,
                        weight=weight,
                        input_data=input_data,
                    )

        elif operand_type in ['f1', 'f2']:
            # Focal length operands
            tolerancing.add_operand(
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
                "Py": 0,
                "wavelength": Wave[0],
            }
            tolerancing.add_operand(
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
                "Py": 0,
                "wavelength": Wave[0],
            }
            tolerancing.add_operand(
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
                    tolerancing.add_operand(
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
            tolerancing.add_operand(
                operand_type="AOI",
                target=target,
                weight=weight,
                input_data=input_data,
            )
        print('operand',flush=True)
    print('monte_carlo is called',flush=True)
    monte_carlo = MonteCarlo(tolerancing)
    monte_carlo.run(num_iterations=1000)
    print('monte_carlo is done',flush=True)
    res=monte_carlo.get_results()
    return res

def optimize_opt(request):
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
    payload = request.get_json(force=True)
    use_optimization,lens,surface_diameters=creat_lens(request)
    print('-----lens created for optimization----',flush=True)
    lens.info()
    optim_config = payload["optim_config"]
    # Extract configuration

    operands = optim_config.get('operands', [])
    variables = optim_config.get('variables', [])
    optimizer_settings = optim_config.get('optimizer_settings', {})
    print('operands:',flush=True)
    print(operands,flush=True)
    print('variables:',flush=True)
    print(variables,flush=True)
    print('optimizer_settings:',flush=True)
    print(optimizer_settings,flush=True)
    
    # Get optimizer settings with defaults
    method = optimizer_settings.get('method', 'L-BFGS-B')
    max_iterations = optimizer_settings.get('max_iterations', 1000)
    tolerance = optimizer_settings.get('tolerance', 0.00001)
    display = optimizer_settings.get('display', True)
    Modify_thickness=optimizer_settings.get('modify_thickness', True)
    Modify_thickness=False
    print('Modify_thickness',flush=True)
    print(Modify_thickness,flush=True)
    counter=0
    while counter<3:
        counter+=1
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
                    "Py": 0,
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
                    "Py": 0,
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
                        'coeff_number': int(coeff_num)
                    }
                    if min_value is not None:
                        kwargs['min_val'] = min_value
                    if max_value is not None:
                        kwargs['max_val'] = max_value
                    problem.add_variable(lens, var_type, **kwargs)
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
        if display:
            problem.info()

        print(f"\n=== Running Optimization (method={method}) ===",flush=True)
        res = optimizer.optimize(method=method, maxiter=max_iterations, disp=display, tol=tolerance)

        print("\n=== Optimization Results ===")
        if display:
            problem.info()
        if Modify_thickness:
            surfaces=build_surfaces_for_geometry_summary(lens, surface_diameters)
            lens,Modify_thickness=fix_lens_geometry_from_summary(lens,surfaces,.1)
            print(counter,flush=True)
            print('modified',flush=True)
        else:
            print(counter,flush=True)
            print('not_modified',flush=True)
            for i in range(len(lens.surface_group.surfaces)-2):
                print(lens.surface_group.surfaces[i+1].geometry.coefficients,flush=True)
            return lens,surface_diameters
    for i in range(len(lens.surface_group.surfaces)-2):
        print(lens.surface_group.surfaces[i+1].geometry.coefficients,flush=True)
    return lens,surface_diameters
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
    thicknesses = be.diff(
    be.ravel(lens.surface_group.positions), append=be.array([be.nan])
)

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

def build_lens(surfaces_json, light_sources=None, wavelengths=None,Apr=10):

    lens = optic.Optic()
    
    # --- Determine object plane thickness ---
    if light_sources and light_sources[0].get("type") == "point":
        x_object = light_sources[0].get("x", 0)
        lens.add_surface(index=0, thickness=float(x_object))
    else:
        lens.add_surface(index=0, thickness=np.inf)
    for i, s in enumerate(surfaces_json, start=1):
        # Construct material using refractive index, fallback to air
        print('surface .... '+str(i)+'  :  ',flush=True)
        print(s,flush=True)
        if "index" in s:
            material = AbbeMaterial(n=s["index"], abbe=60)
        else:
            material = "Air"

        if i==1:
            kwargs = {
                "index":        i,
                "radius":       s["radius"],
                "thickness":    s["thickness"],
                "material":     material,
                "surface_type": s.get("surface_type", "even_asphere"),
                "conic":        s.get("conic"),
                "coefficients": s.get("coefficients"),
                "is_stop": True,
            }
        else:
            kwargs = {
                "index":        i,
                "radius":       s["radius"],
                "thickness":    s["thickness"],
                "material":     material,
                "surface_type": s.get("surface_type", "even_asphere"),
                "conic":        s.get("conic"),
                "coefficients": s.get("coefficients"),
            }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        lens.add_surface(**kwargs)

    lens.add_surface(index=len(surfaces_json) + 1)
    lens.set_aperture(aperture_type="EPD", value=Apr)
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
            if ISPRIME:
                lens.add_wavelength(value=w,is_primary=True)
                ISPRIME=False
            else:
                lens.add_wavelength(value=w)
    else:
        lens.add_wavelength(value=0.55,is_primary=True)
    draw_called_list = [False]  # Track if draw() has been called

    diameters = [get_diameter(s, lens, draw_called_list) for s in lens.surface_group.surfaces]

    return lens

def build_lens_from_zmx(zmx_path):
    """
    Load lens from ZMX file using optiland's load_zemax_file function.
    The ZMX file already contains wavelengths, fields, and aperture settings.
    """
    lens = load_zemax_file(zmx_path)
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

def get_diameter(surface, lens=None, draw_called_list=None):
    """
    Get the diameter of a surface safely.

    Parameters
    ----------
    surface : Surface
        The surface object to get the diameter from
    lens : Optic, optional
        The lens/optic object, used for calling draw() if needed
    draw_called_list : list, optional
        A single-element list [bool] to track if draw() has been called

    Returns
    -------
    float
        The diameter of the surface
    """
    try:
        if surface.semi_aperture is not None and np.isfinite(surface.semi_aperture):
            return float(2 * surface.semi_aperture)
        else:
            raise ValueError("semi_aperture not available")
    except:
        # Call lens.draw() once if not already called
        if lens is not None and draw_called_list is not None and not draw_called_list[0]:
            try:
                lens.draw()
                draw_called_list[0] = True
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

def extract_optical_data(lens, surface_diameters=None):
    spot = analysis.SpotDiagram(lens, num_rings=20)
    fan = analysis.RayFan(lens)

    # Try to calculate distortion, but it may fail for some lens configurations
    distortion = None
    try:
        distortion = analysis.Distortion(lens)
    except Exception as e:
        print(f"⚠️ Distortion analysis failed: {e}", flush=True)
        print("Continuing without distortion data...", flush=True)
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
    # Use the module-level get_diameter function
    draw_called_list = [False]  # Track if draw() has been called
    diameters = [get_diameter(s, lens, draw_called_list) for s in lens.surface_group.surfaces]
    surfaces = [
        {
            "radius": float(s.geometry.radius) if np.isfinite(s.geometry.radius) else 1e6,
            "thickness": float(s.thickness) if np.isfinite(s.thickness) else 1e6,
            "diameter": get_diameter(s, lens, draw_called_list),
            "Is_Lens": float(type(s.material_post.n(lens.primary_wavelength))==np.ndarray),
            "coef": (
                list(s.geometry.coefficients)
                if hasattr(s.geometry, "coefficients") and s.geometry.coefficients is not None
                else []
            ),
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

        # Read the JSON file that was created
        with open(temp_path, 'r') as f:
            file_content = f.read()

        lens_json = json.loads(file_content)

        return lens_json
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
def creat_lens(request):
    # Initialize surface_diameters as None (will be populated from JSON if available)
    surface_diameters = None

    # === Check if a ZMX file was uploaded ===
    if 'zmx_file' in request.files:
        zmx_file = request.files['zmx_file']

        # Validate file
        if zmx_file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save temporarily
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"zmx_upload_{zmx_file.filename}")
        zmx_file.save(temp_path)

        try:
            # Build lens from ZMX file
            lens = build_lens_from_zmx(temp_path)
            use_optimization = False
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # === Original JSON-based approach ===
    else:
        payload = request.get_json(force=True)
        surfaces = payload["surfaces"]
        #Aperture = payload["aperture"]
        
        if "aperture" in payload:
            Aperture = payload["aperture"]
        else:
            Aperture = 10
            print("Default aperture selected: 10",flush=True)
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
        lens = build_lens(surfaces, light_sources, wavelengths,Aperture)
        if lens.surface_group.surfaces[-2].thickness==0:
            use_optimization = True
        else:
            for s in lens.surface_group.surfaces:
                s.is_stop = False
            # Set candidate
            lens.surface_group.surfaces[1].is_stop = True
            use_optimization=False

    return use_optimization,lens,surface_diameters

# -----------------------------------------
# API Route
# -----------------------------------------
@app.route("/tolerance", methods=["POST"])
def tolerance():
    result=tolerancing(request)
    res = df_to_columnar_payload(result, max_rows=1000)
    return jsonify(res)
@app.route("/optimize", methods=["POST"])
def optimize():

    print('Optimization has been called----------------', flush=True)
    lens,surface_diameters=optimize_opt(request)
    normalize_asphere_coefficients(lens)

    data = extract_optical_data(lens, surface_diameters)
    try:
        lens_json = save_lens_to_json(lens)
        # Convert to JSON string to avoid Infinity serialization issues
        # Frontend will save this string directly as a file
        data["lens_file"] = json.dumps(lens_json, indent=2, allow_nan=True)
    except Exception as e:
        print(f"⚠️ Failed to save lens JSON_optimize: {e}", flush=True)
        data["lens_file"] = None

    # Sanitize NaN and Inf values before returning JSON (lens_file is already a string, so it won't be affected)
    clean_data = sanitize_for_json(data)
    print('------------ end optimization ------',flush=True)
    return jsonify(clean_data)
@app.route("/analyze_geometry", methods=["POST"])
def analyze_geometry():
    print('----------------manufacturability is running------',flush=True)

    payload = request.get_json(force=True)

    # Ensure None radii are converted (the function also does it, but ok)
    for s in payload["surfaces"]:
        if s.get("radius") is None:
            s["radius"] = float("inf")
    surfaces = payload["surfaces"]
    print(surfaces,flush=True)
    out = geometry_summary_from_request(surfaces, n_pts=800)
    print('----------------manufacturability is done------',flush=True)
    print(out,flush=True)
    return jsonify(out)


@app.route("/simulate", methods=["POST"])

def simulate():
    try:
        print('----------------simulate is running------',flush=True)
        use_optimization,lens,surface_diameters=creat_lens(request)
        print('-------lens created-------',flush=True)


        lens.info()
        # === Apply optimization and stop surface finding (if needed) ===
        if use_optimization:
            lens=find_image_plane(lens)
            data = extract_optical_data(lens, surface_diameters)
        else:
            # No optimization needed, just extract data
            data = extract_optical_data(lens, surface_diameters)


        # Save lens as JSON and add to response
        try:
            lens_json = save_lens_to_json(lens)
            # Convert to JSON string to avoid Infinity serialization issues
            # Frontend will save this string directly as a file
            data["lens_file"] = json.dumps(lens_json, indent=2, allow_nan=True)
        except Exception as e:
            print(f"⚠️ Failed to save lens JSON_simulate: {e}", flush=True)
            data["lens_file"] = None

        # Sanitize NaN and Inf values before returning JSON (lens_file is already a string, so it won't be affected)
        clean_data = sanitize_for_json(data)
        print('----------------simulate is done------',flush=True)

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
