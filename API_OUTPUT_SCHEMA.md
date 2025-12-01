# Optiland API - Output Schema

## Response Structure

The `/simulate` endpoint returns a JSON object with the following structure:

```json
{
  "spot": { ... },
  "rayfan": { ... },
  "distortion": { ... } | null,
  "all_fields_rays": [ ... ],
  "all_fields_rays_3d": [ ... ],
  "surface_diameters": [ ... ],
  "surfaces": [ ... ],
  "paraxial": [ ... ],
  "lens_file": "..." | null
}
```

---

## Field Descriptions

### `spot`
Spot diagram data showing ray intersections at the image plane for each field and wavelength.

```json
{
  "spot": {
    "0,0": {
      "x": [array of x-coordinates],
      "y_centered": [array of y-coordinates centered at mean]
    },
    "0,1": { ... },
    "1,0": { ... }
  }
}
```

**Key format**: `"fieldIndex,wavelengthIndex"`
- Used to visualize aberrations and image quality
- Coordinates are in mm at the image plane

### `rayfan`
Ray aberration fans showing transverse ray errors vs. pupil position.

```json
{
  "rayfan": {
    "Py": [array of normalized pupil Y coordinates],
    "Px": [array of normalized pupil X coordinates],
    "fields": [
      {
        "field": "field value",
        "wavelengths": [
          {
            "wavelength": 0.55,
            "y": [array of Y-aberrations in mm],
            "x": [array of X-aberrations in mm]
          }
        ]
      }
    ]
  }
}
```

- `Py`, `Px`: Normalized pupil coordinates (-1 to +1)
- `y`, `x`: Transverse ray aberrations in mm

### `distortion`
Field distortion data (percentage distortion vs. field height).

```json
{
  "distortion": {
    "yaxis": [array of field heights],
    "wavelengths": [array of wavelengths],
    "data": [
      [array of distortion percentages for wavelength 1],
      [array of distortion percentages for wavelength 2]
    ]
  }
}
```

**Or `null`** if distortion analysis fails (e.g., for afocal systems).

### `all_fields_rays`
2D ray trace paths through the optical system for each field point.

```json
{
  "all_fields_rays": [
    {
      "field_number": 0,
      "Hx": 0.0,
      "Hy": 0.0,
      "x": [
        [surface1_ray1, surface1_ray2, ...],
        [surface2_ray1, surface2_ray2, ...],
        ...
      ],
      "y": [
        [surface1_ray1_height, surface1_ray2_height, ...],
        [surface2_ray1_height, surface2_ray2_height, ...],
        ...
      ]
    }
  ]
}
```

- **`x`**: Z-axis positions (optical axis) in mm
- **`y`**: Y-axis heights (perpendicular to optical axis) in mm
- Array dimensions: `[num_surfaces, num_rays]`
- Blocked rays show as `null` values
- Uses `line_y` distribution with 10 rays

### `all_fields_rays_3d`
3D ray trace paths for full spatial visualization.

```json
{
  "all_fields_rays_3d": [
    {
      "field_number": 0,
      "Hx": 0.0,
      "Hy": 0.0,
      "x": [...],  // Z-axis (optical axis)
      "y": [...],  // Y-axis (meridional plane)
      "z": [...]   // X-axis (sagittal plane)
    }
  ]
}
```

- Uses `hexapolar` distribution with 10 rays
- Full 3D coordinates for each ray at each surface

### `surface_diameters`
Array of calculated clear aperture diameters for each surface.

```json
{
  "surface_diameters": [10.5, 12.3, 15.2, 8.7, 10.0]
}
```

- Values in mm
- Calculated from `semi_aperture` or ray trace extents
- Fallback to 10.0 mm if calculation fails

### `surfaces`
Structured surface data with geometry and material properties.

```json
{
  "surfaces": [
    {
      "radius": 50.0,        // Radius of curvature (mm)
      "thickness": 5.0,      // Distance to next surface (mm)
      "diameter": 10.5,      // Clear aperture diameter (mm)
      "Is_Lens": 1.0         // 1.0 if surface is in glass, 0.0 if in air
    }
  ]
}
```

- **`radius`**: 1000000.0 for planar surfaces (originally Inf)
- **`thickness`**: 1000000.0 for infinite conjugates
- **`Is_Lens`**: Indicates if surface has refractive material

### `paraxial`
Paraxial (first-order) optical properties of the system.

```json
{
  "paraxial": [
    {
      "magnification": -0.25,
      "invariant": 1.234,
      "F-Number": 2.8,
      "Exit_pupil_diameter": 5.2,
      "Entrance_pupil_diameter": 10.0,
      "Front_focal_length": 48.5,
      "Back_focal_point": 52.3,
      "Front_focal_point": -48.5,
      "Front_principal_plane": 2.1,
      "Back_principal_plane": -2.1,
      "Front_nodal_plane": 0.0,
      "Back_nodal_plane": 0.0
    }
  ]
}
```

All distances in mm. Key properties:
- **magnification**: Image height / object height
- **F-Number**: Focal length / entrance pupil diameter
- **focal lengths**: Effective focal length of the system
- **principal planes**: Reference planes for paraxial calculations

### `lens_file`
Serialized lens configuration in Optiland's JSON format (as string).

```json
{
  "lens_file": "{\"aperture\": {...}, \"surfaces\": [...], \"fields\": [...], \"wavelengths\": [...]}"
}
```

- Contains complete lens prescription
- Can be saved as `.json` file and reloaded into Optiland
- `null` if serialization fails
- Use `JSON.parse()` to extract the nested structure

---

## Example Response

```json
{
  "spot": {
    "0,0": {
      "x": [-0.001, 0.002, -0.003, 0.001, 0.000],
      "y_centered": [0.002, -0.001, 0.003, -0.002, 0.000]
    }
  },
  "rayfan": {
    "Py": [-1.0, -0.5, 0.0, 0.5, 1.0],
    "Px": [-1.0, -0.5, 0.0, 0.5, 1.0],
    "fields": [
      {
        "field": "0.0",
        "wavelengths": [
          {
            "wavelength": 0.55,
            "y": [0.001, 0.0005, 0.0, -0.0005, -0.001],
            "x": [0.0, 0.0, 0.0, 0.0, 0.0]
          }
        ]
      }
    ]
  },
  "distortion": {
    "yaxis": [0.0, 2.5, 5.0],
    "wavelengths": [0.55],
    "data": [[0.0, -0.12, -0.45]]
  },
  "all_fields_rays": [
    {
      "field_number": 0,
      "Hx": 0.0,
      "Hy": 0.0,
      "x": [
        [0.0, 0.0, 0.0],
        [5.0, 5.0, 5.0],
        [50.0, 50.0, 50.0]
      ],
      "y": [
        [0.0, 2.5, 5.0],
        [0.5, 2.0, 3.5],
        [0.01, 0.0, -0.01]
      ]
    }
  ],
  "all_fields_rays_3d": [
    {
      "field_number": 0,
      "Hx": 0.0,
      "Hy": 0.0,
      "x": [[0.0, 0.0], [5.0, 5.0]],
      "y": [[0.0, 2.5], [0.5, 2.0]],
      "z": [[0.0, 0.0], [0.1, -0.1]]
    }
  ],
  "surface_diameters": [10.0, 12.5, 15.0, 10.0],
  "surfaces": [
    {
      "radius": 50.0,
      "thickness": 5.0,
      "diameter": 10.0,
      "Is_Lens": 0.0
    },
    {
      "radius": 1000000.0,
      "thickness": 45.0,
      "diameter": 12.5,
      "Is_Lens": 1.0
    }
  ],
  "paraxial": [
    {
      "magnification": -1.0,
      "invariant": 0.0,
      "F-Number": 5.0,
      "Exit_pupil_diameter": 10.0,
      "Entrance_pupil_diameter": 10.0,
      "Front_focal_length": 50.0,
      "Back_focal_point": 45.0,
      "Front_focal_point": -50.0,
      "Front_principal_plane": 0.0,
      "Back_principal_plane": 0.0,
      "Front_nodal_plane": 0.0,
      "Back_nodal_plane": 0.0
    }
  ],
  "lens_file": "{...complete lens JSON...}"
}
```

---

## Error Response

If an error occurs, the response will have status code 500 and format:

```json
{
  "error": "Error message describing what went wrong"
}
```

---

## Data Sanitization

- **NaN** values are converted to `null`
- **Infinity** values are converted to `null` (except in `lens_file` string)
- Ensures valid JSON serialization
- Frontend should handle `null` values gracefully (e.g., skip plotting)

---

## Visualization Tips

### Spot Diagram
Plot `spot[key].x` vs `spot[key].y_centered` as scatter plots, one per field/wavelength combination.

### Ray Fan
Plot `rayfan.Py` (x-axis) vs `rayfan.fields[i].wavelengths[j].y` (y-axis) for tangential fan.

### Ray Trace
Plot `all_fields_rays[i].x[surface_idx]` vs `all_fields_rays[i].y[surface_idx]` for 2D layout.

### Distortion
Plot `distortion.yaxis` (x-axis) vs `distortion.data[wavelength_idx]` (y-axis) as line plot.
