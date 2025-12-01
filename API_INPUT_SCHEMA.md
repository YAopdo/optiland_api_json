# Optiland API - Input Schema

## Endpoint
```
POST /simulate
Content-Type: application/json (for JSON mode)
Content-Type: multipart/form-data (for ZMX file mode)
```

## Input Mode 1: JSON Payload

### Request Body Structure

```json
{
  "surfaces": [
    {
      "radius": number | null,           // Surface radius (null = infinite/planar)
      "thickness": number,                // Distance to next surface
      "index": number,                    // Refractive index (optional, default: air)
      "diameter": number,                 // Surface diameter (optional)
      "surface_type": string,             // Surface type (optional)
      "conic": number,                    // Conic constant (optional)
      "coefficients": array               // Aspheric coefficients (optional)
    }
  ],
  "lightSources": [
    {
      "type": "infinity" | "point",

      // For type="infinity":
      "angle": number,                    // Field angle in degrees

      // For type="point":
      "x": number,                        // Object distance
      "y": number                         // Object height
    }
  ],
  "wavelengths": [
    number,                               // Wavelength values in micrometers
    ...
  ]
}
```

### Field Descriptions

#### `surfaces` (required)
Array of surface objects defining the optical system.

- **`radius`**: Radius of curvature in mm
  - Positive = center of curvature to the right
  - Negative = center of curvature to the left
  - `null` or `Infinity` = planar surface

- **`thickness`**: Distance to next surface in mm
  - Set to `0` on the last surface to trigger automatic image plane optimization

- **`index`**: Refractive index (e.g., 1.5168 for N-BK7 glass)
  - Omit or null for air gaps

- **`diameter`**: Surface clear aperture diameter in mm (optional)
  - Used for ray blocking at apertures

- **`surface_type`**: Surface type string (optional)
  - Examples: "standard", "even_asphere"

- **`conic`**: Conic constant for aspheric surfaces (optional)

- **`coefficients`**: Array of aspheric coefficients (optional)

#### `lightSources` (optional)
Array of light source configurations. If omitted, defaults to two on-axis fields at 0° and 5°.

**Type: "infinity"** (collimated light from infinity)
- **`angle`**: Field angle in degrees

**Type: "point"** (point source at finite distance)
- **`x`**: Object distance in mm
- **`y`**: Object height in mm

#### `wavelengths` (optional)
Array of wavelengths in micrometers (μm). First wavelength is set as primary.
- If omitted, defaults to 0.55 μm (green light)

---

## Input Mode 2: ZMX File Upload

### Multipart Form Data
```
POST /simulate
Content-Type: multipart/form-data

Form field: zmx_file
File: [Zemax .zmx file]
```

Upload a Zemax (.zmx) lens prescription file. The file contains:
- Surface definitions (curvature, thickness, glass)
- Aperture settings
- Wavelengths
- Field points

---

## Example Requests

### Example 1: Simple Biconvex Lens

```json
{
  "surfaces": [
    {
      "radius": 50.0,
      "thickness": 5.0,
      "index": 1.5168
    },
    {
      "radius": -50.0,
      "thickness": 0
    }
  ],
  "lightSources": [
    {
      "type": "infinity",
      "angle": 0
    }
  ],
  "wavelengths": [0.55]
}
```

### Example 2: Point Source with Multiple Wavelengths

```json
{
  "surfaces": [
    {
      "radius": 30.0,
      "thickness": 4.0,
      "index": 1.5168,
      "diameter": 20.0
    },
    {
      "radius": -30.0,
      "thickness": 0,
      "diameter": 20.0
    }
  ],
  "lightSources": [
    {
      "type": "point",
      "x": -100.0,
      "y": 0.0
    },
    {
      "type": "point",
      "x": -100.0,
      "y": 5.0
    }
  ],
  "wavelengths": [0.486, 0.588, 0.656]
}
```

### Example 3: Aspheric Lens

```json
{
  "surfaces": [
    {
      "radius": 25.0,
      "thickness": 3.5,
      "index": 1.5168,
      "surface_type": "even_asphere",
      "conic": -0.5,
      "coefficients": [0, 0, 1e-5, 1e-7]
    },
    {
      "radius": null,
      "thickness": 0
    }
  ],
  "lightSources": [
    {
      "type": "infinity",
      "angle": 0
    }
  ],
  "wavelengths": [0.55]
}
```

### Example 4: Python Request using `requests`

```python
import requests
import json

url = "http://localhost:5000/simulate"

payload = {
    "surfaces": [
        {"radius": 50.0, "thickness": 5.0, "index": 1.5168},
        {"radius": -50.0, "thickness": 0}
    ],
    "lightSources": [
        {"type": "infinity", "angle": 0}
    ],
    "wavelengths": [0.55]
}

response = requests.post(url, json=payload)
result = response.json()

print(result.keys())  # ['spot', 'rayfan', 'distortion', 'all_fields_rays', ...]
```

### Example 5: JavaScript/Fetch Request

```javascript
const url = 'http://localhost:5000/simulate';

const payload = {
  surfaces: [
    { radius: 50.0, thickness: 5.0, index: 1.5168 },
    { radius: -50.0, thickness: 0 }
  ],
  lightSources: [
    { type: 'infinity', angle: 0 }
  ],
  wavelengths: [0.55]
};

fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload)
})
  .then(response => response.json())
  .then(data => console.log(data))
  .catch(error => console.error('Error:', error));
```

### Example 6: cURL with ZMX File

```bash
curl -X POST http://localhost:5000/simulate \
  -F "zmx_file=@/path/to/lens.zmx"
```

---

## Special Behaviors

### Automatic Image Plane Optimization
If the last surface has `thickness: 0`, the system will:
1. Automatically find the best stop surface position
2. Optimize the image plane distance for minimum spot size
3. Try different stop surface positions until one succeeds

### Surface Diameter Ray Blocking
When diameters are specified for all surfaces:
- Rays exceeding the diameter at any surface are blocked
- Blocked rays show as NaN in ray trace data

### Default Values
- **No light sources**: Defaults to infinity type at 0° and 5°
- **No wavelengths**: Defaults to 0.55 μm (green)
- **No index**: Air (n=1.0)

---

## Notes

- All distances are in millimeters (mm)
- All wavelengths are in micrometers (μm)
- Angles are in degrees
- First wavelength in array becomes the primary wavelength
- CORS is enabled for cross-origin requests
