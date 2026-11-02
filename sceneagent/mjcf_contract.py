"""Section 3.7 MuJoCo XML from the phase-1 physics contract.

MuJoCo does not parse USDA. This writer is the repair after that rejection.
A missing mass or density stays null in the ledger. The geom then uses a
loader density so the file can be simulated. That density is not a paper lookup.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

# Used only when the contract leaves both mass and density null.
LOADER_DENSITY_KG_M3 = 1000.0
LOADER_HALF_EXTENT_M = 0.05


def _token(name):
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
    if not cleaned or cleaned[0].isdigit():
        cleaned = "obj_" + cleaned
    return cleaned


def _geom_type(primitive):
    return {
        "box": "box",
        "sphere": "sphere",
        "cylinder": "cylinder",
        "capsule": "capsule",
    }.get(primitive, "box")


def _size_attr(geom_type):
    half = LOADER_HALF_EXTENT_M
    if geom_type == "sphere":
        return f"{half}"
    if geom_type in ("cylinder", "capsule"):
        return f"{half} {half * 2}"
    return f"{half} {half} {half}"


def write_physics_mjcf(contract, path):
    """Write one body per contract object. Returns the path."""
    destination = Path(path)
    bodies = []
    for index, obj in enumerate(contract["objects"]):
        name = _token(obj["name"])
        geom_type = _geom_type(obj.get("collision_approximation"))
        static = obj.get("body_type") == "static rigid" or not obj.get("rigid_body")
        joint = "" if static else '<freejoint/>'
        mass = obj.get("mass_kg")
        density = obj.get("density_kg_m3")
        mass_attr = ""
        density_attr = ""
        mass_note = "contract"
        if mass is not None:
            mass_attr = f' mass="{float(mass)}"'
        elif density is not None:
            density_attr = f' density="{float(density)}"'
        else:
            density_attr = f' density="{LOADER_DENSITY_KG_M3}"'
            mass_note = "contract_null_loader_density"
        slide = obj.get("friction_static")
        spin = obj.get("friction_dynamic")
        friction = ""
        if slide is not None or spin is not None:
            friction = f' friction="{float(slide or 0)} {float(spin or 0)} 0.001"'
        x = 0.25 * index
        bodies.append(
            "\n".join([
                f'    <body name="{name}" pos="{x} 0 0.2">',
                f"      {joint}" if joint else "      <!-- static rigid: no free joint -->",
                (
                    f'      <geom name="{name}_geom" type="{geom_type}" size="{_size_attr(geom_type)}"'
                    f'{mass_attr}{density_attr}{friction}/>'
                ),
                f"      <!-- material_class={escape(str(obj.get('material_class')))} "
                f"mass_source={mass_note} restitution={obj.get('restitution')} "
                f"body_type={escape(str(obj.get('body_type')))} -->",
                "    </body>",
            ])
        )
    xml = "\n".join([
        "<mujoco model=\"sceneagent_contract\">",
        "  <!-- Section 3.7 rigid attributes. Deformables are not in this file. -->",
        "  <option gravity=\"0 0 -9.81\" timestep=\"0.002\"/>",
        "  <worldbody>",
        *bodies,
        "  </worldbody>",
        "</mujoco>",
        "",
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(xml, encoding="utf-8")
    return destination
