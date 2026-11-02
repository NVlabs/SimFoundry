"""Section 3.7 rigid-body USDA from the phase-1 JSON contract.

The phase-1 pipeline still writes ``usd_written: false``. This writer is the
step that emits a file. It does not load Isaac Lab or MuJoCo.
"""

from __future__ import annotations

from pathlib import Path


def _token(name):
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
    if not cleaned or cleaned[0].isdigit():
        cleaned = "obj_" + cleaned
    return cleaned


def _literal(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _float_attr(name, value):
    if value is None:
        return f"        # {name} is null in the contract"
    return f"        float {name} = {float(value)}"


def write_physics_usda(contract, path):
    """Write one Xform per contract object. Returns the path."""
    destination = Path(path)
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        '    doc = "SceneAgent section 3.7 rigid attributes from the JSON contract"',
        ")",
        "",
        'def Xform "World"',
        "{",
    ]
    for obj in contract["objects"]:
        lines.extend([
            f'    def Xform "{_token(obj["name"])}" (',
            '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"]',
            "    )",
            "    {",
            f'        bool physics:rigidBodyEnabled = {"true" if obj["rigid_body"] else "false"}',
            _float_attr("physics:mass", obj["mass_kg"]),
            _float_attr("physics:density", obj["density_kg_m3"]),
            _float_attr("physics:staticFriction", obj["friction_static"]),
            _float_attr("physics:dynamicFriction", obj["friction_dynamic"]),
            _float_attr("physics:restitution", obj["restitution"]),
            f'        string sceneagent:body_type = "{_literal(obj["body_type"])}"',
            f'        string sceneagent:collision = "{_literal(obj["collision_approximation"])}"',
            f'        string sceneagent:material_class = "{_literal(obj["material_class"])}"',
            "    }",
        ])
    lines.extend(["}", ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def _mesh_block(name, vertices, faces, physics):
    """One rigid mesh. ``physics`` uses the same keys as a contract object."""
    points = ", ".join(f"({float(x)}, {float(y)}, {float(z)})" for x, y, z in vertices)
    counts = ", ".join("3" for _ in faces)
    indices = ", ".join(str(int(index)) for face in faces for index in face)
    return [
        f'    def Xform "{_token(name)}" (',
        '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"]',
        "    )",
        "    {",
        f'        bool physics:rigidBodyEnabled = {"true" if physics.get("rigid_body") else "false"}',
        _float_attr("physics:mass", physics.get("mass_kg")),
        _float_attr("physics:density", physics.get("density_kg_m3")),
        _float_attr("physics:staticFriction", physics.get("friction_static")),
        _float_attr("physics:dynamicFriction", physics.get("friction_dynamic")),
        _float_attr("physics:restitution", physics.get("restitution")),
        f'        string sceneagent:body_type = "{_literal(physics.get("body_type"))}"',
        f'        string sceneagent:collision = "mesh"',
        f'        string sceneagent:material_class = "{_literal(physics.get("material_class"))}"',
        '        def Mesh "geometry"',
        "        {",
        f"            point3f[] points = [{points}]",
        f"            int[] faceVertexCounts = [{counts}]",
        f"            int[] faceVertexIndices = [{indices}]",
        "        }",
        "    }",
    ]


def write_aligned_mesh_usda(contract, vertices, faces, path, mesh_name="aligned_mesh", replace_name="box"):
    """Keep every contract object except ``replace_name``, and write that one as a mesh.

    Section 3.7. The mesh vertices are already in the stage frame.
    """
    destination = Path(path)
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        '    doc = "SceneAgent section 3.7 bowl plus the aligned SAM 3D mesh"',
        ")",
        "",
        'def Xform "World"',
        "{",
    ]
    replaced = False
    for obj in contract["objects"]:
        if obj["name"] == replace_name:
            lines.extend(_mesh_block(mesh_name, vertices, faces, obj))
            replaced = True
        else:
            lines.extend([
                f'    def Xform "{_token(obj["name"])}" (',
                '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"]',
                "    )",
                "    {",
                f'        bool physics:rigidBodyEnabled = {"true" if obj["rigid_body"] else "false"}',
                _float_attr("physics:mass", obj["mass_kg"]),
                _float_attr("physics:density", obj["density_kg_m3"]),
                _float_attr("physics:staticFriction", obj["friction_static"]),
                _float_attr("physics:dynamicFriction", obj["friction_dynamic"]),
                _float_attr("physics:restitution", obj["restitution"]),
                f'        string sceneagent:body_type = "{_literal(obj["body_type"])}"',
                f'        string sceneagent:collision = "{_literal(obj["collision_approximation"])}"',
                f'        string sceneagent:material_class = "{_literal(obj["material_class"])}"',
                "    }",
            ])
    if not replaced:
        raise KeyError(replace_name)
    lines.extend(["}", ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
