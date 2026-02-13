# SPDX-License-Identifier: MIT
# SketchUp Tools V3.1 - True SketchUp Behavior inside Blender
# Core philosophy: Everything draws into ONE mesh, smart-welds, auto-faces, continuous drawing.

bl_info = {
    "name": "SketchUp Tools V5 – Fixed",
    "author": "Sultan + OpenClaw + AI Enhancement",
    "version": (5, 0, 1),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > SketchUp | Add > SketchUp",
    "description": "SketchUp behavior: smart-weld, auto-face, face-split, 6 arc types, push/pull, GPU preview",
    "category": "3D View",
}

import bpy
import bmesh
import traceback
import time
import math
try:
    import blf
    BLF_AVAILABLE = True
except Exception:
    blf = None
    BLF_AVAILABLE = False
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree
from bpy_extras import view3d_utils

try:
    import gpu
    from gpu_extras.batch import batch_for_shader
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

# =============================================================================
# GLOBALS & CONSTANTS
# =============================================================================
ADDON_ID = __package__ or __name__
_IMPORT_ERROR_TEXT = ""
_registered = False
_addon_keymaps = []
_timer_scheduled = False

# Performance
KDTREE_CACHE_TIME = 0.3
MAX_SNAP_DISTANCE_WORLD = 50.0
MAX_INPUT_LENGTH = 100
MAX_VALUE_MAGNITUDE = 1e6
MERGE_DISTANCE = 0.001  # NEW: weld threshold (1mm)

# Colors (RGBA)
SNAP_COLOR = (0.2, 1.0, 0.2, 1.0)
PREVIEW_COLOR = (1.0, 0.7, 0.0, 0.8)
AXIS_X_COLOR = (1.0, 0.0, 0.0, 1.0)
AXIS_Y_COLOR = (0.0, 1.0, 0.0, 1.0)
AXIS_Z_COLOR = (0.0, 0.5, 1.0, 1.0)
AXIS_LOCKED_COLOR = (1.0, 1.0, 0.0, 1.0)
AXIS_DIM_ALPHA = 0.25
FACE_FILL_COLOR = (0.3, 0.6, 1.0, 0.15)  # NEW: auto-face preview
WELD_MARKER_COLOR = (1.0, 0.5, 0.0, 1.0)  # NEW: weld indicator
ARC_PREVIEW_COLOR = (1.0, 0.7, 0.0, 0.9)
ARC_CENTER_COLOR = (0.9, 0.2, 0.9, 1.0)
ARC_TANGENT_COLOR = (0.2, 0.9, 0.9, 1.0)

# VCB colors
VCB_BG_COLOR = (0.08, 0.08, 0.08, 1.00)  # Fully opaque for better visibility
VCB_TEXT_COLOR = (1.00, 1.00, 1.00, 1.00)
VCB_HIGHLIGHT_COLOR = (1.00, 0.85, 0.20, 0.35)

# Arc defaults
DEFAULT_ARC_SEGMENTS = 24
MIN_ARC_SEGMENTS = 6
MAX_ARC_SEGMENTS = 256

# Navigation pass-through
_NAV_EVENTS = {
    'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
    'WHEELINMOUSE', 'WHEELOUTMOUSE',
    'TRACKPADPAN', 'TRACKPADZOOM',
}

# GPU draw handler storage
_draw_handlers = {}

# VCB (Value Control Box) state
_vcb_state = {
    'visible': False,
    'label': '',
    'value': '',
    'typed': '',
    'position': (20, 40),  # bottom-left
}
_vcb_last_err_t = 0.0


def update_vcb(label="", value="", typed="", visible=True):
    """Update the VCB display state."""
    global _vcb_state
    _vcb_state.update({
        'label': label or '',
        'value': value or '',
        'typed': typed or '',
        'visible': bool(visible),
    })


def hide_vcb():
    """Hide the VCB."""
    global _vcb_state
    _vcb_state['visible'] = False


# =============================================================================
# TARGET MESH MANAGER  (NEW – core of SketchUp behavior)
# =============================================================================
TARGET_MESH_NAME = "SU_Model"


def _get_target_name(context):
    """Return the name of the current target mesh for snap priority."""
    obj = context.active_object
    if obj and obj.type == 'MESH' and not obj.name.startswith("SU_TMP"):
        return obj.name
    if TARGET_MESH_NAME in bpy.data.objects:
        return TARGET_MESH_NAME
    return None


def get_or_create_target(context):
    """
    Return the single target mesh object.  Create it once if it doesn't exist.
    If the user has an active Mesh selected, use that instead (allows multi-model).
    """
    # Option A – user's active mesh
    obj = context.active_object
    if obj and obj.type == 'MESH' and not obj.name.startswith("SU_TMP"):
        return obj

    # Option B – find existing SU_Model
    if TARGET_MESH_NAME in bpy.data.objects:
        obj = bpy.data.objects[TARGET_MESH_NAME]
        if obj.name in context.view_layer.objects:
            return obj

    # Option C – create fresh
    mesh = bpy.data.meshes.new(TARGET_MESH_NAME)
    obj = bpy.data.objects.new(TARGET_MESH_NAME, mesh)
    safe_link_object(context, obj)
    return obj


def commit_geometry_to_target(context, verts_world, edges_idx, faces_idx=None,
                              merge_dist=None):
    """
    Add geometry into the target mesh with smart-weld, face-split, and auto-face.
    When new edges cross existing faces, those faces are split (SketchUp behavior).
    """
    if merge_dist is None:
        merge_dist = MERGE_DISTANCE

    target = get_or_create_target(context)

    was_edit = (context.mode == 'EDIT_MESH' and context.active_object == target)
    if was_edit:
        bm = bmesh.from_edit_mesh(target.data)
    else:
        bm = bmesh.new()
        bm.from_mesh(target.data)

    inv = target.matrix_world.inverted()
    local_verts = [inv @ v for v in verts_world]

    # ---- SMART WELD: build KDTree of existing verts, reuse if within dist ----
    bm.verts.ensure_lookup_table()
    existing_count = len(bm.verts)
    exist_kd = None
    if existing_count > 0:
        exist_kd = KDTree(existing_count)
        for i, v in enumerate(bm.verts):
            exist_kd.insert(v.co, i)
        exist_kd.balance()

    new_bmverts = []
    truly_new_verts = []
    batch_new_cos = []

    for co in local_verts:
        reused = False
        if exist_kd is not None:
            found_co, found_idx, found_dist = exist_kd.find(co)
            if found_idx is not None and found_dist <= merge_dist:
                bm.verts.ensure_lookup_table()
                new_bmverts.append(bm.verts[found_idx])
                reused = True
        if not reused:
            for bco, bv in batch_new_cos:
                if (bco - co).length <= merge_dist:
                    new_bmverts.append(bv)
                    reused = True
                    break
        if not reused:
            v = bm.verts.new(co)
            new_bmverts.append(v)
            truly_new_verts.append(v)
            batch_new_cos.append((co.copy(), v))

    bm.verts.ensure_lookup_table()

    # ---- FACE SPLIT: before adding edges, check if new verts land on existing faces ----
    # For each truly new vert that lies on an existing face, split that face
    _face_split_for_new_verts(bm, truly_new_verts, merge_dist)

    # ---- REMAP VERT REFERENCES AFTER OPERATIONS THAT MAY MERGE/REMOVE VERTS ----
    # Some face-splitting / point-merge operations can invalidate BMVert handles stored in new_bmverts.
    # We remap any invalid entries by locating the nearest valid vert to the intended coordinate.
    bm.verts.ensure_lookup_table()
    try:
        remap_kd = None
        if len(bm.verts) > 0:
            remap_kd = KDTree(len(bm.verts))
            for _i, _v in enumerate(bm.verts):
                if _v.is_valid:
                    remap_kd.insert(_v.co, _i)
            remap_kd.balance()
        if remap_kd is not None:
            for _idx, _vref in enumerate(new_bmverts):
                if _vref is None or (hasattr(_vref, "is_valid") and not _vref.is_valid):
                    _co = local_verts[_idx]
                    _fco, _find_idx, _fdist = remap_kd.find(_co)
                    if _find_idx is not None and _fdist <= (merge_dist * 10.0):
                        bm.verts.ensure_lookup_table()
                        new_bmverts[_idx] = bm.verts[_find_idx]
    except Exception:
        # Best-effort only; validity guards below will still prevent crashes.
        pass

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # ---- Add edges (skip if already exists between those two verts) ----
    new_edges = []
    for i, j in edges_idx:
        va, vb = new_bmverts[i], new_bmverts[j]
        # Guard: earlier merge operations may invalidate stored vert refs
        if (va is None) or (vb is None):
            continue
        try:
            if (hasattr(va, "is_valid") and not va.is_valid) or (hasattr(vb, "is_valid") and not vb.is_valid):
                continue
        except Exception:
            continue
        if va == vb:
            continue
        edge_exists = False
        for e in va.link_edges:
            if e.other_vert(va) == vb:
                new_edges.append(e)
                edge_exists = True
                break
        if not edge_exists:
            try:
                e = bm.edges.new((va, vb))
                new_edges.append(e)
            except ValueError:
                pass

    # ---- ENSURE VERTICES ARE ON FACE BOUNDARIES BEFORE SPLITTING ----
    # For each new edge, ensure its vertices are on face boundaries if they intersect faces
    _ensure_verts_on_face_boundaries(bm, new_edges, merge_dist)
    
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # ---- FACE SPLIT for new edges crossing existing faces ----
    _face_split_for_new_edges(bm, new_edges, merge_dist)

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # ---- FINALIZE FACE SPLITS: ensure all new edges properly split faces ----
    _finalize_face_splits(bm, new_edges, merge_dist)
    
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # ---- Add explicit faces ----
    if faces_idx:
        # After merges/splits some BMVert references stored in new_bmverts can become invalid.
        # Remap face vertices by nearest-coordinate lookup (best-effort) and skip invalid faces.
        bm.verts.ensure_lookup_table()
        _face_kd = None
        try:
            if len(bm.verts) > 0:
                _face_kd = KDTree(len(bm.verts))
                for _i, _v in enumerate(bm.verts):
                    if _v.is_valid:
                        _face_kd.insert(_v.co, _i)
                _face_kd.balance()
        except Exception:
            _face_kd = None

        def _safe_face_vert(_k):
            if _k is None:
                return None
            try:
                _v = new_bmverts[_k]
            except Exception:
                return None
            try:
                if _v is not None and getattr(_v, "is_valid", False):
                    return _v
            except Exception:
                pass
            if _face_kd is None:
                return None
            try:
                _co = local_verts[_k]
                _fco, _idx, _dist = _face_kd.find(_co)
                if _idx is None or _dist > (merge_dist * 10.0):
                    return None
                bm.verts.ensure_lookup_table()
                _vv = bm.verts[_idx]
                if _vv is not None and getattr(_vv, "is_valid", False):
                    new_bmverts[_k] = _vv
                    return _vv
            except Exception:
                return None
            return None

        for ftup in faces_idx:
            try:
                fverts = [_safe_face_vert(k) for k in ftup]
            except Exception:
                continue
            if any(v is None for v in fverts):
                continue
            # Unique valid verts
            try:
                if len({v for v in fverts if getattr(v, "is_valid", False)}) < 3:
                    continue
            except Exception:
                continue
            try:
                bm.faces.new(fverts)
            except (ValueError, TypeError):
                # ValueError: face exists / invalid; TypeError: invalid verts
                pass

    # ---- Remove doubles ONLY on truly new verts (not entire mesh) ----
    if truly_new_verts:
        try:
            valid_verts = [v for v in truly_new_verts if hasattr(v, 'is_valid') and v.is_valid]
            if valid_verts:
                bmesh.ops.remove_doubles(bm, verts=valid_verts, dist=merge_dist)
        except (ReferenceError, RuntimeError):
            pass

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # ---- AUTO-FACE: only on new edges, with planarity check ----
    prefs = _get_prefs(context)
    if prefs and prefs.auto_face:
        valid_new_edges = [e for e in new_edges if e.is_valid]
        _auto_create_faces(bm, valid_new_edges)

    if was_edit:
        bmesh.update_edit_mesh(target.data, loop_triangles=True, destructive=True)
    else:
        bm.to_mesh(target.data)
        target.data.update()
        bm.free()

    _kdtree_cache.invalidate()
    return target


def _face_split_for_new_verts(bm, new_verts, merge_dist):
    """Split existing faces that contain new interior vertices."""
    if not new_verts:
        return
    
    # Build a simple spatial lookup: only check faces near each vertex
    face_list = [f for f in bm.faces if f.is_valid]
    if not face_list:
        return
    
    for nv in new_verts:
        if not nv.is_valid:
            continue
            
        co = nv.co
        
        # Find face containing this vert (not on boundary)
        for face in face_list:
            if not face.is_valid:
                continue
                
            if nv in face.verts:
                continue
                
            fn = face.normal
            if fn.length < 1e-9:
                continue
                
            dist_to_plane = abs((co - face.verts[0].co).dot(fn))
            if dist_to_plane > merge_dist * 10:
                continue
                
            if not _point_in_face(face, co, merge_dist * 5):
                continue
                
            _split_edge_for_interior_vertex(bm, face, nv, merge_dist)
            break
    
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()


def _face_split_for_new_edges(bm, new_edges, merge_dist):
    """Split faces that are crossed by new edges (SketchUp face-split behavior)."""
    if not new_edges:
        return
    
    # Build chain of connected edges
    edge_chains = []
    processed = set()
    
    for edge in new_edges:
        if not edge.is_valid or edge in processed:
            continue
        
        # Start a new chain
        chain = []
        stack = [edge]
        
        while stack:
            current = stack.pop()
            if current in processed or not current.is_valid:
                continue
                
            processed.add(current)
            chain.append(current)
            
            # Find connected edges in new_edges
            for v in current.verts:
                if not v.is_valid:
                    continue
                for e in v.link_edges:
                    if e in new_edges and e not in processed and e.is_valid:
                        stack.append(e)
        
        if chain:
            edge_chains.append(chain)
    
    # Process each chain
    for chain in edge_chains:
        if not chain:
            continue
            
        # Get all vertices in this chain
        chain_verts = set()
        for edge in chain:
            if edge.is_valid:
                chain_verts.update(edge.verts)
        
        if len(chain_verts) < 2:
            continue
            
        # Convert to ordered list of vertices along the chain
        ordered_verts = _order_verts_in_chain(chain)
        if len(ordered_verts) < 2:
            continue
            
        # Process each segment in the chain
        for i in range(len(ordered_verts) - 1):
            v1 = ordered_verts[i]
            v2 = ordered_verts[i + 1]
            
            if not v1.is_valid or not v2.is_valid:
                continue
                
            # Find faces that might be split by this segment
            _split_face_with_segment(bm, v1, v2, merge_dist)
    
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()


def _order_verts_in_chain(edge_chain):
    """Order vertices along a chain of connected edges."""
    if not edge_chain:
        return []
    
    # Build adjacency map
    adj = {}
    for edge in edge_chain:
        if not edge.is_valid:
            continue
        v1, v2 = edge.verts
        adj.setdefault(v1, set()).add(v2)
        adj.setdefault(v2, set()).add(v1)
    
    # Find endpoints (vertices with only one connection in this chain)
    endpoints = [v for v, neighbors in adj.items() if len(neighbors) == 1]
    
    if not endpoints:
        # Closed loop, start anywhere
        start_vert = next(iter(adj.keys()))
    else:
        # Open chain, start at an endpoint
        start_vert = endpoints[0]
    
    # Traverse the chain
    ordered = [start_vert]
    visited = {start_vert}
    current = start_vert
    
    while True:
        neighbors = adj.get(current, set())
        unvisited = [n for n in neighbors if n not in visited]
        
        if not unvisited:
            break
            
        next_vert = unvisited[0]
        ordered.append(next_vert)
        visited.add(next_vert)
        current = next_vert
    
    return ordered


def _split_face_with_segment(bm, v1, v2, merge_dist):
    """Split a face with a line segment between two vertices."""
    if not v1.is_valid or not v2.is_valid:
        return
        
    # Only check faces connected to v1 or v2 (not ALL faces)
    candidate_faces = []
    checked = set()
    
    for v in (v1, v2):
        if not v.is_valid:
            continue
        for f in v.link_faces:
            if f.is_valid and f not in checked:
                checked.add(f)
                candidate_faces.append(f)
    
    # Also check faces whose edges are near the segment midpoint
    mid = (v1.co + v2.co) * 0.5
    for f in bm.faces:
        if not f.is_valid or f in checked:
            continue
        fn = f.normal
        if fn.length < 1e-9:
            continue
        d1 = abs((v1.co - f.verts[0].co).dot(fn))
        d2 = abs((v2.co - f.verts[0].co).dot(fn))
        if d1 > merge_dist * 10 and d2 > merge_dist * 10:
            continue
        if _point_in_face(f, mid, merge_dist * 5):
            checked.add(f)
            candidate_faces.append(f)
    
    for face in candidate_faces:
        if not face.is_valid:
            continue
            
        # Check if edge already exists on this face
        edge_exists = False
        for e in face.edges:
            if (e.verts[0] == v1 and e.verts[1] == v2) or (e.verts[0] == v2 and e.verts[1] == v1):
                edge_exists = True
                break
        
        if edge_exists:
            continue
        
        # If one or both endpoints are not on face boundary, bring them there first
        try:
            if v1 not in face.verts:
                _split_edge_for_interior_vertex(bm, face, v1, merge_dist)
            if face.is_valid and v2 not in face.verts:
                _split_edge_for_interior_vertex(bm, face, v2, merge_dist)
        except Exception:
            pass

        # Handle the segment on this face
        if face.is_valid:
            _split_face_with_segment_on_face(bm, face, v1, v2, merge_dist)


def _segment_intersects_face(face, p1, p2, tol=1e-6):
    """Check if line segment p1-p2 intersects the face polygon."""
    if not face or not face.is_valid:
        return False
    
    # Quick bounding box check
    face_verts = [v.co for v in face.verts]
    face_min = Vector((
        min(v.x for v in face_verts),
        min(v.y for v in face_verts),
        min(v.z for v in face_verts)
    ))
    face_max = Vector((
        max(v.x for v in face_verts),
        max(v.y for v in face_verts),
        max(v.z for v in face_verts)
    ))
    
    # Expand bbox by tolerance
    face_min -= Vector((tol, tol, tol))
    face_max += Vector((tol, tol, tol))
    
    # Check if segment bounding box intersects face bbox
    seg_min = Vector((
        min(p1.x, p2.x),
        min(p1.y, p2.y),
        min(p1.z, p2.z)
    ))
    seg_max = Vector((
        max(p1.x, p2.x),
        max(p1.y, p2.y),
        max(p1.z, p2.z)
    ))
    
    if (seg_max.x < face_min.x or seg_min.x > face_max.x or
        seg_max.y < face_min.y or seg_min.y > face_max.y or
        seg_max.z < face_min.z or seg_min.z > face_max.z):
        return False
    
    # More detailed check: test if segment crosses face boundary or is inside
    # Check if either endpoint is inside face
    if _point_in_face(face, p1, tol) or _point_in_face(face, p2, tol):
        return True
    
    # Check if segment intersects any face edge
    dir_vec = p2 - p1
    seg_len = dir_vec.length
    if seg_len < 1e-9:
        return _point_in_face(face, p1, tol)
    
    for edge in face.edges:
        if not edge.is_valid:
            continue
            
        e1 = edge.verts[0].co
        e2 = edge.verts[1].co
        edge_vec = e2 - e1
        edge_len = edge_vec.length
        
        if edge_len < 1e-9:
            continue
            
        # Check for intersection
        cross_de = dir_vec.cross(edge_vec)
        if cross_de.length < 1e-12:
            continue  # Parallel
            
        diff = e1 - p1
        try:
            t = diff.cross(edge_vec).dot(cross_de) / cross_de.length_squared
            u = diff.cross(dir_vec).dot(cross_de) / cross_de.length_squared
        except ZeroDivisionError:
            continue
            
        if -tol <= t <= 1 + tol and -tol <= u <= 1 + tol:
            return True
    
    return False


def _split_face_with_segment_on_face(bm, face, v1, v2, merge_dist):
    """Split a specific face with a segment between two vertices."""
    if not face.is_valid or not v1.is_valid or not v2.is_valid:
        return
    
    # Step 1: Ensure both vertices are on face boundary
    # If vertices are inside face, find where they intersect face edges
    
    # Check if v1 is on face boundary
    v1_on_boundary = v1 in face.verts
    if not v1_on_boundary:
        # Find intersection with face edges
        edge1, t1, intersect1 = _find_face_edge_intersection(face, v1.co, v2.co, merge_dist)
        if edge1 and edge1.is_valid:
            # Split the edge at intersection point
            try:
                new_edge, split_vert = bmesh.utils.edge_split(edge1, edge1.verts[0], t1)
                # Merge split vertex with v1 if close
                if split_vert and split_vert.is_valid:
                    if (split_vert.co - v1.co).length < merge_dist * 10:
                        bmesh.ops.pointmerge(bm, verts=[split_vert, v1], merge_co=intersect1)
                        v1_on_boundary = True
            except Exception:
                pass
    
    # Check if v2 is on face boundary  
    v2_on_boundary = v2 in face.verts
    if not v2_on_boundary:
        # Find intersection with face edges (reverse direction)
        edge2, t2, intersect2 = _find_face_edge_intersection(face, v2.co, v1.co, merge_dist)
        if edge2 and edge2.is_valid:
            # Split the edge at intersection point
            try:
                new_edge, split_vert = bmesh.utils.edge_split(edge2, edge2.verts[0], t2)
                # Merge split vertex with v2 if close
                if split_vert and split_vert.is_valid:
                    if (split_vert.co - v2.co).length < merge_dist * 10:
                        bmesh.ops.pointmerge(bm, verts=[split_vert, v2], merge_co=intersect2)
                        v2_on_boundary = True
            except Exception:
                pass
    
    # If still not on boundary, try projecting onto closest edge
    if not v1_on_boundary:
        _split_edge_for_interior_vertex(bm, face, v1, merge_dist)
        v1_on_boundary = v1 in face.verts
        
    if not v2_on_boundary:
        _split_edge_for_interior_vertex(bm, face, v2, merge_dist)
        v2_on_boundary = v2 in face.verts
    
    # Step 2: Connect vertices if both are on boundary
    if v1_on_boundary and v2_on_boundary and v1 != v2:
        # Check if vertices are adjacent on face boundary
        face_verts = list(face.verts)
        try:
            idx1 = face_verts.index(v1)
            idx2 = face_verts.index(v2)
        except ValueError:
            return
            
        n = len(face_verts)
        # Skip if vertices are adjacent
        if abs(idx1 - idx2) == 1 or abs(idx1 - idx2) == n - 1:
            return
            
        # Check if edge already exists
        edge_exists = False
        for e in v1.link_edges:
            if e.other_vert(v1) == v2:
                edge_exists = True
                break
        
        if not edge_exists:
            try:
                bmesh.utils.face_split(face, v1, v2)
            except Exception:
                try:
                    result = bmesh.ops.connect_verts(bm, verts=[v1, v2])
                except Exception:
                    pass
            return


def _ensure_verts_on_face_boundaries(bm, new_edges, merge_dist):
    """Ensure vertices of new edges are on face boundaries if they intersect faces."""
    if not new_edges:
        return
    
    # Collect all vertices from new edges
    all_verts = set()
    for edge in new_edges:
        if edge.is_valid:
            all_verts.update(edge.verts)
    
    face_list = [f for f in bm.faces if f.is_valid]
    for vert in all_verts:
        if not vert.is_valid:
            continue
        # Skip if already on a face boundary
        if vert.link_faces:
            continue
        for face in face_list:
            if not face.is_valid:
                continue
            if vert in face.verts:
                continue
            fn = face.normal
            if fn.length < 1e-9:
                continue
            dist_to_plane = abs((vert.co - face.verts[0].co).dot(fn))
            if dist_to_plane > merge_dist * 10:
                continue
            if not _point_in_face(face, vert.co, merge_dist * 5):
                continue
            _split_edge_for_interior_vertex(bm, face, vert, merge_dist)
            break


def _split_edge_for_interior_vertex(bm, face, vertex, merge_dist):
    """Split an edge of a face to bring an interior vertex to the boundary."""
    if not face.is_valid or not vertex.is_valid:
        return
        
    # Find the edge that the vertex projects onto
    best_edge = None
    best_t = 0.5
    best_dist = float('inf')
    
    for edge in face.edges:
        if not edge.is_valid:
            continue
            
        v1 = edge.verts[0]
        v2 = edge.verts[1]
        if not v1.is_valid or not v2.is_valid:
            continue
            
        # Calculate projection onto edge line
        edge_vec = v2.co - v1.co
        edge_len = edge_vec.length
        
        if edge_len < 1e-9:
            continue
            
        t = (vertex.co - v1.co).dot(edge_vec) / (edge_len * edge_len)
        
        # Only consider projections that are on the edge segment (with tolerance)
        if t < -0.01 or t > 1.01:
            continue
            
        proj = v1.co + edge_vec * max(0.0, min(1.0, t))
        dist = (vertex.co - proj).length
        
        if dist < best_dist:
            best_dist = dist
            best_edge = edge
            best_t = max(0.01, min(0.99, t))  # Keep away from endpoints
    
    if best_edge and best_edge.is_valid and best_dist < merge_dist * 10:
        try:
            # Split the edge
            new_edge, split_vert = bmesh.utils.edge_split(best_edge, best_edge.verts[0], best_t)
            
            # Merge the split vertex with our vertex
            if split_vert and split_vert.is_valid:
                # Use pointmerge to weld them
                bmesh.ops.pointmerge(bm, verts=[split_vert, vertex], merge_co=vertex.co)
                
                # Update vertex reference (vertex now at split location)
                # The original vertex might be invalid after merge, so we need to find it
                bm.verts.ensure_lookup_table()
                        
        except Exception:
            pass


def _split_face_with_interior_segment(bm, face, v1, v2, merge_dist):
    """Split a face with a segment where vertices might be interior or on edges."""
    if not face.is_valid or not v1.is_valid or not v2.is_valid:
        return
    
    # Check if vertices are on face plane
    face_normal = face.normal
    if face_normal.length < 1e-9:
        return
        
    # Distance to plane check
    dist1 = abs((v1.co - face.verts[0].co).dot(face_normal))
    dist2 = abs((v2.co - face.verts[0].co).dot(face_normal))
    
    if dist1 > merge_dist * 10 or dist2 > merge_dist * 10:
        return
    
    # Check if points are inside face
    inside1 = _point_in_face(face, v1.co, merge_dist * 5)
    inside2 = _point_in_face(face, v2.co, merge_dist * 5)
    
    if not inside1 and not inside2:
        return
    
    # For each vertex not on face boundary, split edge to bring it to boundary
    verts_to_process = []
    
    for v in [v1, v2]:
        if v not in face.verts:
            # Try to split an edge to bring this vertex to boundary
            _split_edge_for_interior_vertex(bm, face, v, merge_dist)
            
            # Check if vertex is now on face boundary
            if v in face.verts:
                verts_to_process.append(v)
        else:
            verts_to_process.append(v)
    
    # Now try to connect the vertices if both are on face boundary
    if len(verts_to_process) == 2:
        v1_final, v2_final = verts_to_process
        if v1_final.is_valid and v2_final.is_valid and v1_final != v2_final:
            # Check if they're on the same face
            faces1 = set(f for f in v1_final.link_faces if f.is_valid)
            faces2 = set(f for f in v2_final.link_faces if f.is_valid)
            shared = faces1 & faces2
            
            for f in shared:
                if f.is_valid and f == face:
                    # Check if edge already exists
                    edge_exists = False
                    for e in v1_final.link_edges:
                        if e.other_vert(v1_final) == v2_final:
                            edge_exists = True
                            break
                    
                    if not edge_exists:
                        try:
                            bmesh.ops.connect_verts(bm, verts=[v1_final, v2_final])
                        except Exception:
                            pass


def _auto_create_faces(bm, new_edges=None):
    """
    Auto-fill closed planar edge loops formed by new_edges and their
    connected naked (faceless) neighbours.  Never scans the entire mesh.
    Uses island detection so separate loops each get their own face.
    """
    if not new_edges:
        return

    naked_new = [e for e in new_edges if e.is_valid and len(e.link_faces) == 0]
    if not naked_new:
        return

    touched_verts = set()
    for e in naked_new:
        if e.is_valid:
            touched_verts.update(e.verts)

    candidate_edges = set(naked_new)
    for v in touched_verts:
        if v.is_valid:
            for e in v.link_edges:
                if e.is_valid and len(e.link_faces) == 0:
                    candidate_edges.add(e)

    if len(candidate_edges) < 3:
        return

    # Split into connected islands
    remaining = set(candidate_edges)
    islands = []
    while remaining:
        seed = remaining.pop()
        island = {seed}
        queue = [seed]
        while queue:
            cur = queue.pop()
            if not cur.is_valid:
                continue
            for v in cur.verts:
                if not v.is_valid:
                    continue
                for e in v.link_edges:
                    if e in remaining and e.is_valid:
                        remaining.discard(e)
                        island.add(e)
                        queue.append(e)
        islands.append(island)

    for island in islands:
        edges_list = [e for e in island if e.is_valid]
        if len(edges_list) < 3:
            continue
        all_verts = set()
        for e in edges_list:
            if e.is_valid:
                all_verts.update(e.verts)
        if len(all_verts) < 3:
            continue
        # Planarity check
        vlist = [v for v in all_verts if v.is_valid]
        if len(vlist) < 3:
            continue
        p0 = vlist[0].co
        normal = None
        for i in range(1, len(vlist)):
            v1 = vlist[i].co - p0
            for j in range(i + 1, len(vlist)):
                v2 = vlist[j].co - p0
                n = v1.cross(v2)
                if n.length > 1e-9:
                    normal = n.normalized()
                    break
            if normal:
                break
        if normal is None:
            continue
        coplanar = True
        for v in vlist:
            if abs((v.co - p0).dot(normal)) > 1e-4:
                coplanar = False
                break
        if not coplanar:
            continue
        try:
            bmesh.ops.contextual_create(bm, geom=edges_list)
        except Exception:
            pass



# =============================================================================
# FACE SPLIT UTILITY
# =============================================================================
def _point_in_face(face, co, tol=1e-4):
    """Check if a point is inside a face (including boundary)."""
    if not face or not face.is_valid:
        return False
        
    verts = face.verts
    n = face.normal
    
    if n.length < 1e-9:
        return False
    
    # First check if point is on face plane
    dist = abs((co - verts[0].co).dot(n))
    if dist > tol:
        return False
    
    # Check if point is inside face using winding number
    # For convex faces, we can use the half-plane test
    for i in range(len(verts)):
        v0 = verts[i].co
        v1 = verts[(i + 1) % len(verts)].co
        edge_vec = v1 - v0
        to_point = co - v0
        cross = edge_vec.cross(to_point)
        
        # Check if point is on the "inside" side of each edge
        if cross.dot(n) < -tol:
            return False
    
    return True


def _point_on_face_boundary(face, co, tol=1e-4):
    """Check if a point is on the boundary of a face."""
    if not face or not face.is_valid:
        return False
        
    # Check each edge
    for edge in face.edges:
        if not edge.is_valid:
            continue
            
        v1 = edge.verts[0].co
        v2 = edge.verts[1].co
        edge_vec = v2 - v1
        edge_len = edge_vec.length
        
        if edge_len < 1e-9:
            continue
            
        # Project point onto edge line
        t = (co - v1).dot(edge_vec) / (edge_len * edge_len)
        
        if -tol <= t <= 1 + tol:
            # Point is on edge line segment (with tolerance)
            proj = v1 + edge_vec * max(0.0, min(1.0, t))
            if (co - proj).length <= tol:
                return True
    
    return False


def _find_face_edge_intersection(face, p1, p2, tol=1e-6):
    """Find intersection point between line segment p1-p2 and face edges."""
    if not face or not face.is_valid:
        return None, None, None
    
    dir_vec = p2 - p1
    seg_len = dir_vec.length
    if seg_len < 1e-9:
        return None, None, None
    
    dir_normalized = dir_vec / seg_len
    
    best_edge = None
    best_t = 0.5
    best_u = 0.5
    best_dist = float('inf')
    
    for edge in face.edges:
        if not edge.is_valid:
            continue
            
        e1 = edge.verts[0].co
        e2 = edge.verts[1].co
        edge_vec = e2 - e1
        edge_len = edge_vec.length
        
        if edge_len < 1e-9:
            continue
            
        # Solve for intersection: p1 + t*dir = e1 + u*edge_vec
        # Using cross product method
        cross_de = dir_vec.cross(edge_vec)
        cross_len = cross_de.length
        
        if cross_len < 1e-12:
            # Lines are parallel
            continue
            
        diff = e1 - p1
        t = diff.cross(edge_vec).dot(cross_de) / (cross_len * cross_len)
        u = diff.cross(dir_vec).dot(cross_de) / (cross_len * cross_len)
        
        # Check if intersection is within both segments
        if -tol <= t <= 1 + tol and -tol <= u <= 1 + tol:
            # Calculate actual intersection point
            intersect = p1 + dir_vec * max(0.0, min(1.0, t))
            
            # Calculate distance from edge line
            proj_on_edge = e1 + edge_vec * max(0.0, min(1.0, u))
            dist = (intersect - proj_on_edge).length
            
            if dist < best_dist:
                best_dist = dist
                best_edge = edge
                best_t = max(0.01, min(0.99, u))  # Parameter on edge
                best_u = max(0.0, min(1.0, t))    # Parameter on segment
    
    if best_edge and best_edge.is_valid and best_dist < tol * 10:
        # Calculate intersection point on edge
        e1 = best_edge.verts[0].co
        e2 = best_edge.verts[1].co
        intersect_point = e1 + (e2 - e1) * best_t
        return best_edge, best_t, intersect_point
    
    return None, None, None


def _finalize_face_splits(bm, new_edges, merge_dist):
    """Finalize face splits by ensuring all new edges properly cut through faces."""
    if not new_edges:
        return
    
    # Process each new edge
    for edge in new_edges:
        if not edge.is_valid:
            continue
            
        v1, v2 = edge.verts
        if not v1.is_valid or not v2.is_valid:
            continue
            
        # Find faces shared by both vertices
        faces1 = set(f for f in v1.link_faces if f.is_valid)
        faces2 = set(f for f in v2.link_faces if f.is_valid)
        shared_faces = faces1 & faces2
        
        for face in shared_faces:
            if not face.is_valid:
                continue
                
            # Check if edge is already part of this face
            if edge in face.edges:
                continue
                
            # Check adjacency - skip if adjacent
            fverts = list(face.verts)
            try:
                i1 = fverts.index(v1)
                i2 = fverts.index(v2)
                n = len(fverts)
                if n < 4 or abs(i1 - i2) == 1 or abs(i1 - i2) == n - 1:
                    continue
            except ValueError:
                continue
            # Check if edge already exists between them
            edge_exists = any(e.other_vert(v1) == v2 for e in v1.link_edges if e.is_valid)
            if edge_exists:
                continue
            try:
                bmesh.utils.face_split(face, v1, v2)
            except Exception:
                try:
                    bmesh.ops.connect_verts(bm, verts=[v1, v2])
                except Exception:
                    pass
    
    # Clean up: remove any degenerate faces (faces with < 3 edges)
    faces_to_remove = []
    for face in bm.faces:
        if not face.is_valid:
            continue
            
        # Count valid edges
        valid_edges = 0
        for edge in face.edges:
            if edge.is_valid:
                valid_edges += 1
        
        if valid_edges < 3:
            faces_to_remove.append(face)
    
    for face in faces_to_remove:
        if face.is_valid:
            try:
                bm.faces.remove(face)
            except Exception:
                pass
    
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()


def _split_face_with_chain(bm, face, chain):
    """Split a face with a chain of vertices (for arcs)."""
    if not face or not face.is_valid or len(chain) < 2:
        return
        
    # Ensure all chain vertices are on face boundary
    for i, vert in enumerate(chain):
        if not vert.is_valid:
            return
            
        if vert not in face.verts:
            # Try to split an edge to bring this vertex to boundary
            _split_edge_for_interior_vertex(bm, face, vert, 0.001)
    
    # Now try to connect consecutive vertices in the chain
    for i in range(len(chain) - 1):
        v1 = chain[i]
        v2 = chain[i + 1]
        
        if not v1.is_valid or not v2.is_valid or v1 == v2:
            continue
            
        # Check if both vertices are on the same face
        faces1 = set(f for f in v1.link_faces if f.is_valid)
        faces2 = set(f for f in v2.link_faces if f.is_valid)
        shared_faces = faces1 & faces2
        
        for f in shared_faces:
            if f.is_valid and f == face:
                # Check if edge already exists
                edge_exists = False
                for e in v1.link_edges:
                    if e.other_vert(v1) == v2:
                        edge_exists = True
                        break
                
                if not edge_exists:
                    try:
                        bmesh.ops.connect_verts(bm, verts=[v1, v2])
                    except Exception:
                        pass


# =============================================================================
# ARC GEOMETRY HELPERS
# =============================================================================
def _get_plane_normal(plane):
    normals = {'XY': Vector((0, 0, 1)), 'XZ': Vector((0, 1, 0)), 'YZ': Vector((1, 0, 0))}
    return normals.get(plane, Vector((0, 0, 1)))


def _project_to_2d(point, normal):
    n = normal.normalized()
    if abs(n.z) > 0.9:
        ref = Vector((1, 0, 0))
    else:
        ref = Vector((0, 0, 1))
    u = n.cross(ref).normalized()
    v = n.cross(u).normalized()
    return point.dot(u), point.dot(v)


def _unproject_from_2d(x, y, ref_point, normal):
    n = normal.normalized()
    if abs(n.z) > 0.9:
        ref = Vector((1, 0, 0))
    else:
        ref = Vector((0, 0, 1))
    u = n.cross(ref).normalized()
    v = n.cross(u).normalized()
    offset = ref_point.dot(n)
    return u * x + v * y + n * offset


def _circle_from_3pts(p1, p2, p3, pn):
    ax, ay = _project_to_2d(p1, pn)
    bx, by = _project_to_2d(p2, pn)
    cx, cy = _project_to_2d(p3, pn)
    D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-12:
        return None, None
    ux = ((ax*ax+ay*ay)*(by-cy)+(bx*bx+by*by)*(cy-ay)+(cx*cx+cy*cy)*(ay-by)) / D
    uy = ((ax*ax+ay*ay)*(cx-bx)+(bx*bx+by*by)*(ax-cx)+(cx*cx+cy*cy)*(bx-ax)) / D
    center = _unproject_from_2d(ux, uy, p1, pn)
    radius = (center - p1).length
    return center, radius


def _angle_on_plane(point, center, pn):
    n = pn.normalized()
    if abs(n.z) > 0.9:
        ref = Vector((1, 0, 0))
    else:
        ref = Vector((0, 0, 1))
    u = n.cross(ref).normalized()
    v = n.cross(u).normalized()
    d = point - center
    return math.atan2(d.dot(v), d.dot(u))


def _gen_arc_pts(center, radius, a_start, a_end, segs, pn):
    n = pn.normalized()
    if abs(n.z) > 0.9:
        ref = Vector((1, 0, 0))
    else:
        ref = Vector((0, 0, 1))
    u = n.cross(ref).normalized()
    v = n.cross(u).normalized()
    pts = []
    for i in range(segs + 1):
        t = i / segs
        a = a_start + t * (a_end - a_start)
        pts.append(center + u * radius * math.cos(a) + v * radius * math.sin(a))
    return pts


def _order_arc_angles(a_s, a_through, a_e):
    def _norm(a):
        while a < 0: a += 2*math.pi
        while a >= 2*math.pi: a -= 2*math.pi
        return a
    s = _norm(a_s); t = _norm(a_through); e = _norm(a_e)
    def _ccw(s, t, e):
        if s <= e: return s <= t <= e
        return t >= s or t <= e
    if _ccw(s, t, e):
        if e < s: return s, e + 2*math.pi
        return s, e
    else:
        if s < e: return e, s + 2*math.pi
        return e, s


def _tangent_arc_pts(v_shared, v1, v2, radius, pn, segs):
    """Compute tangent arc points between two edges meeting at v_shared."""
    len1 = (v1 - v_shared).length
    len2 = (v2 - v_shared).length
    if len1 < 1e-9 or len2 < 1e-9:
        return [], None, None
    d1 = (v1 - v_shared).normalized()
    d2 = (v2 - v_shared).normalized()
    dot = max(-1.0, min(1.0, d1.dot(d2)))
    if abs(dot) > 0.9999:
        return [], None, None
    half = math.acos(dot) * 0.5
    sin_half = math.sin(half)
    if abs(sin_half) < 1e-9:
        return [], None, None
    tan_len = radius / math.tan(half)
    if tan_len > len1 * 0.99 or tan_len > len2 * 0.99:
        max_r = min(len1, len2) * 0.99 * math.tan(half)
        if max_r < 1e-9:
            return [], None, None
        radius = max_r
        tan_len = radius / math.tan(half)
    pt1 = v_shared + d1 * tan_len
    pt2 = v_shared + d2 * tan_len
    bisector = (d1 + d2)
    if bisector.length < 1e-9:
        perp = pn.cross(d1)
        if perp.length < 1e-9:
            return [], None, None
        bisector = perp.normalized()
    else:
        bisector.normalize()
    cdist = radius / sin_half
    center = v_shared + bisector * cdist
    if (center - pt1).length - radius > radius * 0.1:
        center = v_shared - bisector * cdist
    a1 = _angle_on_plane(pt1, center, pn)
    a2 = _angle_on_plane(pt2, center, pn)
    diff = a2 - a1
    if diff > math.pi: diff -= 2*math.pi
    elif diff < -math.pi: diff += 2*math.pi
    if abs(diff) > math.pi:
        diff = diff - 2*math.pi if diff > 0 else diff + 2*math.pi
    pts = _gen_arc_pts(center, radius, a1, a1+diff, segs, pn)
    return pts, pt1, pt2


# =============================================================================
# SAFE OBJECT LINKING
# =============================================================================
def safe_link_object(context, obj):
    try:
        active_col = context.view_layer.active_layer_collection.collection
        active_col.objects.link(obj)
        return True
    except Exception:
        try:
            context.scene.collection.objects.link(obj)
            return True
        except Exception as e:
            print(f"[SU] Failed to link object '{obj.name}': {e}")
            return False


# =============================================================================
# KDTREE CACHE
# =============================================================================
class VertexKDTreeCache:
    def __init__(self):
        self.kd = None
        self.meta = None
        self.depsgraph = None
        self.last_update = 0
        self.cache_time = KDTREE_CACHE_TIME

    def get_or_build(self, context, force_rebuild=False):
        current_time = time.time()
        if force_rebuild or (current_time - self.last_update) > self.cache_time:
            self.kd, self.meta, self.depsgraph = build_vertex_kdtree(context)
            self.last_update = current_time
        return self.kd, self.meta, self.depsgraph

    def invalidate(self):
        self.last_update = 0


_kdtree_cache = VertexKDTreeCache()


# =============================================================================
# PREFERENCES
# =============================================================================
class SU_Preferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    default_plane: bpy.props.EnumProperty(
        name="Default Drawing Plane",
        items=[("XY", "XY", ""), ("XZ", "XZ", ""), ("YZ", "YZ", "")],
        default="XY",
    )
    default_snap: bpy.props.BoolProperty(name="Vertex Snapping Enabled", default=True)
    snap_pixel_radius: bpy.props.IntProperty(name="Snap Pixel Radius", default=12, min=2, max=64)
    enable_hotkeys: bpy.props.BoolProperty(name="Enable Hotkeys (L/R/P/A)", default=True)
    hotkeys_shift: bpy.props.BoolProperty(name="Use Shift+L/R/P/A", default=False)
    show_visual_feedback: bpy.props.BoolProperty(name="Show Visual Feedback", default=True)
    grid_snap_enabled: bpy.props.BoolProperty(name="Grid Snapping", default=False)
    grid_size: bpy.props.FloatProperty(name="Grid Size", default=0.1, min=0.001, max=10.0)
    default_individual_faces: bpy.props.BoolProperty(name="Default Individual Faces", default=False)

    # Merge / Weld
    merge_distance: bpy.props.FloatProperty(
        name="Weld Distance", default=MERGE_DISTANCE,
        min=0.0001, max=1.0, precision=4,
        description="Auto-weld vertices closer than this distance (SketchUp behaviour)",
    )
    auto_face: bpy.props.BoolProperty(
        name="Auto Create Faces",
        default=True,
        description="Automatically fill closed edge loops with faces",
    )
    continuous_draw: bpy.props.BoolProperty(
        name="Continuous Drawing",
        default=True,
        description="After finishing a line, start a new one from the endpoint (SketchUp style)",
    )
    keep_axis_lock_in_continuous: bpy.props.BoolProperty(
        name="Keep Axis Lock in Continuous",
        default=True,
        description="Preserve axis lock between continuous line segments unless explicitly unlocked",
    )

    # Snap marker
    show_snap_marker: bpy.props.BoolProperty(name="Show Snap Marker", default=True)
    snap_marker_size: bpy.props.IntProperty(name="Snap Marker Size", default=8, min=4, max=32)

    # Axis guides
    show_axis_guides: bpy.props.BoolProperty(name="Show Axis Guides", default=True)
    # VCB
    show_vcb: bpy.props.BoolProperty(
        name="Show Value Control Box",
        default=True,
        description="Display SketchUp-style value input box at bottom-left",
    )

    axis_guide_length_mode: bpy.props.EnumProperty(
        name="Axis Guide Length",
        items=[("WORLD", "World Space", ""), ("SCREEN", "Screen Space", "")],
        default="SCREEN",
    )
    axis_guide_world_length: bpy.props.FloatProperty(name="World Length", default=5.0, min=0.1, max=100.0)
    axis_guide_screen_length: bpy.props.IntProperty(name="Screen Length", default=80, min=20, max=300)

    # Arc
    arc_segments: bpy.props.IntProperty(
        name="Arc Segments", default=DEFAULT_ARC_SEGMENTS,
        min=MIN_ARC_SEGMENTS, max=MAX_ARC_SEGMENTS,
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="SketchUp Tools V5.0 – Preferences", icon='SETTINGS')

        box = layout.box()
        box.label(text="SketchUp Behaviour", icon='MESH_DATA')
        col = box.column(align=True)
        col.prop(self, "merge_distance")
        col.prop(self, "auto_face")
        col.prop(self, "continuous_draw")
        col.prop(self, "keep_axis_lock_in_continuous")

        box = layout.box()
        box.label(text="Drawing Settings", icon='GREASEPENCIL')
        col = box.column(align=True)
        col.prop(self, "default_plane")
        col.prop(self, "default_snap")
        col.prop(self, "snap_pixel_radius")

        box = layout.box()
        box.label(text="Grid Settings", icon='GRID')
        col = box.column(align=True)
        col.prop(self, "grid_snap_enabled")
        sub = col.row()
        sub.enabled = self.grid_snap_enabled
        sub.prop(self, "grid_size")

        box = layout.box()
        box.label(text="Push/Pull Settings", icon='MOD_SOLIDIFY')
        box.prop(self, "default_individual_faces")

        box = layout.box()
        box.label(text="Visual Feedback", icon='HIDE_OFF')
        if not GPU_AVAILABLE:
            box.label(text="GPU module unavailable", icon='ERROR')
        col = box.column(align=True)
        col.prop(self, "show_visual_feedback")
        col.prop(self, "show_vcb")
        col.separator()
        col.prop(self, "show_snap_marker")
        sub = col.row()
        sub.enabled = self.show_snap_marker
        sub.prop(self, "snap_marker_size")
        col.separator()
        col.prop(self, "show_axis_guides")
        if self.show_axis_guides:
            sub = col.column(align=True)
            sub.prop(self, "axis_guide_length_mode", text="Mode")
            if self.axis_guide_length_mode == "WORLD":
                sub.prop(self, "axis_guide_world_length")
            else:
                sub.prop(self, "axis_guide_screen_length")

        box = layout.box()
        box.label(text="Arc Tool Settings", icon='MESH_CIRCLE')
        box.prop(self, "arc_segments")

        box = layout.box()
        box.label(text="Keyboard Shortcuts", icon='KEYINGSET')
        col = box.column(align=True)
        col.prop(self, "enable_hotkeys")
        sub = col.row()
        sub.enabled = self.enable_hotkeys
        sub.prop(self, "hotkeys_shift")


# Error panel
class SU_PT_error_panel(bpy.types.Panel):
    bl_label = "SketchUp Tools - Error Report"
    bl_idname = "SU_PT_error_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SketchUp"

    @classmethod
    def poll(cls, context):
        return bool(_IMPORT_ERROR_TEXT)

    def draw(self, context):
        layout = self.layout
        layout.alert = True
        layout.label(text="Add-on Error Detected:", icon='ERROR')
        box = layout.box()
        for line in _IMPORT_ERROR_TEXT.splitlines()[:60]:
            box.label(text=line[:220])


# =============================================================================
# SCENE PROPERTIES
# =============================================================================
def _register_scene_props():
    bpy.types.Scene.su_draw_plane = bpy.props.EnumProperty(
        name="Drawing Plane",
        items=[("XY", "XY", ""), ("XZ", "XZ", ""), ("YZ", "YZ", "")],
        default="XY",
    )
    bpy.types.Scene.su_plane_offset = bpy.props.FloatProperty(
        name="Plane Offset", default=0.0, precision=4, step=10,
    )
    bpy.types.Scene.su_enable_vertex_snap = bpy.props.BoolProperty(
        name="Vertex Snapping", default=True,
    )
    bpy.types.Scene.su_snap_px = bpy.props.IntProperty(
        name="Snap Radius (px)", default=12, min=2, max=64,
    )
    bpy.types.Scene.su_pushpull_individual = bpy.props.BoolProperty(
        name="Individual Faces", default=False,
    )
    bpy.types.Scene.su_prefs_applied = bpy.props.BoolProperty(
        name="Preferences Applied", default=False, options={'HIDDEN'},
    )
    bpy.types.Scene.su_arc_mode = bpy.props.EnumProperty(
        name="Arc Mode",
        items=[
            ("TWO_POINT", "Two-Point Arc", "Start, End, Bulge"),
            ("THREE_POINT", "Three-Point Arc", "Start, End, Third point on arc"),
            ("CENTER", "Center-Point Arc", "Center, Radius start, End angle"),
            ("TANGENT", "Tangent Arc", "Fillet between two edges"),
            ("EDGE_CONNECT", "Edge-Connect Arc", "Arc between two vertices on face"),
            ("FILLET", "Auto-Fillet Arc", "Replace corner with arc radius"),
        ],
        default="TWO_POINT",
    )


def _unregister_scene_props():
    props = [
        "su_draw_plane", "su_plane_offset", "su_enable_vertex_snap",
        "su_snap_px", "su_pushpull_individual", "su_prefs_applied", "su_arc_mode",
    ]
    for prop_name in props:
        try:
            delattr(bpy.types.Scene, prop_name)
        except Exception:
            pass


# =============================================================================
# PREFERENCES APPLICATION TIMER
# =============================================================================
def _apply_defaults_timer():
    try:
        ctx = bpy.context
        scn = getattr(ctx, "scene", None)
        if scn is None:
            return 0.25

        addon = None
        prefs = None
        try:
            addon = ctx.preferences.addons.get(ADDON_ID)
            prefs = addon.preferences if addon else None
        except Exception:
            addon = None
            prefs = None

        # Apply once, then stop the timer. If running as a script (not installed add-on),
        # fall back to sensible defaults without looping forever.
        if not getattr(scn, "su_prefs_applied", False):
            if prefs is not None:
                scn.su_draw_plane = prefs.default_plane
                scn.su_enable_vertex_snap = bool(prefs.default_snap)
                scn.su_snap_px = int(prefs.snap_pixel_radius)
                scn.su_pushpull_individual = bool(prefs.default_individual_faces)
            else:
                # Fallbacks if preferences are unavailable (e.g., run from Text Editor)
                if hasattr(scn, "su_draw_plane"):
                    scn.su_draw_plane = getattr(scn, "su_draw_plane", "XY") or "XY"
                if hasattr(scn, "su_enable_vertex_snap"):
                    scn.su_enable_vertex_snap = bool(getattr(scn, "su_enable_vertex_snap", True))
                if hasattr(scn, "su_snap_px"):
                    scn.su_snap_px = int(getattr(scn, "su_snap_px", 12))
                if hasattr(scn, "su_pushpull_individual"):
                    scn.su_pushpull_individual = bool(getattr(scn, "su_pushpull_individual", False))
            scn.su_prefs_applied = True
        else:
            if prefs is not None and hasattr(scn, "su_snap_px"):
                scn.su_snap_px = int(prefs.snap_pixel_radius)
        return None
    except Exception:
        return None

def _schedule_apply_defaults():
    global _timer_scheduled
    if _timer_scheduled:
        return
    _timer_scheduled = True
    try:
        bpy.app.timers.register(_apply_defaults_timer, first_interval=0.1)
    except Exception:
        pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def _set_header(context, text):
    if context.area:
        context.area.header_text_set(text)


def _clear_header(context):
    if context.area:
        context.area.header_text_set(None)


def _parse_length(context, s: str):
    """Parse length with unit suffixes. Plain number = mm by default (converted to Blender units = meters)."""
    s = (s or "").strip()
    if not s or len(s) > MAX_INPUT_LENGTH:
        return None
    sl = s.lower().strip()
    manual_units = {
        'mm': 0.001, 'cm': 0.01, 'm': 1.0,
        'in': 0.0254, '"': 0.0254,
        'ft': 0.3048, "'": 0.3048,
    }
    for suffix, factor in sorted(manual_units.items(), key=lambda x: -len(x[0])):
        if sl.endswith(suffix):
            num_part = sl[:-len(suffix)].strip()
            try:
                val = float(num_part) * factor
                if abs(val) > MAX_VALUE_MAGNITUDE or abs(val) < 1e-9:
                    return None
                return val
            except (ValueError, TypeError):
                return None
    try:
        unit_system = context.scene.unit_settings.system
        v = bpy.utils.units.to_value(unit_system, 'LENGTH', s)
        if abs(v) > MAX_VALUE_MAGNITUDE or abs(v) < 1e-9:
            return None
        return float(v)
    except Exception:
        try:
            val = float(s)
            val_m = val * 0.001
            if abs(val_m) > MAX_VALUE_MAGNITUDE or abs(val_m) < 1e-9:
                return None
            return val_m
        except Exception:
            return None


def _parse_rect_two(context, text: str):
    t = (text or "").strip()
    # Replace semicolons with commas
    t = t.replace(";", ",")
    # Replace spaces with commas if no comma present
    if "," not in t:
        t = t.replace(" ", ",")
    if "," not in t:
        return None, None
    parts = t.split(",", 1)
    if len(parts) != 2:
        return None, None
    w = _parse_length(context, parts[0])
    h = _parse_length(context, parts[1])
    return w, h


def _parse_arc_input(context, text: str):
    """Parse arc input: radius, angle, or radius=value angle=value format."""
    import re
    t = (text or "").strip().lower()
    
    # Try to parse as simple radius
    radius = _parse_length(context, t)
    if radius is not None:
        return radius, None
    
    # Try to parse angle formats
    angle_patterns = [
        (r'(\d+(?:\.\d+)?)\s*deg(?:rees?)?', 1),
        (r'(\d+(?:\.\d+)?)\s*°', 1),
        (r'(\d+(?:\.\d+)?)\s*rad(?:ians?)?', 1),
    ]
    
    for pattern, group in angle_patterns:
        match = re.match(pattern, t)
        if match:
            try:
                angle_value = float(match.group(group))
                if 'deg' in pattern or '°' in pattern:
                    angle_value = math.radians(angle_value)
                return None, angle_value
            except (ValueError, IndexError):
                pass
    
    # Try to parse R=value A=value format
    r_match = re.search(r'r\s*=\s*(\d+(?:\.\d+)?)', t)
    a_match = re.search(r'a\s*=\s*(\d+(?:\.\d+)?)', t)
    
    radius_val = None
    angle_val = None
    
    if r_match:
        try:
            radius_val = _parse_length(context, r_match.group(1))
        except (ValueError, IndexError):
            pass
    
    if a_match:
        try:
            angle_deg = float(a_match.group(1))
            angle_val = math.radians(angle_deg)
        except (ValueError, IndexError):
            pass
    
    return radius_val, angle_val


def _format_length(context, value: float):
    """Format a length for VCB display. Shows mm by default."""
    try:
        us = context.scene.unit_settings
        if us.system and us.system != 'NONE':
            return bpy.utils.units.to_string(us.system, 'LENGTH', float(value), precision=4)
    except Exception:
        pass
    try:
        mm_val = float(value) * 1000.0
        if abs(mm_val) >= 1000.0:
            return f"{float(value):.4f} m"
        elif abs(mm_val) >= 10.0:
            return f"{float(value)*100:.2f} cm"
        else:
            return f"{mm_val:.2f} mm"
    except Exception:
        return ""


def _parse_angle(text: str):
    """Parse an angle string (deg) for arc VCB. Accepts: 30, 30deg, 30°, 30d."""
    t = (text or "").strip().lower()
    if not t:
        return None
    t = t.replace("°", "").replace("deg", "").replace("d", "").strip()
    try:
        v = float(t)
        if abs(v) < 1e-9 or abs(v) > 360000.0:
            return None
        return math.radians(v)
    except Exception:
        return None



def snap_to_grid(point, grid_size=0.1):
    if grid_size <= 0:
        return point
    return Vector((
        round(point.x / grid_size) * grid_size,
        round(point.y / grid_size) * grid_size,
        round(point.z / grid_size) * grid_size,
    ))


def _get_prefs(context):
    addon = context.preferences.addons.get(ADDON_ID)
    return addon.preferences if addon else None


# =============================================================================
# PLANE & RAY CASTING
# =============================================================================
def get_plane_hit(context, event, plane='XY', offset=0.0):
    region = context.region
    rv3d = context.region_data
    if not region or not rv3d:
        return None
    coord = (event.mouse_region_x, event.mouse_region_y)
    try:
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
    except Exception:
        return None
    normals = {'XY': Vector((0, 0, 1)), 'XZ': Vector((0, 1, 0)), 'YZ': Vector((1, 0, 0))}
    n = normals.get(plane, Vector((0, 0, 1)))
    d = float(offset)
    denom = n.dot(direction)
    if abs(denom) < 1e-9:
        return None
    t = (d - n.dot(origin)) / denom
    if t < 0.0:
        return None
    return origin + direction * t


# =============================================================================
# KDTREE & VERTEX SNAPPING
# =============================================================================
def build_vertex_kdtree(context):
    depsgraph = context.evaluated_depsgraph_get()
    verts = []
    target_name = _get_target_name(context)
    # Build ordered list: target mesh first, then active, then others
    target_objs = []
    other_objs = []
    for obj in context.view_layer.objects:
        if obj.type != 'MESH' or not obj.visible_get():
            continue
        if obj.name.startswith("SU_TMP"):
            continue
        if target_name and obj.name == target_name:
            target_objs.insert(0, obj)
        else:
            other_objs.append(obj)
    ordered = target_objs + other_objs
    for obj in ordered:
        try:
            obj_eval = obj.evaluated_get(depsgraph)
            try:
                me_eval = obj_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
            except TypeError:
                try:
                    me_eval = obj_eval.to_mesh(preserve_all_data_layers=False)
                except TypeError:
                    me_eval = obj_eval.to_mesh()
            if not me_eval:
                continue
            mw = obj_eval.matrix_world.copy()
            for i, v in enumerate(me_eval.vertices):
                verts.append((obj.name, i, mw @ v.co))
            obj_eval.to_mesh_clear()
        except Exception:
            continue
    kd = KDTree(len(verts)) if verts else None
    meta = []
    if kd:
        for idx, (oname, vidx, co) in enumerate(verts):
            kd.insert(co, idx)
            meta.append((oname, vidx, co))
        kd.balance()
    return kd, meta, depsgraph


def find_vertex_snap(context, event, hit_point, kd, meta, depsgraph,
                     max_px=12.0, max_world=50.0, target_name=None):
    """Target-aware snap: prioritise target mesh verts over others."""
    if kd is None or not meta or hit_point is None:
        return None, None
    region = context.region
    rv3d = context.region_data
    if not region or not rv3d:
        return None, None
    # Find nearest candidates
    try:
        results = kd.find_n(hit_point, 10)
    except Exception:
        return None, None

    best_target = None
    best_other = None

    for co, idx, dist in results:
        if idx is None or dist > max_world:
            continue
        oname, vidx, wco = meta[idx]
        try:
            p2d = view3d_utils.location_3d_to_region_2d(region, rv3d, wco)
        except Exception:
            continue
        if p2d is None:
            continue
        dx = p2d.x - event.mouse_region_x
        dy = p2d.y - event.mouse_region_y
        px_dist = (dx * dx + dy * dy) ** 0.5
        if px_dist > max_px:
            continue
        snap_info = {
            "object": oname,
            "vert_index": int(vidx),
            "px_dist": float(px_dist),
            "world_dist": float(dist),
            "screen_pos": p2d,
        }
        # Priority: target mesh first
        if target_name and oname == target_name:
            if best_target is None or px_dist < best_target[1]["px_dist"]:
                best_target = (wco.copy(), snap_info)
        else:
            if best_other is None or px_dist < best_other[1]["px_dist"]:
                best_other = (wco.copy(), snap_info)

    # Return target mesh snap immediately if found
    if best_target is not None:
        return best_target
    if best_other is not None:
        return best_other
    return None, None


# =============================================================================
# PLANE & AXIS HELPERS
# =============================================================================
def _plane_axes(plane: str):
    if plane == "XY":
        return Vector((1, 0, 0)), Vector((0, 1, 0))
    elif plane == "XZ":
        return Vector((1, 0, 0)), Vector((0, 0, 1))
    elif plane == "YZ":
        return Vector((0, 1, 0)), Vector((0, 0, 1))
    return Vector((1, 0, 0)), Vector((0, 1, 0))


def _detect_axis_key(event):
    if event.value != 'PRESS':
        return None
    if event.type in {'X', 'Y', 'Z'}:
        return event.type
    if hasattr(event, 'unicode') and event.unicode:
        char = event.unicode.upper()
        if char in {'X', 'Y', 'Z'}:
            return char
    return None


def _infer_dominant_axis(start, end, plane, axis_lock):
    if axis_lock and axis_lock != "FREE":
        return axis_lock
    delta = end - start
    if delta.length < 1e-6:
        return "FREE"
    plane_axes = {
        "XY": [("X", abs(delta.x)), ("Y", abs(delta.y))],
        "XZ": [("X", abs(delta.x)), ("Z", abs(delta.z))],
        "YZ": [("Y", abs(delta.y)), ("Z", abs(delta.z))],
    }
    axes = plane_axes.get(plane, [("X", abs(delta.x)), ("Y", abs(delta.y))])
    dominant = max(axes, key=lambda x: x[1])
    return dominant[0] if dominant[1] >= 1e-6 else "FREE"


def _get_axis_color(axis, locked=False):
    base = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.5, 1.0)}
    color = base.get(axis, (0.7, 0.7, 0.7))
    alpha = 1.0 if locked else 0.8
    return (*color, alpha)


def constrain_to_plane(point, start, plane):
    result = point.copy()
    if plane == "XY":
        result.z = start.z
    elif plane == "XZ":
        result.y = start.y
    elif plane == "YZ":
        result.x = start.x
    return result


def constrain_to_axis(point, start, axis_name, plane):
    if not axis_name or axis_name == "FREE":
        return constrain_to_plane(point, start, plane)
    axis_vectors = {"X": Vector((1, 0, 0)), "Y": Vector((0, 1, 0)), "Z": Vector((0, 0, 1))}
    axis = axis_vectors.get(axis_name)
    if not axis:
        return constrain_to_plane(point, start, plane)
    delta = point - start
    projection_length = delta.dot(axis)
    projected_point = start + axis * projection_length
    return constrain_to_plane(projected_point, start, plane)


def _compute_perpendicular_axis_movement(context, event, start, axis_name, last_mouse_pos=None):
    region = context.region
    rv3d = context.region_data
    if not region or not rv3d:
        return start.copy()
    axis_vectors = {"X": Vector((1, 0, 0)), "Y": Vector((0, 1, 0)), "Z": Vector((0, 0, 1))}
    axis = axis_vectors.get(axis_name)
    if not axis:
        return start.copy()
    try:
        start_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, start)
        if not start_2d:
            mx, my = event.mouse_region_x, event.mouse_region_y
            projected = view3d_utils.region_2d_to_location_3d(region, rv3d, (mx, my), start)
            return start + axis * (projected - start).dot(axis)
        test_point = start + axis * 1.0
        test_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, test_point)
        if not test_2d:
            mx, my = event.mouse_region_x, event.mouse_region_y
            projected = view3d_utils.region_2d_to_location_3d(region, rv3d, (mx, my), start)
            return start + axis * (projected - start).dot(axis)
        screen_axis = test_2d - start_2d
        if screen_axis.length < 1e-6:
            mx, my = event.mouse_region_x, event.mouse_region_y
            projected = view3d_utils.region_2d_to_location_3d(region, rv3d, (mx, my), start)
            return start + axis * (projected - start).dot(axis)
        screen_axis.normalize()
        mouse_pos = Vector((event.mouse_region_x, event.mouse_region_y))
        mouse_delta = mouse_pos - (last_mouse_pos if last_mouse_pos else start_2d)
        screen_distance = mouse_delta.dot(screen_axis)
        screen_length_per_unit = (test_2d - start_2d).length
        world_distance = screen_distance / screen_length_per_unit if screen_length_per_unit > 1e-6 else 0.0
        return start + axis * world_distance
    except Exception:
        try:
            mx, my = event.mouse_region_x, event.mouse_region_y
            projected = view3d_utils.region_2d_to_location_3d(region, rv3d, (mx, my), start)
            return start + axis * (projected - start).dot(axis)
        except Exception:
            return start.copy()


def _apply_axis_lock(start, cur, plane, axis_lock):
    return constrain_to_axis(cur, start, axis_lock, plane)


def _auto_axis_from_shift(start, cur, plane):
    delta = cur - start
    ax_a, ax_b = _plane_axes(plane)
    da = abs(delta.dot(ax_a))
    db = abs(delta.dot(ax_b))
    if da >= db:
        return "X" if plane in ("XY", "XZ") else "Y"
    else:
        return "Y" if plane == "XY" else "Z"


def _handle_plane_keys(context, event):
    scn = context.scene
    if event.type == "ONE" and event.value == "PRESS":
        scn.su_draw_plane = "XY"
        return True
    elif event.type == "TWO" and event.value == "PRESS":
        scn.su_draw_plane = "XZ"
        return True
    elif event.type == "THREE" and event.value == "PRESS":
        scn.su_draw_plane = "YZ"
        return True
    return False


def _format_measurement(value_m, unit_system='METRIC'):
    """Format measurement value for display with appropriate units."""
    if value_m is None:
        return ""
    
    if unit_system == 'METRIC':
        if abs(value_m) < 0.001:  # Less than 1mm
            return f"{value_m*1000:.1f}mm"
        elif abs(value_m) < 0.01:  # Less than 1cm
            return f"{value_m*1000:.0f}mm"
        elif abs(value_m) < 1.0:  # Less than 1m
            return f"{value_m*100:.1f}cm"
        else:  # 1m or more
            return f"{value_m:.2f}m"
    else:  # IMPERIAL
        inches = value_m / 0.0254
        if inches < 12:  # Less than 1 foot
            return f"{inches:.1f}\""
        else:  # Feet and inches
            feet = int(inches // 12)
            inch_remainder = inches % 12
            return f"{feet}'{inch_remainder:.1f}\""


def _header_status(tool_name, plane, axis_lock, typing, snap_info, measurement=""):
    axis = axis_lock if axis_lock and axis_lock != "FREE" else "FREE"
    snap = " [SNAP]" if snap_info else ""
    t = f" | Type: {typing}" if typing else ""
    measure = f" | {measurement}" if measurement else ""
    return f"{tool_name} | Plane: {plane} | Axis: {axis}{snap}{measure}{t} | Esc/RMB: Cancel"


# =============================================================================
# VCB DRAWING FUNCTION
# =============================================================================
def draw_vcb_box():
    """Draw the Value Control Box (VCB) at bottom-left like SketchUp."""
    if not GPU_AVAILABLE or not BLF_AVAILABLE:
        return
    if not _vcb_state.get('visible', False):
        return
    
    try:
        import gpu
        from gpu_extras.batch import batch_for_shader
        
        # Get VCB state
        label = _vcb_state.get('label', '')
        value = _vcb_state.get('value', '')
        typed = _vcb_state.get('typed', '')
        
        # Build display text
        display_text = f"{label}: "
        if typed:
            display_text += typed
        elif value:
            display_text += value
        
        if not display_text:
            return
        
        # Get font and calculate text dimensions
        font_id = 0  # Default Blender font
        blf.size(font_id, 18)  # Fixed for Blender 4.5: no DPI arg
        text_width = 0
        text_height = 0
        
        # Calculate text dimensions using blf
        for line in display_text.split('\n'):
            dims = blf.dimensions(font_id, line)
            text_width = max(text_width, dims[0])
            text_height += dims[1] * 1.2  # Add line spacing
        
        # Box dimensions with padding
        padding = 12  # Increased padding for larger font
        box_width = text_width + padding * 2
        box_height = text_height + padding * 2
        
        # Box position (bottom-left) - Fixed coordinates
        x = 30
        y = 60
        
        # Draw background box
        vertices = [
            (x, y),
            (x + box_width, y),
            (x + box_width, y + box_height),
            (x, y + box_height),
        ]
        indices = [(0, 1, 2), (0, 2, 3)]
        
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)
        shader.bind()
        shader.uniform_float("color", VCB_BG_COLOR)
        gpu.state.blend_set('ALPHA')
        batch.draw(shader)
        
        # Draw text (single line, truncate if too long)
        blf.position(font_id, x + padding, y + padding + 6, 0)  # Adjusted for larger font
        blf.color(font_id, *VCB_TEXT_COLOR)
        blf.draw(font_id, display_text)
        
        gpu.state.blend_set('NONE')
        
    except Exception as e:
        global _vcb_last_err_t
        t = time.time()
        if t - _vcb_last_err_t > 2.0:
            print(f"[SU] VCB draw error: {e}")
            _vcb_last_err_t = t


# =============================================================================
# GPU DRAW HANDLERS
# =============================================================================
def register_draw_handler(key, handler, args, region_type='WINDOW', space='POST_PIXEL'):
    global _draw_handlers
    if not GPU_AVAILABLE:
        return
    try:
        if key in _draw_handlers:
            unregister_draw_handler(key)
        handle = bpy.types.SpaceView3D.draw_handler_add(handler, args, region_type, space)
        _draw_handlers[key] = (handle, region_type, space)
    except Exception as e:
        print(f"[SU] Warning: Failed to register draw handler: {e}")


def unregister_draw_handler(key):
    global _draw_handlers
    if not GPU_AVAILABLE:
        return
    try:
        if key in _draw_handlers:
            data = _draw_handlers[key]
            if isinstance(data, tuple):
                if len(data) == 2:
                    h1, h2 = data
                    bpy.types.SpaceView3D.draw_handler_remove(h1, 'WINDOW')
                    bpy.types.SpaceView3D.draw_handler_remove(h2, 'WINDOW')
                elif len(data) == 3:
                    if isinstance(data[1], str):
                        handle, region_type, space = data
                        bpy.types.SpaceView3D.draw_handler_remove(handle, region_type)
                    else:
                        h1, h2, h3 = data
                        bpy.types.SpaceView3D.draw_handler_remove(h1, 'WINDOW')
                        bpy.types.SpaceView3D.draw_handler_remove(h2, 'WINDOW')
                        bpy.types.SpaceView3D.draw_handler_remove(h3, 'WINDOW')
                elif len(data) == 4:
                    h1, h2, h3, h4 = data
                    bpy.types.SpaceView3D.draw_handler_remove(h1, 'WINDOW')
                    bpy.types.SpaceView3D.draw_handler_remove(h2, 'WINDOW')
                    bpy.types.SpaceView3D.draw_handler_remove(h3, 'WINDOW')
                    bpy.types.SpaceView3D.draw_handler_remove(h4, 'WINDOW')
                else:
                    bpy.types.SpaceView3D.draw_handler_remove(data, 'WINDOW')
            else:
                bpy.types.SpaceView3D.draw_handler_remove(data, 'WINDOW')
            del _draw_handlers[key]
    except Exception as e:
        print(f"[SU] Warning: Failed to unregister draw handler: {e}")


# ---- Drawing functions (GPU overlay only – no temp objects) ----

def draw_snap_marker_simple(snap_info, prefs):
    if not snap_info or not prefs or not prefs.show_snap_marker:
        return
    try:
        screen_pos = snap_info.get("screen_pos")
        if not screen_pos:
            return
        size = prefs.snap_marker_size
        x, y = screen_pos.x, screen_pos.y
        segments = 16
        vertices = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            vertices.append((x + size * math.cos(angle), y + size * math.sin(angle)))
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": vertices})
        shader.bind()
        shader.uniform_float("color", SNAP_COLOR)
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)
        batch.draw(shader)
        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)
    except Exception:
        pass


def draw_axis_guides_simple(context, anchor, plane, axis_lock, prefs):
    if not prefs or not prefs.show_axis_guides or not anchor or not context:
        return
    try:
        region = context.region
        rv3d = context.region_data
        if not region or not rv3d:
            return
        anchor_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, anchor)
        if not anchor_2d:
            return
        axes = [
            ("X", Vector((1, 0, 0)), AXIS_X_COLOR),
            ("Y", Vector((0, 1, 0)), AXIS_Y_COLOR),
            ("Z", Vector((0, 0, 1)), AXIS_Z_COLOR),
        ]
        vertices = []
        colors = []
        for axis_name, axis_vec, axis_color in axes:
            if axis_lock == axis_name:
                color = AXIS_LOCKED_COLOR
            elif axis_lock != "FREE" and axis_lock != axis_name:
                color = (*axis_color[:3], AXIS_DIM_ALPHA)
            else:
                color = axis_color
            if prefs.axis_guide_length_mode == "SCREEN":
                p1_world = anchor + axis_vec
                p1_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, p1_world)
                if not p1_2d:
                    continue
                screen_dir = p1_2d - anchor_2d
                screen_len = screen_dir.length
                if screen_len < 1e-3:
                    continue
                screen_dir_norm = screen_dir / screen_len
                pixel_length = prefs.axis_guide_screen_length
                p_end_2d = anchor_2d + screen_dir_norm * pixel_length
                p_end_world = view3d_utils.region_2d_to_location_3d(region, rv3d, p_end_2d, anchor)
                if not p_end_world:
                    continue
                vertices.extend([anchor, p_end_world])
                colors.extend([color, color])
            else:
                length = prefs.axis_guide_world_length
                vertices.extend([anchor, anchor + axis_vec * length])
                colors.extend([color, color])
        if not vertices:
            return
        shader = gpu.shader.from_builtin('FLAT_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": vertices, "color": colors})
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(3.0)
        gpu.state.depth_test_set('LESS_EQUAL')
        batch.draw(shader)
        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)
        gpu.state.depth_test_set('NONE')
    except Exception:
        pass


def draw_preview_line_colored(start, end, axis, locked):
    if not GPU_AVAILABLE:
        return
    try:
        color = _get_axis_color(axis, locked)
        shader = gpu.shader.from_builtin('FLAT_COLOR')
        batch = batch_for_shader(shader, 'LINES',
                                 {"pos": [start, end], "color": [color, color]})
        line_width = 3.0 if locked else 2.5
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(line_width)
        gpu.state.depth_test_set('LESS_EQUAL')
        batch.draw(shader)
        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)
        gpu.state.depth_test_set('NONE')
    except Exception:
        pass


def draw_preview_rectangle_colored(start, end, plane, axis, locked):
    if not GPU_AVAILABLE:
        return
    try:
        ax_a, ax_b = _plane_axes(plane)
        d = end - start
        w = d.dot(ax_a)
        h = d.dot(ax_b)
        p1 = start
        p2 = start + ax_a * w
        p3 = start + ax_a * w + ax_b * h
        p4 = start + ax_b * h
        edges = [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]
        color = _get_axis_color(axis, locked)
        vertices = []
        colors = []
        for es, ee in edges:
            vertices.extend([es, ee])
            colors.extend([color, color])
        shader = gpu.shader.from_builtin('FLAT_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": vertices, "color": colors})
        line_width = 3.0 if locked else 2.5
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(line_width)
        gpu.state.depth_test_set('LESS_EQUAL')
        batch.draw(shader)

        # Semi-transparent face fill (SketchUp preview)
        face_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        face_batch = batch_for_shader(face_shader, 'TRIS',
                                      {"pos": [p1, p2, p3, p1, p3, p4]})
        face_shader.bind()
        face_shader.uniform_float("color", FACE_FILL_COLOR)
        face_batch.draw(face_shader)

        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)
        gpu.state.depth_test_set('NONE')
    except Exception:
        pass


# =============================================================================
# BASE DRAW OPERATOR  (GPU-only preview, no temp objects)
# =============================================================================
class _BaseDrawOperator:
    """
    Shared modal infrastructure.
    Subclasses implement: _tool_name(), _update_preview(), _commit()
    """
    _tmp_prefix = "SU_TMP"

    def _tool_name(self):
        return "Tool"

    def _update_preview(self, context):
        raise NotImplementedError

    def _commit(self, context):
        raise NotImplementedError

    def _get_measurement(self, context):
        """Get measurement string for current tool state."""
        return ""

    def _invoke_common(self, context, event):
        self._start = None
        self._end = None
        self._drawing = False
        self._typing = ""
        self._axis_lock = "FREE"
        self._shift_down = False
        self._kd = None
        self._meta = None
        self._depsgraph = None
        self._snap_info = None
        self._frame_count = 0
        self._draw_handler_key = None
        self._inferred_axis = "FREE"
        self._mouse_start = None
        self._continuous_start = None  # NEW: for continuous drawing

        if not context.area or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "Must be in 3D View area")
            return {'CANCELLED'}
        try:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)
            self._register_draw_handlers(context)
            # Add VCB draw handler
            self._vcb_handle = bpy.types.SpaceView3D.draw_handler_add(
                draw_vcb_box, (), 'WINDOW', 'POST_PIXEL'
            )
            context.window_manager.modal_handler_add(self)
            _set_header(context, _header_status(
                self._tool_name(), context.scene.su_draw_plane,
                self._axis_lock, self._typing, None,
            ))
            return {'RUNNING_MODAL'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start {self._tool_name()} tool: {e}")
            self._cleanup(context)
            return {'CANCELLED'}

    def _register_draw_handlers(self, context):
        if not GPU_AVAILABLE:
            return
        prefs = _get_prefs(context)
        if not prefs:
            return
        op_ref = self
        self._draw_handler_key = f"{self._tool_name().lower()}_{id(self)}"

        def draw_snap_marker():
            snap_info = getattr(op_ref, '_snap_info', None)
            if snap_info and prefs.show_snap_marker:
                draw_snap_marker_simple(snap_info, prefs)

        def draw_axis_guides():
            start = getattr(op_ref, '_start', None)
            axis_lock = getattr(op_ref, '_axis_lock', 'FREE')
            if start and prefs.show_axis_guides:
                draw_axis_guides_simple(
                    context, start, context.scene.su_draw_plane, axis_lock, prefs)

        def draw_preview_colored():
            start = getattr(op_ref, '_start', None)
            end = getattr(op_ref, '_end', None)
            drawing = getattr(op_ref, '_drawing', False)
            axis_lock = getattr(op_ref, '_axis_lock', 'FREE')
            inferred_axis = getattr(op_ref, '_inferred_axis', 'FREE')
            tool_name = getattr(op_ref, '_tool_name', lambda: "")()
            if start and end and drawing:
                axis = axis_lock if axis_lock != "FREE" else inferred_axis
                locked = (axis_lock != "FREE")
                if axis != "FREE":
                    if tool_name == "Rectangle":
                        draw_preview_rectangle_colored(
                            start, end, context.scene.su_draw_plane, axis, locked)
                    else:
                        draw_preview_line_colored(start, end, axis, locked)

        try:
            h_pixel = bpy.types.SpaceView3D.draw_handler_add(
                draw_snap_marker, (), 'WINDOW', 'POST_PIXEL')
            h_view = bpy.types.SpaceView3D.draw_handler_add(
                draw_axis_guides, (), 'WINDOW', 'POST_VIEW')
            h_preview = bpy.types.SpaceView3D.draw_handler_add(
                draw_preview_colored, (), 'WINDOW', 'POST_VIEW')
            _draw_handlers[self._draw_handler_key] = (h_pixel, h_view, h_preview)
        except Exception as e:
            print(f"[SU] Failed to register draw handlers: {e}")

    def _cleanup(self, context=None):
        if self._draw_handler_key:
            unregister_draw_handler(self._draw_handler_key)
            self._draw_handler_key = None
        # Remove VCB draw handler
        if hasattr(self, "_vcb_handle") and self._vcb_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._vcb_handle, 'WINDOW')
            self._vcb_handle = None
        hide_vcb()

    def _finish(self, context):
        _clear_header(context)
        self._cleanup(context)
        hide_vcb()

    def _snap_pipeline(self, context, event, hit):
        snap_info = None
        if context.scene.su_enable_vertex_snap:
            target_name = _get_target_name(context)
            snap_co, sinfo = find_vertex_snap(
                context, event, hit, self._kd, self._meta, self._depsgraph,
                max_px=float(context.scene.su_snap_px),
                max_world=MAX_SNAP_DISTANCE_WORLD,
                target_name=target_name,
            )
            if snap_co is not None:
                return snap_co, sinfo
        prefs = _get_prefs(context)
        if prefs and prefs.grid_snap_enabled:
            hit = snap_to_grid(hit, prefs.grid_size)
        return hit, snap_info

    def _get_vcb_label(self):
        return ""

    def _get_vcb_value(self, context):
        return ""

    def _modal_common(self, context, event):
        if context.area:
            context.area.tag_redraw()
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)

        # VCB update - ALWAYS visible while tool is running
        try:
            update_vcb(label=self._get_vcb_label(), value=self._get_vcb_value(context), typed=self._typing, visible=True)
        except Exception:
            pass

        # Axis lock
        axis_key = _detect_axis_key(event)
        if axis_key:
            self._axis_lock = "FREE" if self._axis_lock == axis_key else axis_key
            self._update_header(context)
            return {'RUNNING_MODAL'}

        if event.type in _NAV_EVENTS and event.type not in {'X', 'Y', 'Z'}:
            return {'PASS_THROUGH'}
        if _handle_plane_keys(context, event):
            self._update_header(context)
            return {'RUNNING_MODAL'}
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._finish(context)
            return {'CANCELLED'}
        if event.type in {'LEFT_SHIFT', 'RIGHT_SHIFT'}:
            self._shift_down = event.value in {'PRESS', 'CLICK_DRAG'}
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and event.type in {'BACK_SPACE', 'DEL'}:
            self._typing = self._typing[:-1]
            self._update_header(context)
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and len(event.unicode) == 1:
            ch = event.unicode
            if ch.isdigit() or ch in ".,-+mMcCiInNfFtT\"\' ":
                self._typing += ch
                self._update_header(context)
                return {'RUNNING_MODAL'}
        return None

    def _update_header(self, context):
        measurement = self._get_measurement(context)
        _set_header(context, _header_status(
            self._tool_name(), context.scene.su_draw_plane,
            self._axis_lock, self._typing, self._snap_info,
            measurement
        ))


# =============================================================================
# LINE TOOL  (draws into target mesh, continuous, auto-weld, auto-face)
# =============================================================================
class SU_OT_line_grid(_BaseDrawOperator, bpy.types.Operator):
    """Draw a line on the current drawing plane – SketchUp style (single mesh)"""
    bl_idname = "su.line_grid"
    bl_label = "Line (SketchUp)"
    bl_options = {'REGISTER', 'UNDO'}

    def _tool_name(self):
        return "Line"

    def _get_measurement(self, context):
        """Get line length measurement."""
        if not self._drawing or self._start is None or self._end is None:
            return ""
        
        length = (self._end - self._start).length
        if length < 1e-9:
            return ""
        
        unit_system = context.scene.unit_settings.system
        return f"Length: {_format_measurement(length, unit_system)}"

    def _get_vcb_label(self):
        return "Length"

    def _get_vcb_value(self, context):
        if self._start is None or self._end is None:
            return ""
        return _format_length(context, (self._end - self._start).length)

    def invoke(self, context, event):
        return self._invoke_common(context, event)

    def _update_preview(self, context):
        pass  # GPU-only preview via draw handlers

    def _commit(self, context):
        if self._start is None or self._end is None:
            return
        if (self._end - self._start).length < 1e-9:
            return
        prefs = _get_prefs(context)
        merge = prefs.merge_distance if prefs else MERGE_DISTANCE
        commit_geometry_to_target(
            context,
            verts_world=[self._start.copy(), self._end.copy()],
            edges_idx=[(0, 1)],
            merge_dist=merge,
        )
        self.report({'INFO'}, f"Line: {(self._end - self._start).length:.4f}")

    def modal(self, context, event):
        res = self._modal_common(context, event)
        if res is not None:
            return res

        # Enter – apply typed length and commit line
        if event.value == 'PRESS' and event.type == 'RET':
            if self._drawing and self._start is not None:
                if self._typing:
                    length = _parse_length(context, self._typing)
                    if length is not None:
                        direction = (self._end - self._start)
                        if direction.length > 1e-9:
                            direction.normalize()
                            self._end = self._start + direction * float(length)
                    self._typing = ""
                
                # Commit the line with the typed length
                self._commit(context)
                
                # Check for continuous drawing
                prefs = _get_prefs(context)
                if prefs and prefs.continuous_draw:
                    # Rebuild KDTree so the new vert can be snapped to
                    self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(
                        context, force_rebuild=True)
                    self._start = self._end.copy()
                    self._end = self._start.copy()
                    self._typing = ""
                    # Only reset axis_lock if user doesn't want to keep it
                    if not (prefs.keep_axis_lock_in_continuous):
                        self._axis_lock = "FREE"
                    self._mouse_start = Vector((event.mouse_region_x, event.mouse_region_y))
                    self._update_header(context)
                    return {'RUNNING_MODAL'}
                else:
                    self._finish(context)
                    return {'FINISHED'}
            return {'RUNNING_MODAL'}

        plane = context.scene.su_draw_plane
        offset = context.scene.su_plane_offset
        hit = get_plane_hit(context, event, plane=plane, offset=offset)
        if hit is None:
            return {'RUNNING_MODAL'}

        hit, self._snap_info = self._snap_pipeline(context, event, hit)

        # Left click – start / commit+continue
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if not self._drawing:
                self._start = hit.copy()
                self._end = hit.copy()
                self._drawing = True
                self._mouse_start = Vector((event.mouse_region_x, event.mouse_region_y))
            else:
                self._end = hit.copy()
                self._commit(context)

                # ---- CONTINUOUS DRAWING (SketchUp style) ----
                prefs = _get_prefs(context)
                if prefs and prefs.continuous_draw:
                    # Rebuild KDTree so the new vert can be snapped to
                    self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(
                        context, force_rebuild=True)
                    self._start = self._end.copy()
                    self._end = self._start.copy()
                    self._typing = ""
                    # Only reset axis_lock if user doesn't want to keep it
                    if not (prefs.keep_axis_lock_in_continuous):
                        self._axis_lock = "FREE"
                    self._mouse_start = Vector((event.mouse_region_x, event.mouse_region_y))
                    self._update_header(context)
                    return {'RUNNING_MODAL'}
                else:
                    self._finish(context)
                    return {'FINISHED'}

        # Update during drawing
        if self._drawing:
            perpendicular = False
            if self._axis_lock == "Z" and plane == "XY":
                perpendicular = True
            elif self._axis_lock == "Y" and plane == "XZ":
                perpendicular = True
            elif self._axis_lock == "X" and plane == "YZ":
                perpendicular = True
            if perpendicular:
                cur = _compute_perpendicular_axis_movement(
                    context, event, self._start, self._axis_lock, self._mouse_start)
            else:
                cur = hit.copy()
                if self._shift_down and self._axis_lock == "FREE":
                    lock = _auto_axis_from_shift(self._start, cur, plane)
                    cur = _apply_axis_lock(self._start, cur, plane, lock)
                else:
                    cur = _apply_axis_lock(self._start, cur, plane, self._axis_lock)
            self._end = cur
            self._inferred_axis = _infer_dominant_axis(
                self._start, self._end, plane, self._axis_lock)
            self._update_header(context)

        return {'RUNNING_MODAL'}


# =============================================================================
# RECTANGLE TOOL  (single mesh, auto-face)
# =============================================================================
class SU_OT_rectangle_grid(_BaseDrawOperator, bpy.types.Operator):
    """Draw a rectangle on the current drawing plane – SketchUp style"""
    bl_idname = "su.rect_grid"
    bl_label = "Rectangle (SketchUp)"
    bl_options = {'REGISTER', 'UNDO'}

    def _tool_name(self):
        return "Rectangle"

    def _get_measurement(self, context):
        """Get rectangle width and height measurements."""
        if not self._drawing or self._start is None or self._end is None:
            return ""
        
        plane = context.scene.su_draw_plane
        ax_a, ax_b = _plane_axes(plane)
        delta = self._end - self._start
        width = abs(delta.dot(ax_a))
        height = abs(delta.dot(ax_b))
        
        if width < 1e-9 or height < 1e-9:
            return ""
        
        unit_system = context.scene.unit_settings.system
        width_str = _format_measurement(width, unit_system)
        height_str = _format_measurement(height, unit_system)
        
        return f"Width: {width_str} | Height: {height_str}"

    def _get_vcb_label(self):
        return "Length"

    def _get_vcb_value(self, context):
        if self._start is None or self._end is None:
            return ""
        plane = context.scene.su_draw_plane
        ax_a, ax_b = _plane_axes(plane)
        d = (self._end - self._start)
        w = abs(d.dot(ax_a))
        h = abs(d.dot(ax_b))
        return f"{_format_length(context, w)}, { _format_length(context, h)}"

    def invoke(self, context, event):
        return self._invoke_common(context, event)

    def _update_preview(self, context):
        pass  # GPU-only preview

    def _commit(self, context):
        if self._start is None or self._end is None:
            return
        plane = context.scene.su_draw_plane
        ax_a, ax_b = _plane_axes(plane)
        delta = self._end - self._start
        w = float(delta.dot(ax_a))
        h = float(delta.dot(ax_b))
        if abs(w) < 1e-9 or abs(h) < 1e-9:
            self.report({'WARNING'}, "Rectangle too small")
            return
        p0 = self._start.copy()
        p1 = p0 + ax_a * w
        p2 = p1 + ax_b * h
        p3 = p0 + ax_b * h
        prefs = _get_prefs(context)
        merge = prefs.merge_distance if prefs else MERGE_DISTANCE
        commit_geometry_to_target(
            context,
            verts_world=[p0, p1, p2, p3],
            edges_idx=[(0, 1), (1, 2), (2, 3), (3, 0)],
            faces_idx=[(0, 1, 2, 3)],
            merge_dist=merge,
        )
        self.report({'INFO'}, f"Rectangle: {abs(w):.4f} x {abs(h):.4f}")

    def modal(self, context, event):
        res = self._modal_common(context, event)
        if res is not None:
            return res

        if event.value == 'PRESS' and event.type == 'RET':
            if self._drawing and self._start is not None:
                if self._typing:
                    w, h = _parse_rect_two(context, self._typing)
                    if w is not None and h is not None:
                        plane = context.scene.su_draw_plane
                        ax_a, ax_b = _plane_axes(plane)
                        self._end = self._start + ax_a * float(w) + ax_b * float(h)
                    self._typing = ""
                    self._update_header(context)
                
                # Commit the rectangle with the typed dimensions
                self._commit(context)
                self._finish(context)
                return {'FINISHED'}
            return {'RUNNING_MODAL'}

        plane = context.scene.su_draw_plane
        offset = context.scene.su_plane_offset
        hit = get_plane_hit(context, event, plane=plane, offset=offset)
        if hit is None:
            return {'RUNNING_MODAL'}
        hit, self._snap_info = self._snap_pipeline(context, event, hit)

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if not self._drawing:
                self._start = hit.copy()
                self._end = hit.copy()
                self._drawing = True
                self._mouse_start = Vector((event.mouse_region_x, event.mouse_region_y))
            else:
                self._end = hit.copy()
                self._commit(context)
                self._finish(context)
                return {'FINISHED'}

        if self._drawing:
            perpendicular = False
            if self._axis_lock == "Z" and plane == "XY":
                perpendicular = True
            elif self._axis_lock == "Y" and plane == "XZ":
                perpendicular = True
            elif self._axis_lock == "X" and plane == "YZ":
                perpendicular = True
            if perpendicular:
                cur = _compute_perpendicular_axis_movement(
                    context, event, self._start, self._axis_lock, self._mouse_start)
            else:
                cur = hit.copy()
                cur = _apply_axis_lock(self._start, cur, plane, self._axis_lock)
                if self._shift_down:
                    ax_a, ax_b = _plane_axes(plane)
                    d = cur - self._start
                    ww = d.dot(ax_a)
                    hh = d.dot(ax_b)
                    s = ww if abs(ww) >= abs(hh) else hh
                    cur = self._start + ax_a * s + ax_b * s
            self._end = cur
            self._inferred_axis = _infer_dominant_axis(
                self._start, self._end, plane, self._axis_lock)
            self._update_header(context)
        return {'RUNNING_MODAL'}


# =============================================================================
# ARC TOOL  (single mesh)
# =============================================================================
def _arc_header_status(stage, plane, typing, snap_info, measurement=""):
    stages = {
        "START": "Arc: Click start point | 1/2/3: Plane | Esc: Cancel",
        "END": "Arc: Click end point | 1/2/3: Plane | Esc: Cancel",
        "BULGE": "Arc: Set bulge/radius | Type radius + Enter | LMB: Confirm | Esc: Cancel",
    }
    base = stages.get(stage, "Arc Tool")
    snap = " [SNAP]" if snap_info else ""
    t = f" | Type: {typing}" if typing else ""
    measure = f" | {measurement}" if measurement else ""
    return f"{base} | Plane: {plane}{snap}{measure}{t}"


class SU_OT_arc_grid(bpy.types.Operator):
    """Draw an arc on the current drawing plane – SketchUp style"""
    bl_idname = "su.arc_grid"
    bl_label = "Arc (SketchUp)"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        self._stage = "START"
        self._start = None
        self._end = None
        self._bulge_point = None
        self._radius = 1.0
        self._typing = ""
        self._kd = None
        self._meta = None
        self._depsgraph = None
        self._snap_info = None
        self._frame_count = 0
        self._draw_handler_key = None
        self._inferred_axis = "FREE"
        self._mouse_start = None

        if context.space_data.type != 'VIEW_3D':
            self.report({'WARNING'}, "Must be in 3D View")
            return {'CANCELLED'}
        try:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)
            self._register_draw_handlers(context)
            # Add VCB draw handler
            self._vcb_handle = bpy.types.SpaceView3D.draw_handler_add(
                draw_vcb_box, (), 'WINDOW', 'POST_PIXEL'
            )
            context.window_manager.modal_handler_add(self)
            measurement = self._get_arc_measurement(context) if self._stage == "BULGE" else ""
            _set_header(context, _arc_header_status(
                self._stage, context.scene.su_draw_plane, self._typing, None, measurement))
            return {'RUNNING_MODAL'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start Arc tool: {e}")
            self._cleanup(context)
            return {'CANCELLED'}

    def _register_draw_handlers(self, context):
        if not GPU_AVAILABLE:
            return
        prefs = _get_prefs(context)
        if not prefs:
            return
        op_ref = self
        self._draw_handler_key = f"arc_{id(self)}"

        def draw_snap_marker():
            si = getattr(op_ref, '_snap_info', None)
            if si and prefs.show_snap_marker:
                draw_snap_marker_simple(si, prefs)

        def draw_axis_guides():
            stage = getattr(op_ref, '_stage', 'START')
            start = getattr(op_ref, '_start', None)
            end = getattr(op_ref, '_end', None)
            if prefs.show_axis_guides:
                anchor = None
                if stage == "END" and start:
                    anchor = start
                elif stage == "BULGE" and start and end:
                    anchor = (start + end) * 0.5
                if anchor:
                    draw_axis_guides_simple(
                        context, anchor, context.scene.su_draw_plane, "FREE", prefs)

        def draw_preview():
            stage = getattr(op_ref, '_stage', 'START')
            start = getattr(op_ref, '_start', None)
            end = getattr(op_ref, '_end', None)
            inferred = getattr(op_ref, '_inferred_axis', 'FREE')
            if stage == "END" and start and end and inferred != "FREE":
                draw_preview_line_colored(start, end, inferred, False)
            # Arc curve preview during BULGE
            if stage == "BULGE" and start and end:
                points = op_ref._calculate_arc_geometry(context)
                if len(points) >= 2:
                    color = (1.0, 0.7, 0.0, 0.9)
                    verts = []
                    cols = []
                    for i in range(len(points) - 1):
                        verts.extend([points[i], points[i + 1]])
                        cols.extend([color, color])
                    try:
                        shader = gpu.shader.from_builtin('FLAT_COLOR')
                        batch = batch_for_shader(shader, 'LINES',
                                                 {"pos": verts, "color": cols})
                        gpu.state.blend_set('ALPHA')
                        gpu.state.line_width_set(2.5)
                        gpu.state.depth_test_set('LESS_EQUAL')
                        batch.draw(shader)
                        gpu.state.blend_set('NONE')
                        gpu.state.line_width_set(1.0)
                        gpu.state.depth_test_set('NONE')
                    except Exception:
                        pass

        try:
            h1 = bpy.types.SpaceView3D.draw_handler_add(
                draw_snap_marker, (), 'WINDOW', 'POST_PIXEL')
            h2 = bpy.types.SpaceView3D.draw_handler_add(
                draw_axis_guides, (), 'WINDOW', 'POST_VIEW')
            h3 = bpy.types.SpaceView3D.draw_handler_add(
                draw_preview, (), 'WINDOW', 'POST_VIEW')
            _draw_handlers[self._draw_handler_key] = (h1, h2, h3)
        except Exception as e:
            print(f"[SU] Failed to register draw handlers: {e}")

    def _calculate_arc_geometry(self, context):
        if not self._start or not self._end:
            return []
        plane = context.scene.su_draw_plane
        chord = self._end - self._start
        chord_len = chord.length
        if chord_len < 1e-6:
            return []
        mid = (self._start + self._end) * 0.5
        if self._bulge_point:
            normals = {
                "XY": Vector((0, 0, 1)), "XZ": Vector((0, 1, 0)), "YZ": Vector((1, 0, 0))}
            chord_norm = chord.cross(normals.get(plane, Vector((0, 0, 1))))
            if chord_norm.length > 1e-6:
                chord_norm.normalize()
                bulge_dist = (self._bulge_point - mid).dot(chord_norm)
            else:
                bulge_dist = (self._bulge_point - mid).length
        else:
            bulge_dist = self._radius
        if abs(bulge_dist) < 1e-6:
            return [self._start, self._end]
        radius = (chord_len ** 2) / (8.0 * abs(bulge_dist)) + abs(bulge_dist) / 2.0
        perp_map = {
            "XY": Vector((-chord.y, chord.x, 0.0)),
            "XZ": Vector((-chord.z, 0.0, chord.x)),
            "YZ": Vector((0.0, -chord.z, chord.y)),
        }
        perp = perp_map.get(plane, Vector((-chord.y, chord.x, 0.0)))
        if perp.length > 1e-6:
            perp.normalize()
        else:
            return [self._start, self._end]
        center_dist = math.sqrt(max(0, radius ** 2 - (chord_len * 0.5) ** 2))
        center = mid + perp * center_dist * (1.0 if bulge_dist > 0 else -1.0)
        v1 = self._start - center
        v2 = self._end - center
        cross = v1.cross(v2)
        dot = v1.dot(v2)
        angle = math.atan2(cross.length, dot)
        axis_map = {"XY": 2, "XZ": 1, "YZ": 0}
        ci = axis_map.get(plane, 2)
        sign_check = cross[ci] * bulge_dist
        if plane == "XZ":
            if sign_check > 0:
                angle = -angle
        else:
            if sign_check < 0:
                angle = -angle
        prefs = _get_prefs(context)
        segments = prefs.arc_segments if prefs else DEFAULT_ARC_SEGMENTS
        segments = max(MIN_ARC_SEGMENTS, min(MAX_ARC_SEGMENTS, segments))
        axis_vec_map = {
            "XY": Vector((0, 0, 1)), "XZ": Vector((0, 1, 0)), "YZ": Vector((1, 0, 0))}
        axis_vec = axis_vec_map.get(plane, Vector((0, 0, 1)))
        points = []
        for i in range(segments + 1):
            t = i / segments
            ca = t * angle
            cos_a = math.cos(ca)
            sin_a = math.sin(ca)
            rotated = v1 * cos_a + axis_vec.cross(v1) * sin_a + axis_vec * axis_vec.dot(v1) * (1 - cos_a)
            points.append(center + rotated)
        return points

    def _commit_arc(self, context):
        points = self._calculate_arc_geometry(context)
        if len(points) < 2:
            self.report({'WARNING'}, "Arc too small")
            return
        prefs = _get_prefs(context)
        merge = prefs.merge_distance if prefs else MERGE_DISTANCE
        edges = [(i, i + 1) for i in range(len(points) - 1)]
        commit_geometry_to_target(
            context,
            verts_world=points,
            edges_idx=edges,
            merge_dist=merge,
        )
        self.report({'INFO'}, f"Arc: {len(points)} points")

    def _get_arc_measurement(self, context):
        """Get arc measurements: radius, arc length, angle."""
        if self._stage != "BULGE" or not self._start or not self._end:
            return ""
        
        points = self._calculate_arc_geometry(context)
        if len(points) < 2:
            return ""
        
        # Calculate chord length
        chord_len = (self._end - self._start).length
        if chord_len < 1e-9:
            return ""
        
        # Calculate radius from geometry
        mid = (self._start + self._end) * 0.5
        if self._bulge_point:
            bulge_dist = (self._bulge_point - mid).length
        else:
            bulge_dist = self._radius
        
        if abs(bulge_dist) < 1e-6:
            return ""
        
        radius = (chord_len ** 2) / (8.0 * abs(bulge_dist)) + abs(bulge_dist) / 2.0
        
        # Calculate angle
        v1 = self._start - mid
        v2 = self._end - mid
        if v1.length < 1e-9 or v2.length < 1e-9:
            return ""
        
        # Calculate arc length
        plane = context.scene.su_draw_plane
        axis_map = {"XY": Vector((0, 0, 1)), "XZ": Vector((0, 1, 0)), "YZ": Vector((1, 0, 0))}
        axis_vec = axis_map.get(plane, Vector((0, 0, 1)))
        
        cross = v1.cross(v2)
        dot = v1.dot(v2)
        angle = math.atan2(cross.length, dot)
        
        # Adjust angle direction
        ci = {"XY": 2, "XZ": 1, "YZ": 0}.get(plane, 2)
        sign_check = cross[ci] * bulge_dist
        if plane == "XZ":
            if sign_check > 0:
                angle = -angle
        else:
            if sign_check < 0:
                angle = -angle
        
        arc_length = abs(radius * angle)
        
        unit_system = context.scene.unit_settings.system
        radius_str = _format_measurement(radius, unit_system)
        length_str = _format_measurement(arc_length, unit_system)
        angle_deg = math.degrees(abs(angle))
        
        return f"Radius: {radius_str} | Length: {length_str} | Angle: {angle_deg:.1f}°"

    def _update_arc_header(self, context):
        measurement = self._get_arc_measurement(context) if self._stage == "BULGE" else ""
        _set_header(context, _arc_header_status(
            self._stage, context.scene.su_draw_plane, self._typing, self._snap_info, measurement))

    def _cleanup(self, context=None):
        if self._draw_handler_key:
            unregister_draw_handler(self._draw_handler_key)
            self._draw_handler_key = None
        # Remove VCB draw handler
        if hasattr(self, "_vcb_handle") and self._vcb_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._vcb_handle, 'WINDOW')
            self._vcb_handle = None

    def _finish(self, context):
        _clear_header(context)
        self._cleanup(context)

    def modal(self, context, event):
        if context.area:
            context.area.tag_redraw()
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)

        axis_key = _detect_axis_key(event)
        if axis_key:
            return {'RUNNING_MODAL'}

        if event.type in _NAV_EVENTS and event.type not in {'X', 'Y', 'Z'}:
            return {'PASS_THROUGH'}
        if _handle_plane_keys(context, event):
            self._update_arc_header(context)
            return {'RUNNING_MODAL'}
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._finish(context)
            return {'CANCELLED'}
        if event.value == 'PRESS' and event.type in {'BACK_SPACE', 'DEL'}:
            if self._typing:
                self._typing = self._typing[:-1]
            else:
                if self._stage == "BULGE":
                    self._stage = "END"
                    self._bulge_point = None
                elif self._stage == "END":
                    self._stage = "START"
                    self._end = None
            self._update_arc_header(context)
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and event.type == 'RET':
            if self._stage == "BULGE" and self._typing:
                radius, angle = _parse_arc_input(context, self._typing)
                if radius is not None and radius > 0:
                    self._radius = float(radius)
                # Note: angle parsing would need more complex handling
                # to adjust the arc based on angle input
                self._typing = ""
                self._update_arc_header(context)
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and len(event.unicode) == 1:
            ch = event.unicode
            if ch.isdigit() or ch in ".-+mMcCiInNfFtT\"\' ":
                self._typing += ch
                self._update_arc_header(context)
                return {'RUNNING_MODAL'}

        
        # VCB update
        try:
            if self._stage == "BULGE":
                if self._start and self._end:
                    rad_txt = _format_length(context, self._radius) if self._radius else ""
                    arc_pts = self._calculate_arc_geometry(context)
                    arc_len = 0.0
                    if len(arc_pts) >= 2:
                        for ai in range(len(arc_pts) - 1):
                            arc_len += (arc_pts[ai + 1] - arc_pts[ai]).length
                    angle_deg = 0.0
                    if self._radius and self._radius > 1e-9 and arc_len > 0:
                        angle_deg = math.degrees(arc_len / self._radius)
                    vcb_val = f"R:{rad_txt}  L:{_format_length(context, arc_len)}  A:{angle_deg:.1f}°"
                    update_vcb(label="Arc", value=vcb_val, typed=self._typing, visible=True)
                else:
                    rad_txt = "" if self._radius is None else _format_length(context, self._radius)
                    update_vcb(label="Radius", value=rad_txt, typed=self._typing, visible=True)
            elif self._stage == "END" and self._start and self._end:
                update_vcb(label="Chord", value=_format_length(context, (self._end - self._start).length), typed=self._typing, visible=True)
            else:
                update_vcb(label="Arc", value="", typed=self._typing, visible=True)
        except Exception:
            pass

        plane = context.scene.su_draw_plane
        offset = context.scene.su_plane_offset
        hit = get_plane_hit(context, event, plane=plane, offset=offset)
        if hit is None:
            return {'RUNNING_MODAL'}
        self._snap_info = None
        if context.scene.su_enable_vertex_snap:
            target_name = _get_target_name(context)
            snap_co, snap_info = find_vertex_snap(
                context, event, hit, self._kd, self._meta, self._depsgraph,
                max_px=float(context.scene.su_snap_px), max_world=MAX_SNAP_DISTANCE_WORLD,
                target_name=target_name)
            if snap_co is not None:
                hit = snap_co
                self._snap_info = snap_info
        if self._snap_info is None:
            prefs = _get_prefs(context)
            if prefs and prefs.grid_snap_enabled:
                hit = snap_to_grid(hit, prefs.grid_size)

        if self._stage == "BULGE":
            self._bulge_point = hit.copy()
            if self._start and self._end:
                mid = (self._start + self._end) * 0.5
                self._radius = (self._bulge_point - mid).length
            self._update_arc_header(context)
        elif self._stage == "END" and self._start:
            self._end = hit.copy()
            self._inferred_axis = _infer_dominant_axis(self._start, self._end, plane, "FREE")
            self._update_arc_header(context)

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._stage == "START":
                self._start = hit.copy()
                self._stage = "END"
                self._mouse_start = Vector((event.mouse_region_x, event.mouse_region_y))
            elif self._stage == "END":
                self._end = hit.copy()
                self._stage = "BULGE"
            elif self._stage == "BULGE":
                self._commit_arc(context)
                self._finish(context)
                return {'FINISHED'}
            self._update_arc_header(context)
        return {'RUNNING_MODAL'}




# =============================================================================
# THREE-POINT ARC TOOL
# =============================================================================
def _three_pt_arc_header(stage, plane, typing, snap_info, measurement=""):
    stages = {
        "START": "3P Arc: Click start | 1/2/3: Plane | Esc: Cancel",
        "END": "3P Arc: Click end | Esc: Cancel",
        "THIRD": "3P Arc: Click point on arc | Enter: Confirm | Esc: Cancel",
    }
    base = stages.get(stage, "3P Arc")
    snap = " [SNAP]" if snap_info else ""
    t = f" | Type: {typing}" if typing else ""
    measure = f" | {measurement}" if measurement else ""
    return f"{base} | Plane: {plane}{snap}{measure}{t}"


class SU_OT_arc_three_point(bpy.types.Operator):
    """Three-point arc: start, end, point on curve"""
    bl_idname = "su.arc_three_point"
    bl_label = "Three-Point Arc (SketchUp)"
    bl_options = {'REGISTER', 'UNDO'}

    def _get_three_pt_measurement(self, context):
        """Get three-point arc measurements."""
        if self._stage == "THIRD" and self._start and self._end and self._third:
            pn = _get_plane_normal(context.scene.su_draw_plane)
            c, r = _circle_from_3pts(self._start, self._end, self._third, pn)
            if c and r and r > 1e-9:
                a1 = _angle_on_plane(self._start, c, pn)
                at = _angle_on_plane(self._third, c, pn)
                a2 = _angle_on_plane(self._end, c, pn)
                angs = _order_arc_angles(a1, at, a2)
                if angs:
                    sa, ea = angs
                    seg = _get_prefs(context).arc_segments if _get_prefs(context) else DEFAULT_ARC_SEGMENTS
                    pts = _gen_arc_pts(c, r, sa, ea, seg, pn)
                    if len(pts) >= 2:
                        arc_len = sum((pts[i+1]-pts[i]).length for i in range(len(pts)-1))
                        angle_deg = math.degrees(abs(ea-sa))
                        unit_system = context.scene.unit_settings.system
                        rad_str = _format_measurement(r, unit_system)
                        len_str = _format_measurement(arc_len, unit_system)
                        return f"R:{rad_str}  L:{len_str}  A:{angle_deg:.1f}°"
        elif self._stage == "END" and self._start and self._end:
            chord_len = (self._end - self._start).length
            if chord_len > 1e-9:
                unit_system = context.scene.unit_settings.system
                chord_str = _format_measurement(chord_len, unit_system)
                return f"Chord: {chord_str}"
        return ""

    def invoke(self, context, event):
        self._stage = "START"
        self._start = None
        self._end = None
        self._third = None
        self._typing = ""
        self._kd = None
        self._meta = None
        self._depsgraph = None
        self._snap_info = None
        self._frame_count = 0
        self._draw_handler_key = None
        self._vcb_handle = None
        self._arc_center = None
        self._arc_radius = None
        if context.space_data.type != 'VIEW_3D':
            self.report({'WARNING'}, "Must be in 3D View")
            return {'CANCELLED'}
        try:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)
            self._register_gpu(context)
            self._vcb_handle = bpy.types.SpaceView3D.draw_handler_add(draw_vcb_box, (), 'WINDOW', 'POST_PIXEL')
            context.window_manager.modal_handler_add(self)
            _set_header(context, _three_pt_arc_header(self._stage, context.scene.su_draw_plane, self._typing, None, self._get_three_pt_measurement(context)))
            return {'RUNNING_MODAL'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed: {e}")
            self._cleanup(context)
            return {'CANCELLED'}

    def _register_gpu(self, context):
        if not GPU_AVAILABLE:
            return
        prefs = _get_prefs(context)
        op = self
        self._draw_handler_key = f"a3p_{id(self)}"
        def _draw_snap():
            si = getattr(op, '_snap_info', None)
            if si and prefs and prefs.show_snap_marker:
                draw_snap_marker_simple(si, prefs)
        def _draw_prev():
            s = getattr(op, '_start', None)
            e = getattr(op, '_end', None)
            th = getattr(op, '_third', None)
            st = getattr(op, '_stage', 'START')
            if st == "END" and s and e:
                draw_preview_line_colored(s, e, "FREE", False)
            if st == "THIRD" and s and e and th:
                pn = _get_plane_normal(context.scene.su_draw_plane)
                ctr, rad = _circle_from_3pts(s, e, th, pn)
                if ctr and rad and rad > 1e-9:
                    a1 = _angle_on_plane(s, ctr, pn)
                    at = _angle_on_plane(th, ctr, pn)
                    a2 = _angle_on_plane(e, ctr, pn)
                    angs = _order_arc_angles(a1, at, a2)
                    if angs:
                        sa, ea = angs
                        seg = prefs.arc_segments if prefs else DEFAULT_ARC_SEGMENTS
                        pp = _gen_arc_pts(ctr, rad, sa, ea, seg, pn)
                        if len(pp) >= 2:
                            col = ARC_PREVIEW_COLOR
                            vs = []; cs = []
                            for ii in range(len(pp)-1):
                                vs.extend([pp[ii], pp[ii+1]])
                                cs.extend([col, col])
                            try:
                                sh = gpu.shader.from_builtin('FLAT_COLOR')
                                ba = batch_for_shader(sh, 'LINES', {"pos": vs, "color": cs})
                                gpu.state.blend_set('ALPHA')
                                gpu.state.line_width_set(2.5)
                                gpu.state.depth_test_set('LESS_EQUAL')
                                ba.draw(sh)
                                gpu.state.blend_set('NONE')
                                gpu.state.line_width_set(1.0)
                                gpu.state.depth_test_set('NONE')
                            except Exception:
                                pass
        try:
            h1 = bpy.types.SpaceView3D.draw_handler_add(_draw_snap, (), 'WINDOW', 'POST_PIXEL')
            h2 = bpy.types.SpaceView3D.draw_handler_add(_draw_prev, (), 'WINDOW', 'POST_VIEW')
            _draw_handlers[self._draw_handler_key] = (h1, h2)
        except Exception:
            pass

    def _cleanup(self, context=None):
        if self._draw_handler_key:
            unregister_draw_handler(self._draw_handler_key)
            self._draw_handler_key = None
        if hasattr(self, "_vcb_handle") and self._vcb_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._vcb_handle, 'WINDOW')
            self._vcb_handle = None
        hide_vcb()

    def _finish(self, context):
        _clear_header(context)
        self._cleanup(context)
        hide_vcb()

    def _commit(self, context):
        if not self._start or not self._end or not self._third:
            return
        plane = context.scene.su_draw_plane
        pn = _get_plane_normal(plane)
        ctr, rad = _circle_from_3pts(self._start, self._end, self._third, pn)
        if not ctr or not rad or rad < 1e-9:
            self.report({'WARNING'}, "Cannot compute arc")
            return
        a1 = _angle_on_plane(self._start, ctr, pn)
        at = _angle_on_plane(self._third, ctr, pn)
        a2 = _angle_on_plane(self._end, ctr, pn)
        angs = _order_arc_angles(a1, at, a2)
        if not angs:
            self.report({'WARNING'}, "Bad arc angles")
            return
        sa, ea = angs
        prefs = _get_prefs(context)
        seg = prefs.arc_segments if prefs else DEFAULT_ARC_SEGMENTS
        pts = _gen_arc_pts(ctr, rad, sa, ea, seg, pn)
        if len(pts) < 2:
            self.report({'WARNING'}, "Arc too small")
            return
        merge = prefs.merge_distance if prefs else MERGE_DISTANCE
        ei = [(i, i+1) for i in range(len(pts)-1)]
        commit_geometry_to_target(context, verts_world=pts, edges_idx=ei, merge_dist=merge)
        al = sum((pts[i+1]-pts[i]).length for i in range(len(pts)-1))
        ad = math.degrees(abs(ea-sa))
        self.report({'INFO'}, f"3P Arc: R={rad:.4f} L={al:.4f} A={ad:.1f}\u00b0")

    def modal(self, context, event):
        if context.area:
            context.area.tag_redraw()
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)
        if event.type in _NAV_EVENTS:
            return {'PASS_THROUGH'}
        if _handle_plane_keys(context, event):
            _set_header(context, _three_pt_arc_header(self._stage, context.scene.su_draw_plane, self._typing, self._snap_info, self._get_three_pt_measurement(context)))
            return {'RUNNING_MODAL'}
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._finish(context)
            return {'CANCELLED'}
        if event.value == 'PRESS' and event.type in {'BACK_SPACE', 'DEL'}:
            if self._typing:
                self._typing = self._typing[:-1]
            elif self._stage == "THIRD":
                self._stage = "END"; self._third = None
            elif self._stage == "END":
                self._stage = "START"; self._end = None
            _set_header(context, _three_pt_arc_header(self._stage, context.scene.su_draw_plane, self._typing, self._snap_info, self._get_three_pt_measurement(context)))
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and event.type == 'RET':
            if self._stage == "THIRD" and self._start and self._end and self._third:
                self._commit(context)
                self._finish(context)
                return {'FINISHED'}
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and len(event.unicode) == 1:
            ch = event.unicode
            if ch.isdigit() or ch in ".-+mMcCiInNfFtT\"\' ":
                self._typing += ch
                return {'RUNNING_MODAL'}
        try:
            if self._stage == "THIRD" and self._arc_radius:
                update_vcb(label="3P Arc R", value=_format_length(context, self._arc_radius), typed=self._typing, visible=True)
            elif self._stage == "END" and self._start and self._end:
                update_vcb(label="Chord", value=_format_length(context, (self._end-self._start).length), typed=self._typing, visible=True)
            else:
                update_vcb(label="3P Arc", value="", typed=self._typing, visible=True)
        except Exception:
            pass
        plane = context.scene.su_draw_plane
        offset = context.scene.su_plane_offset
        hit = get_plane_hit(context, event, plane=plane, offset=offset)
        if hit is None:
            return {'RUNNING_MODAL'}
        self._snap_info = None
        if context.scene.su_enable_vertex_snap:
            tn = _get_target_name(context)
            sc, si = find_vertex_snap(context, event, hit, self._kd, self._meta, self._depsgraph, max_px=float(context.scene.su_snap_px), max_world=MAX_SNAP_DISTANCE_WORLD, target_name=tn)
            if sc is not None:
                hit = sc; self._snap_info = si
        if self._snap_info is None:
            pr = _get_prefs(context)
            if pr and pr.grid_snap_enabled:
                hit = snap_to_grid(hit, pr.grid_size)
        if self._stage == "THIRD":
            self._third = hit.copy()
            if self._start and self._end:
                pn = _get_plane_normal(plane)
                c, r = _circle_from_3pts(self._start, self._end, self._third, pn)
                if c and r:
                    self._arc_center = c; self._arc_radius = r
            _set_header(context, _three_pt_arc_header(self._stage, context.scene.su_draw_plane, self._typing, self._snap_info, self._get_three_pt_measurement(context)))
        elif self._stage == "END" and self._start:
            self._end = hit.copy()
            _set_header(context, _three_pt_arc_header(self._stage, context.scene.su_draw_plane, self._typing, self._snap_info, self._get_three_pt_measurement(context)))
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._stage == "START":
                self._start = hit.copy(); self._stage = "END"
            elif self._stage == "END":
                self._end = hit.copy(); self._stage = "THIRD"
            elif self._stage == "THIRD":
                self._third = hit.copy()
                self._commit(context)
                self._finish(context)
                return {'FINISHED'}
            _set_header(context, _three_pt_arc_header(self._stage, context.scene.su_draw_plane, self._typing, self._snap_info, self._get_three_pt_measurement(context)))
        return {'RUNNING_MODAL'}


# =============================================================================
# CENTER-POINT ARC TOOL
# =============================================================================
def _center_arc_header(stage, plane, typing, snap_info, measurement=""):
    stages = {
        "CENTER": "Center Arc: Click center | Esc: Cancel",
        "RADIUS": "Center Arc: Set radius | Type+Enter | Esc: Cancel",
        "ANGLE": "Center Arc: Set end angle | LMB: Confirm | Type angle+Enter | Esc: Cancel",
    }
    base = stages.get(stage, "Center Arc")
    snap = " [SNAP]" if snap_info else ""
    t = f" | Type: {typing}" if typing else ""
    measure = f" | {measurement}" if measurement else ""
    return f"{base} | Plane: {plane}{snap}{measure}{t}"


class SU_OT_arc_center_point(bpy.types.Operator):
    """Center-point arc: center, radius, sweep angle"""
    bl_idname = "su.arc_center_point"
    bl_label = "Center-Point Arc (SketchUp)"
    bl_options = {'REGISTER', 'UNDO'}

    def _get_center_arc_measurement(self, context):
        """Get center-point arc measurements."""
        if self._stage == "ANGLE" and self._center and self._radius > 1e-9 and self._radius_point and self._end_point:
            pn = _get_plane_normal(context.scene.su_draw_plane)
            a1 = _angle_on_plane(self._radius_point, self._center, pn)
            a2 = _angle_on_plane(self._end_point, self._center, pn)
            diff = a2 - a1
            if diff > math.pi: diff -= 2*math.pi
            elif diff < -math.pi: diff += 2*math.pi
            seg = _get_prefs(context).arc_segments if _get_prefs(context) else DEFAULT_ARC_SEGMENTS
            pts = _gen_arc_pts(self._center, self._radius, a1, a1+diff, seg, pn)
            if len(pts) >= 2:
                arc_len = sum((pts[i+1]-pts[i]).length for i in range(len(pts)-1))
                angle_deg = math.degrees(abs(diff))
                unit_system = context.scene.unit_settings.system
                rad_str = _format_measurement(self._radius, unit_system)
                len_str = _format_measurement(arc_len, unit_system)
                return f"R:{rad_str}  L:{len_str}  A:{angle_deg:.1f}°"
        elif self._stage == "RADIUS" and self._center and self._radius_point:
            rad = (self._radius_point - self._center).length
            if rad > 1e-9:
                unit_system = context.scene.unit_settings.system
                rad_str = _format_measurement(rad, unit_system)
                return f"Radius: {rad_str}"
        return ""

    def invoke(self, context, event):
        self._stage = "CENTER"
        self._center = None
        self._radius_point = None
        self._end_point = None
        self._radius = 0.0
        self._typing = ""
        self._kd = None; self._meta = None; self._depsgraph = None
        self._snap_info = None; self._frame_count = 0
        self._draw_handler_key = None; self._vcb_handle = None
        if context.space_data.type != 'VIEW_3D':
            self.report({'WARNING'}, "Must be in 3D View"); return {'CANCELLED'}
        try:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)
            self._register_gpu(context)
            self._vcb_handle = bpy.types.SpaceView3D.draw_handler_add(draw_vcb_box, (), 'WINDOW', 'POST_PIXEL')
            context.window_manager.modal_handler_add(self)
            _set_header(context, _center_arc_header(self._stage, context.scene.su_draw_plane, self._typing, None, self._get_center_arc_measurement(context)))
            return {'RUNNING_MODAL'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed: {e}"); self._cleanup(context); return {'CANCELLED'}

    def _register_gpu(self, context):
        if not GPU_AVAILABLE:
            return
        prefs = _get_prefs(context)
        op = self
        self._draw_handler_key = f"acc_{id(self)}"
        def _ds():
            si = getattr(op, '_snap_info', None)
            if si and prefs and prefs.show_snap_marker:
                draw_snap_marker_simple(si, prefs)
        def _dp():
            c = getattr(op, '_center', None)
            rp = getattr(op, '_radius_point', None)
            ep = getattr(op, '_end_point', None)
            st = getattr(op, '_stage', 'CENTER')
            r = getattr(op, '_radius', 0.0)
            if st == "RADIUS" and c and rp:
                draw_preview_line_colored(c, rp, "FREE", False)
            if st == "ANGLE" and c and rp and ep and r > 1e-9:
                pn = _get_plane_normal(context.scene.su_draw_plane)
                a1 = _angle_on_plane(rp, c, pn)
                a2 = _angle_on_plane(ep, c, pn)
                diff = a2 - a1
                if diff > math.pi: diff -= 2*math.pi
                elif diff < -math.pi: diff += 2*math.pi
                seg = prefs.arc_segments if prefs else DEFAULT_ARC_SEGMENTS
                pp = _gen_arc_pts(c, r, a1, a1+diff, seg, pn)
                if len(pp) >= 2:
                    col = ARC_PREVIEW_COLOR; vs=[]; cs=[]
                    for ii in range(len(pp)-1):
                        vs.extend([pp[ii], pp[ii+1]]); cs.extend([col,col])
                    try:
                        sh = gpu.shader.from_builtin('FLAT_COLOR')
                        ba = batch_for_shader(sh, 'LINES', {"pos":vs, "color":cs})
                        gpu.state.blend_set('ALPHA'); gpu.state.line_width_set(2.5)
                        gpu.state.depth_test_set('LESS_EQUAL'); ba.draw(sh)
                        gpu.state.blend_set('NONE'); gpu.state.line_width_set(1.0)
                        gpu.state.depth_test_set('NONE')
                    except Exception: pass
                draw_preview_line_colored(c, rp, "FREE", False)
                draw_preview_line_colored(c, ep, "FREE", False)
        try:
            h1 = bpy.types.SpaceView3D.draw_handler_add(_ds, (), 'WINDOW', 'POST_PIXEL')
            h2 = bpy.types.SpaceView3D.draw_handler_add(_dp, (), 'WINDOW', 'POST_VIEW')
            _draw_handlers[self._draw_handler_key] = (h1, h2)
        except Exception: pass

    def _cleanup(self, context=None):
        if self._draw_handler_key:
            unregister_draw_handler(self._draw_handler_key); self._draw_handler_key = None
        if hasattr(self, "_vcb_handle") and self._vcb_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._vcb_handle, 'WINDOW'); self._vcb_handle = None
        hide_vcb()

    def _finish(self, context):
        _clear_header(context); self._cleanup(context); hide_vcb()

    def _commit_arc(self, context):
        if not self._center or not self._radius_point or not self._end_point or self._radius < 1e-9:
            self.report({'WARNING'}, "Invalid arc parameters"); return
        plane = context.scene.su_draw_plane
        pn = _get_plane_normal(plane)
        a1 = _angle_on_plane(self._radius_point, self._center, pn)
        a2 = _angle_on_plane(self._end_point, self._center, pn)
        diff = a2-a1
        if diff > math.pi: diff -= 2*math.pi
        elif diff < -math.pi: diff += 2*math.pi
        prefs = _get_prefs(context)
        seg = prefs.arc_segments if prefs else DEFAULT_ARC_SEGMENTS
        pts = _gen_arc_pts(self._center, self._radius, a1, a1+diff, seg, pn)
        if len(pts) < 2:
            self.report({'WARNING'}, "Arc too small"); return
        merge = prefs.merge_distance if prefs else MERGE_DISTANCE
        ei = [(i, i+1) for i in range(len(pts)-1)]
        commit_geometry_to_target(context, verts_world=pts, edges_idx=ei, merge_dist=merge)
        al = sum((pts[i+1]-pts[i]).length for i in range(len(pts)-1))
        self.report({'INFO'}, f"Center Arc: R={self._radius:.4f} L={al:.4f} A={abs(math.degrees(diff)):.1f}\u00b0")

    def modal(self, context, event):
        if context.area: context.area.tag_redraw()
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)
        if event.type in _NAV_EVENTS: return {'PASS_THROUGH'}
        if _handle_plane_keys(context, event):
            _set_header(context, _center_arc_header(self._stage, context.scene.su_draw_plane, self._typing, self._snap_info, self._get_center_arc_measurement(context)))
            return {'RUNNING_MODAL'}
        if event.type in {'RIGHTMOUSE','ESC'}:
            self._finish(context); return {'CANCELLED'}
        if event.value=='PRESS' and event.type in {'BACK_SPACE','DEL'}:
            if self._typing: self._typing = self._typing[:-1]
            elif self._stage=="ANGLE": self._stage="RADIUS"; self._end_point=None
            elif self._stage=="RADIUS": self._stage="CENTER"; self._radius_point=None
            _set_header(context, _center_arc_header(self._stage, context.scene.su_draw_plane, self._typing, self._snap_info, self._get_center_arc_measurement(context)))
            return {'RUNNING_MODAL'}
        if event.value=='PRESS' and event.type=='RET':
            if self._stage=="RADIUS" and self._typing and self._center:
                r = _parse_length(context, self._typing)
                if r and r > 0:
                    self._radius = float(r)
                    if not self._radius_point:
                        ax_a, ax_b = _plane_axes(context.scene.su_draw_plane)
                        self._radius_point = self._center + ax_a * self._radius
                    else:
                        d = self._radius_point - self._center
                        if d.length > 1e-9: self._radius_point = self._center + d.normalized()*self._radius
                    self._stage = "ANGLE"
                self._typing = ""
                _set_header(context, _center_arc_header(self._stage, context.scene.su_draw_plane, self._typing, self._snap_info, self._get_center_arc_measurement(context)))
                return {'RUNNING_MODAL'}
            if self._stage=="ANGLE" and self._typing:
                av = _parse_angle(self._typing)
                if av is not None and self._center and self._radius_point:
                    pn = _get_plane_normal(context.scene.su_draw_plane)
                    a1 = _angle_on_plane(self._radius_point, self._center, pn)
                    n2 = pn.normalized()
                    if abs(n2.z)>0.9: ref=Vector((1,0,0))
                    else: ref=Vector((0,0,1))
                    u = n2.cross(ref).normalized(); v = n2.cross(u).normalized()
                    self._end_point = self._center + u*self._radius*math.cos(a1+av) + v*self._radius*math.sin(a1+av)
                    self._commit_arc(context); self._finish(context); return {'FINISHED'}
                self._typing = ""
                return {'RUNNING_MODAL'}
            if self._stage=="ANGLE" and self._center and self._radius_point and self._end_point:
                self._commit_arc(context); self._finish(context); return {'FINISHED'}
            return {'RUNNING_MODAL'}
        if event.value=='PRESS' and len(event.unicode)==1:
            ch = event.unicode
            if ch.isdigit() or ch in ".-+mMcCiInNfFtT\u00b0dD ":
                self._typing += ch
                _set_header(context, _center_arc_header(self._stage, context.scene.su_draw_plane, self._typing, self._snap_info, self._get_center_arc_measurement(context)))
                return {'RUNNING_MODAL'}
        try:
            if self._stage=="ANGLE" and self._radius > 1e-9:
                ad = 0.0
                if self._center and self._radius_point and self._end_point:
                    pn = _get_plane_normal(context.scene.su_draw_plane)
                    a1 = _angle_on_plane(self._radius_point, self._center, pn)
                    a2 = _angle_on_plane(self._end_point, self._center, pn)
                    d = a2-a1
                    if d>math.pi: d-=2*math.pi
                    elif d<-math.pi: d+=2*math.pi
                    ad = abs(math.degrees(d))
                update_vcb(label="Center Arc", value=f"R:{_format_length(context,self._radius)}  A:{ad:.1f}\u00b0", typed=self._typing, visible=True)
            elif self._stage=="RADIUS" and self._center and self._radius_point:
                update_vcb(label="Radius", value=_format_length(context, self._radius), typed=self._typing, visible=True)
            else:
                update_vcb(label="Center Arc", value="", typed=self._typing, visible=True)
        except Exception: pass
        plane = context.scene.su_draw_plane
        offset = context.scene.su_plane_offset
        hit = get_plane_hit(context, event, plane=plane, offset=offset)
        if hit is None: return {'RUNNING_MODAL'}
        self._snap_info = None
        if context.scene.su_enable_vertex_snap:
            tn = _get_target_name(context)
            sc, si = find_vertex_snap(context, event, hit, self._kd, self._meta, self._depsgraph, max_px=float(context.scene.su_snap_px), max_world=MAX_SNAP_DISTANCE_WORLD, target_name=tn)
            if sc is not None: hit=sc; self._snap_info=si
        if self._snap_info is None:
            pr = _get_prefs(context)
            if pr and pr.grid_snap_enabled: hit = snap_to_grid(hit, pr.grid_size)
        if self._stage=="ANGLE" and self._center:
            self._end_point = hit.copy()
            _set_header(context, _center_arc_header(self._stage, context.scene.su_draw_plane, self._typing, self._snap_info, self._get_center_arc_measurement(context)))
        elif self._stage=="RADIUS" and self._center:
            self._radius_point = hit.copy()
            self._radius = (hit-self._center).length
            _set_header(context, _center_arc_header(self._stage, context.scene.su_draw_plane, self._typing, self._snap_info, self._get_center_arc_measurement(context)))
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            if self._stage=="CENTER":
                self._center = hit.copy(); self._stage = "RADIUS"
            elif self._stage=="RADIUS":
                self._radius_point = hit.copy(); self._radius = (hit-self._center).length
                if self._radius < 1e-9:
                    self.report({'WARNING'}, "Radius too small"); return {'RUNNING_MODAL'}
                self._stage = "ANGLE"
            elif self._stage=="ANGLE":
                self._end_point = hit.copy()
                self._commit_arc(context); self._finish(context); return {'FINISHED'}
            _set_header(context, _center_arc_header(self._stage, context.scene.su_draw_plane, self._typing, self._snap_info, self._get_center_arc_measurement(context)))
        return {'RUNNING_MODAL'}


# =============================================================================
# TANGENT ARC TOOL
# =============================================================================
class SU_OT_arc_tangent(bpy.types.Operator):
    """Tangent arc (fillet) between two selected edges"""
    bl_idname = "su.arc_tangent"
    bl_label = "Tangent Arc (SketchUp)"
    bl_options = {'REGISTER', 'UNDO'}

    def _get_tangent_arc_measurement(self, context):
        """Get tangent arc measurements."""
        if self._shared_vert and self._edge1 and self._edge2 and self._radius > 1e-9:
            pn = _get_plane_normal(context.scene.su_draw_plane)
            mw = self._obj.matrix_world
            vw = mw @ self._shared_vert.co
            v1w = mw @ self._edge1.other_vert(self._shared_vert).co
            v2w = mw @ self._edge2.other_vert(self._shared_vert).co
            prefs = _get_prefs(context)
            seg = prefs.arc_segments if prefs else DEFAULT_ARC_SEGMENTS
            pts, _, _ = _tangent_arc_pts(vw, v1w, v2w, self._radius, pn, seg)
            if len(pts) >= 2:
                arc_len = sum((pts[i+1]-pts[i]).length for i in range(len(pts)-1))
                unit_system = context.scene.unit_settings.system
                rad_str = _format_measurement(self._radius, unit_system)
                len_str = _format_measurement(arc_len, unit_system)
                return f"R:{rad_str}  L:{len_str}"
        return ""

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and context.mode in {'EDIT_MESH', 'OBJECT'}

    def invoke(self, context, event):
        self._typing = ""; self._radius = 0.5; self._snap_info = None
        self._kd = None; self._meta = None; self._depsgraph = None
        self._frame_count = 0; self._draw_handler_key = None; self._vcb_handle = None
        self._shared_vert = None; self._edge1 = None; self._edge2 = None
        self._preview_points = []; self._obj = None
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'WARNING'}, "Select mesh"); return {'CANCELLED'}
        if context.mode == 'OBJECT':
            obj.select_set(True); context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
        if context.mode != 'EDIT_MESH':
            self.report({'WARNING'}, "Edit Mode required"); return {'CANCELLED'}
        bm = bmesh.from_edit_mesh(obj.data)
        sel_e = [e for e in bm.edges if e.select]
        if len(sel_e) != 2:
            self.report({'WARNING'}, "Select exactly 2 edges sharing a vertex"); return {'CANCELLED'}
        shared = None
        for v in sel_e[0].verts:
            if v in sel_e[1].verts: shared = v; break
        if not shared:
            self.report({'WARNING'}, "Edges must share a vertex"); return {'CANCELLED'}
        self._shared_vert = shared; self._edge1 = sel_e[0]; self._edge2 = sel_e[1]; self._obj = obj
        plane = context.scene.su_draw_plane; pn = _get_plane_normal(plane); mw = obj.matrix_world
        vw = mw@shared.co; v1w = mw@sel_e[0].other_vert(shared).co; v2w = mw@sel_e[1].other_vert(shared).co
        d1 = v1w-vw; d2 = v2w-vw; cr = d1.cross(d2)
        if cr.length < 1e-6:
            ci = {'XY':2,'XZ':1,'YZ':0}.get(plane,2)
            if abs(d1[ci])>1e-4 or abs(d2[ci])>1e-4:
                self.report({'WARNING'}, "Edges not coplanar"); return {'CANCELLED'}
        try:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)
            self._register_gpu(context)
            self._vcb_handle = bpy.types.SpaceView3D.draw_handler_add(draw_vcb_box, (), 'WINDOW', 'POST_PIXEL')
            context.window_manager.modal_handler_add(self)
            _set_header(context, f"Tangent Arc | Type radius+Enter | LMB: Confirm | {self._get_tangent_arc_measurement(context)}")
            self._update_preview(context)
            return {'RUNNING_MODAL'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed: {e}"); self._cleanup(context); return {'CANCELLED'}

    def _register_gpu(self, context):
        if not GPU_AVAILABLE: return
        prefs = _get_prefs(context); op = self
        self._draw_handler_key = f"at_{id(self)}"
        def _dp():
            pp = getattr(op, '_preview_points', [])
            if len(pp) >= 2:
                col = ARC_TANGENT_COLOR; vs=[]; cs=[]
                for ii in range(len(pp)-1):
                    vs.extend([pp[ii],pp[ii+1]]); cs.extend([col,col])
                try:
                    sh = gpu.shader.from_builtin('FLAT_COLOR')
                    ba = batch_for_shader(sh, 'LINES', {"pos":vs,"color":cs})
                    gpu.state.blend_set('ALPHA'); gpu.state.line_width_set(3.0)
                    gpu.state.depth_test_set('LESS_EQUAL'); ba.draw(sh)
                    gpu.state.blend_set('NONE'); gpu.state.line_width_set(1.0)
                    gpu.state.depth_test_set('NONE')
                except Exception: pass
        try:
            h1 = bpy.types.SpaceView3D.draw_handler_add(_dp, (), 'WINDOW', 'POST_VIEW')
            _draw_handlers[self._draw_handler_key] = (h1, 'WINDOW', 'POST_VIEW')
        except Exception: pass

    def _cleanup(self, context=None):
        if self._draw_handler_key:
            unregister_draw_handler(self._draw_handler_key); self._draw_handler_key = None
        if hasattr(self, "_vcb_handle") and self._vcb_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._vcb_handle, 'WINDOW'); self._vcb_handle = None
        hide_vcb()

    def _finish(self, context):
        _clear_header(context); self._cleanup(context); hide_vcb()

    def _update_preview(self, context):
        if not self._shared_vert or not self._edge1 or not self._edge2:
            self._preview_points = []; return
        if not self._shared_vert.is_valid or not self._edge1.is_valid or not self._edge2.is_valid:
            self._preview_points = []; return
        pn = _get_plane_normal(context.scene.su_draw_plane)
        mw = self._obj.matrix_world
        vw = mw@self._shared_vert.co
        v1w = mw@self._edge1.other_vert(self._shared_vert).co
        v2w = mw@self._edge2.other_vert(self._shared_vert).co
        prefs = _get_prefs(context)
        seg = prefs.arc_segments if prefs else DEFAULT_ARC_SEGMENTS
        pts, _, _ = _tangent_arc_pts(vw, v1w, v2w, self._radius, pn, seg)
        self._preview_points = pts

    def _commit_tangent(self, context):
        if not self._shared_vert or not self._edge1 or not self._edge2: return
        if not self._shared_vert.is_valid: return
        pn = _get_plane_normal(context.scene.su_draw_plane)
        mw = self._obj.matrix_world; inv = mw.inverted()
        vw = mw@self._shared_vert.co
        v1w = mw@self._edge1.other_vert(self._shared_vert).co
        v2w = mw@self._edge2.other_vert(self._shared_vert).co
        prefs = _get_prefs(context)
        seg = prefs.arc_segments if prefs else DEFAULT_ARC_SEGMENTS
        pts, pt1, pt2 = _tangent_arc_pts(vw, v1w, v2w, self._radius, pn, seg)
        if len(pts) < 2 or pt1 is None or pt2 is None:
            self.report({'WARNING'}, "Cannot create tangent arc"); return
        merge = prefs.merge_distance if prefs else MERGE_DISTANCE
        ei = [(i,i+1) for i in range(len(pts)-1)]
        commit_geometry_to_target(context, verts_world=pts, edges_idx=ei, merge_dist=merge)
        bm = bmesh.from_edit_mesh(self._obj.data)
        bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table()
        for edge_ref, tangent_pt in [(self._edge1, pt1), (self._edge2, pt2)]:
            if edge_ref.is_valid:
                pt_l = inv @ tangent_pt
                ev = edge_ref.verts[1].co - edge_ref.verts[0].co
                if ev.length > 1e-9:
                    t = (pt_l - edge_ref.verts[0].co).dot(ev) / ev.length_squared
                    t = max(0.01, min(0.99, t))
                    if 0.01 < t < 0.99:
                        try: bmesh.utils.edge_split(edge_ref, edge_ref.verts[0], t)
                        except Exception: pass
            bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table()
        bmesh.update_edit_mesh(self._obj.data, loop_triangles=True, destructive=True)
        _kdtree_cache.invalidate()
        al = sum((pts[i+1]-pts[i]).length for i in range(len(pts)-1))
        self.report({'INFO'}, f"Tangent Arc: R={self._radius:.4f} L={al:.4f}")

    def modal(self, context, event):
        if context.area: context.area.tag_redraw()
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)
        if event.type in _NAV_EVENTS: return {'PASS_THROUGH'}
        if event.type in {'RIGHTMOUSE','ESC'}: self._finish(context); return {'CANCELLED'}
        if event.value=='PRESS' and event.type in {'BACK_SPACE','DEL'}:
            self._typing = self._typing[:-1]
            _set_header(context, f"Tangent Arc | Type: {self._typing} | {self._get_tangent_arc_measurement(context)}")
            return {'RUNNING_MODAL'}
        if event.value=='PRESS' and event.type=='RET':
            if self._typing:
                r = _parse_length(context, self._typing)
                if r and r > 0: self._radius = float(r); self._update_preview(context)
                self._typing = ""
                _set_header(context, f"Tangent Arc | {self._get_tangent_arc_measurement(context)} | LMB: Confirm")
                return {'RUNNING_MODAL'}
            self._commit_tangent(context); self._finish(context); return {'FINISHED'}
        if event.value=='PRESS' and len(event.unicode)==1:
            ch = event.unicode
            if ch.isdigit() or ch in ".-+mMcCiInNfFtT\"\' ":
                self._typing += ch
                _set_header(context, f"Tangent Arc | Type: {self._typing} | {self._get_tangent_arc_measurement(context)}")
                return {'RUNNING_MODAL'}
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            self._commit_tangent(context); self._finish(context); return {'FINISHED'}
        try:
            update_vcb(label="Tangent R", value=_format_length(context, self._radius), typed=self._typing, visible=True)
        except Exception: pass
        if event.type=='MOUSEMOVE' and not self._typing:
            hit = get_plane_hit(context, event, plane=context.scene.su_draw_plane, offset=context.scene.su_plane_offset)
            if hit and self._shared_vert and self._shared_vert.is_valid:
                vw = self._obj.matrix_world @ self._shared_vert.co
                nr = (hit-vw).length
                if nr > 1e-9:
                    self._radius = nr; self._update_preview(context)
                    _set_header(context, f"Tangent Arc | {self._get_tangent_arc_measurement(context)} | LMB/Enter: Confirm")
        return {'RUNNING_MODAL'}


# =============================================================================
# EDGE-CONNECT ARC TOOL
# =============================================================================
class SU_OT_arc_edge_connect(bpy.types.Operator):
    """Arc between two picked vertices, constrained to plane"""
    bl_idname = "su.arc_edge_connect"
    bl_label = "Edge-Connect Arc (SketchUp)"
    bl_options = {'REGISTER', 'UNDO'}

    def _get_edge_connect_measurement(self, context):
        """Get edge-connect arc measurements."""
        if self._stage == "BULGE" and self._v1 and self._v2 and self._bulge:
            pn = _get_plane_normal(context.scene.su_draw_plane)
            c, r = _circle_from_3pts(self._v1, self._v2, self._bulge, pn)
            if c and r and r > 1e-9:
                a1 = _angle_on_plane(self._v1, c, pn)
                ab = _angle_on_plane(self._bulge, c, pn)
                a2 = _angle_on_plane(self._v2, c, pn)
                angs = _order_arc_angles(a1, ab, a2)
                if angs:
                    sa, ea = angs
                    seg = _get_prefs(context).arc_segments if _get_prefs(context) else DEFAULT_ARC_SEGMENTS
                    pts = _gen_arc_pts(c, r, sa, ea, seg, pn)
                    if len(pts) >= 2:
                        arc_len = sum((pts[i+1]-pts[i]).length for i in range(len(pts)-1))
                        angle_deg = math.degrees(abs(ea-sa))
                        unit_system = context.scene.unit_settings.system
                        rad_str = _format_measurement(r, unit_system)
                        len_str = _format_measurement(arc_len, unit_system)
                        return f"R:{rad_str}  L:{len_str}  A:{angle_deg:.1f}°"
        elif self._stage == "PICK2" and self._v1 and self._v2:
            chord_len = (self._v2 - self._v1).length
            if chord_len > 1e-9:
                unit_system = context.scene.unit_settings.system
                chord_str = _format_measurement(chord_len, unit_system)
                return f"Chord: {chord_str}"
        return ""

    def invoke(self, context, event):
        self._stage = "PICK1"
        self._v1 = None; self._v2 = None; self._bulge = None
        self._typing = ""
        self._kd = None; self._meta = None; self._depsgraph = None
        self._snap_info = None; self._frame_count = 0
        self._draw_handler_key = None; self._vcb_handle = None
        if context.space_data.type != 'VIEW_3D':
            self.report({'WARNING'}, "Must be in 3D View"); return {'CANCELLED'}
        try:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)
            self._register_gpu(context)
            self._vcb_handle = bpy.types.SpaceView3D.draw_handler_add(draw_vcb_box, (), 'WINDOW', 'POST_PIXEL')
            context.window_manager.modal_handler_add(self)
            _set_header(context, "Edge-Connect Arc: Click first vertex | Esc: Cancel")
            return {'RUNNING_MODAL'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed: {e}"); self._cleanup(context); return {'CANCELLED'}

    def _register_gpu(self, context):
        if not GPU_AVAILABLE: return
        prefs = _get_prefs(context); op = self
        self._draw_handler_key = f"ae_{id(self)}"
        def _ds():
            si = getattr(op, '_snap_info', None)
            if si and prefs and prefs.show_snap_marker: draw_snap_marker_simple(si, prefs)
        def _dp():
            v1 = getattr(op, '_v1', None); v2 = getattr(op, '_v2', None)
            bp = getattr(op, '_bulge', None); st = getattr(op, '_stage', 'PICK1')
            if st == "PICK2" and v1 and v2:
                draw_preview_line_colored(v1, v2, "FREE", False)
            if st == "BULGE" and v1 and v2 and bp:
                pn = _get_plane_normal(context.scene.su_draw_plane)
                c, r = _circle_from_3pts(v1, v2, bp, pn)
                if c and r and r > 1e-9:
                    a1 = _angle_on_plane(v1, c, pn)
                    ab = _angle_on_plane(bp, c, pn)
                    a2 = _angle_on_plane(v2, c, pn)
                    angs = _order_arc_angles(a1, ab, a2)
                    if angs:
                        sa, ea = angs
                        seg = prefs.arc_segments if prefs else DEFAULT_ARC_SEGMENTS
                        pp = _gen_arc_pts(c, r, sa, ea, seg, pn)
                        if len(pp)>=2:
                            col=ARC_PREVIEW_COLOR; vs=[]; cs=[]
                            for ii in range(len(pp)-1):
                                vs.extend([pp[ii],pp[ii+1]]); cs.extend([col,col])
                            try:
                                sh=gpu.shader.from_builtin('FLAT_COLOR')
                                ba=batch_for_shader(sh,'LINES',{"pos":vs,"color":cs})
                                gpu.state.blend_set('ALPHA'); gpu.state.line_width_set(2.5)
                                gpu.state.depth_test_set('LESS_EQUAL'); ba.draw(sh)
                                gpu.state.blend_set('NONE'); gpu.state.line_width_set(1.0)
                                gpu.state.depth_test_set('NONE')
                            except Exception: pass
        try:
            h1 = bpy.types.SpaceView3D.draw_handler_add(_ds, (), 'WINDOW', 'POST_PIXEL')
            h2 = bpy.types.SpaceView3D.draw_handler_add(_dp, (), 'WINDOW', 'POST_VIEW')
            _draw_handlers[self._draw_handler_key] = (h1, h2)
        except Exception: pass

    def _cleanup(self, context=None):
        if self._draw_handler_key:
            unregister_draw_handler(self._draw_handler_key); self._draw_handler_key = None
        if hasattr(self, "_vcb_handle") and self._vcb_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._vcb_handle, 'WINDOW'); self._vcb_handle = None
        hide_vcb()

    def _finish(self, context):
        _clear_header(context); self._cleanup(context); hide_vcb()

    def _commit(self, context):
        if not self._v1 or not self._v2 or not self._bulge: return
        pn = _get_plane_normal(context.scene.su_draw_plane)
        c, r = _circle_from_3pts(self._v1, self._v2, self._bulge, pn)
        if not c or not r or r < 1e-9:
            self.report({'WARNING'}, "Cannot compute arc"); return
        a1 = _angle_on_plane(self._v1, c, pn)
        ab = _angle_on_plane(self._bulge, c, pn)
        a2 = _angle_on_plane(self._v2, c, pn)
        angs = _order_arc_angles(a1, ab, a2)
        if not angs: self.report({'WARNING'}, "Bad arc"); return
        sa, ea = angs
        prefs = _get_prefs(context)
        seg = prefs.arc_segments if prefs else DEFAULT_ARC_SEGMENTS
        pts = _gen_arc_pts(c, r, sa, ea, seg, pn)
        if len(pts) < 2: self.report({'WARNING'}, "Arc too small"); return
        merge = prefs.merge_distance if prefs else MERGE_DISTANCE
        ei = [(i,i+1) for i in range(len(pts)-1)]
        commit_geometry_to_target(context, verts_world=pts, edges_idx=ei, merge_dist=merge)
        al = sum((pts[i+1]-pts[i]).length for i in range(len(pts)-1))
        self.report({'INFO'}, f"EC Arc: R={r:.4f} L={al:.4f} A={math.degrees(abs(ea-sa)):.1f}\u00b0")

    def modal(self, context, event):
        if context.area: context.area.tag_redraw()
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)
        if event.type in _NAV_EVENTS: return {'PASS_THROUGH'}
        if event.type in {'RIGHTMOUSE','ESC'}: self._finish(context); return {'CANCELLED'}
        if event.value=='PRESS' and event.type in {'BACK_SPACE','DEL'}:
            if self._typing: self._typing = self._typing[:-1]
            elif self._stage=="BULGE": self._stage="PICK2"; self._bulge=None
            elif self._stage=="PICK2": self._stage="PICK1"; self._v2=None
            return {'RUNNING_MODAL'}
        if event.value=='PRESS' and event.type=='RET':
            if self._stage=="BULGE":
                self._commit(context); self._finish(context); return {'FINISHED'}
            return {'RUNNING_MODAL'}
        if event.value=='PRESS' and len(event.unicode)==1:
            ch = event.unicode
            if ch.isdigit() or ch in ".-+mMcCiInNfFtT ":
                self._typing += ch; return {'RUNNING_MODAL'}
        try:
            if self._stage=="BULGE" and self._v1 and self._v2 and self._bulge:
                pn = _get_plane_normal(context.scene.su_draw_plane)
                c, r = _circle_from_3pts(self._v1, self._v2, self._bulge, pn)
                if c and r:
                    update_vcb(label="EC Arc R", value=_format_length(context, r), typed=self._typing, visible=True)
                else:
                    update_vcb(label="EC Arc", value="", typed=self._typing, visible=True)
            else:
                update_vcb(label="EC Arc", value="", typed=self._typing, visible=True)
        except Exception: pass
        plane = context.scene.su_draw_plane; offset = context.scene.su_plane_offset
        hit = get_plane_hit(context, event, plane=plane, offset=offset)
        if hit is None: return {'RUNNING_MODAL'}
        self._snap_info = None
        if context.scene.su_enable_vertex_snap:
            tn = _get_target_name(context)
            sc, si = find_vertex_snap(context, event, hit, self._kd, self._meta, self._depsgraph, max_px=float(context.scene.su_snap_px), max_world=MAX_SNAP_DISTANCE_WORLD, target_name=tn)
            if sc is not None: hit=sc; self._snap_info=si
        if self._snap_info is None:
            pr = _get_prefs(context)
            if pr and pr.grid_snap_enabled: hit = snap_to_grid(hit, pr.grid_size)
        if self._stage=="BULGE": 
            self._bulge = hit.copy()
            _set_header(context, f"EC Arc: Set bulge | {self._get_edge_connect_measurement(context)} | LMB/Enter: Confirm | Esc: Cancel")
        elif self._stage=="PICK2" and self._v1: 
            self._v2 = hit.copy()
            _set_header(context, f"EC Arc: Click second vertex | {self._get_edge_connect_measurement(context)} | Esc: Cancel")
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            if self._stage=="PICK1":
                self._v1 = hit.copy(); self._stage="PICK2"
                _set_header(context, f"EC Arc: Click second vertex | {self._get_edge_connect_measurement(context)} | Esc: Cancel")
            elif self._stage=="PICK2":
                self._v2 = hit.copy(); self._stage="BULGE"
                _set_header(context, f"EC Arc: Set bulge | {self._get_edge_connect_measurement(context)} | LMB/Enter: Confirm | Esc: Cancel")
            elif self._stage=="BULGE":
                self._bulge = hit.copy()
                self._commit(context); self._finish(context); return {'FINISHED'}
        return {'RUNNING_MODAL'}


# =============================================================================
# AUTO-FILLET ARC TOOL
# =============================================================================
class SU_OT_arc_fillet(bpy.types.Operator):
    """Replace selected vertex corner with fillet arc of given radius"""
    bl_idname = "su.arc_fillet"
    bl_label = "Auto-Fillet Arc (SketchUp)"
    bl_options = {'REGISTER', 'UNDO'}

    def _get_fillet_arc_measurement(self, context):
        """Get fillet arc measurements."""
        if self._target_vert and self._edge1 and self._edge2 and self._radius > 1e-9:
            pn = _get_plane_normal(context.scene.su_draw_plane)
            mw = self._obj.matrix_world
            vw = mw @ self._target_vert.co
            v1w = mw @ self._edge1.other_vert(self._target_vert).co
            v2w = mw @ self._edge2.other_vert(self._target_vert).co
            prefs = _get_prefs(context)
            seg = prefs.arc_segments if prefs else DEFAULT_ARC_SEGMENTS
            pts, _, _ = _tangent_arc_pts(vw, v1w, v2w, self._radius, pn, seg)
            if len(pts) >= 2:
                arc_len = sum((pts[i+1]-pts[i]).length for i in range(len(pts)-1))
                unit_system = context.scene.unit_settings.system
                rad_str = _format_measurement(self._radius, unit_system)
                len_str = _format_measurement(arc_len, unit_system)
                return f"R:{rad_str}  L:{len_str}"
        return ""

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and context.mode in {'EDIT_MESH', 'OBJECT'}

    def invoke(self, context, event):
        self._typing = ""; self._radius = 0.2; self._snap_info = None
        self._kd = None; self._meta = None; self._depsgraph = None
        self._frame_count = 0; self._draw_handler_key = None; self._vcb_handle = None
        self._target_vert = None; self._edge1 = None; self._edge2 = None
        self._preview_points = []; self._obj = None
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'WARNING'}, "Select mesh"); return {'CANCELLED'}
        if context.mode == 'OBJECT':
            obj.select_set(True); context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
        if context.mode != 'EDIT_MESH':
            self.report({'WARNING'}, "Edit Mode required"); return {'CANCELLED'}
        bm = bmesh.from_edit_mesh(obj.data)
        sv = [v for v in bm.verts if v.select]
        if len(sv) != 1:
            self.report({'WARNING'}, "Select exactly 1 vertex"); return {'CANCELLED'}
        vert = sv[0]
        if len(vert.link_edges) < 2:
            self.report({'WARNING'}, "Vertex needs 2+ edges"); return {'CANCELLED'}
        edges = list(vert.link_edges)
        self._target_vert = vert; self._edge1 = edges[0]; self._edge2 = edges[1]; self._obj = obj
        pn = _get_plane_normal(context.scene.su_draw_plane); mw = obj.matrix_world
        vw = mw@vert.co; v1w = mw@edges[0].other_vert(vert).co; v2w = mw@edges[1].other_vert(vert).co
        d1 = v1w-vw; d2 = v2w-vw; cr = d1.cross(d2)
        if cr.length < 1e-6:
            ci = {'XY':2,'XZ':1,'YZ':0}.get(context.scene.su_draw_plane, 2)
            if abs(d1[ci])>1e-4 or abs(d2[ci])>1e-4:
                self.report({'WARNING'}, "Edges not coplanar"); return {'CANCELLED'}
        try:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)
            self._register_gpu(context)
            self._vcb_handle = bpy.types.SpaceView3D.draw_handler_add(draw_vcb_box, (), 'WINDOW', 'POST_PIXEL')
            context.window_manager.modal_handler_add(self)
            _set_header(context, f"Fillet | Type radius+Enter | LMB: Confirm | {self._get_fillet_arc_measurement(context)}")
            self._update_preview(context)
            return {'RUNNING_MODAL'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed: {e}"); self._cleanup(context); return {'CANCELLED'}

    def _register_gpu(self, context):
        if not GPU_AVAILABLE: return
        prefs = _get_prefs(context); op = self
        self._draw_handler_key = f"af_{id(self)}"
        def _dp():
            pp = getattr(op, '_preview_points', [])
            if len(pp)>=2:
                col=ARC_PREVIEW_COLOR; vs=[]; cs=[]
                for ii in range(len(pp)-1):
                    vs.extend([pp[ii],pp[ii+1]]); cs.extend([col,col])
                try:
                    sh=gpu.shader.from_builtin('FLAT_COLOR')
                    ba=batch_for_shader(sh,'LINES',{"pos":vs,"color":cs})
                    gpu.state.blend_set('ALPHA'); gpu.state.line_width_set(3.0)
                    gpu.state.depth_test_set('LESS_EQUAL'); ba.draw(sh)
                    gpu.state.blend_set('NONE'); gpu.state.line_width_set(1.0)
                    gpu.state.depth_test_set('NONE')
                except Exception: pass
        try:
            h1 = bpy.types.SpaceView3D.draw_handler_add(_dp, (), 'WINDOW', 'POST_VIEW')
            _draw_handlers[self._draw_handler_key] = (h1, 'WINDOW', 'POST_VIEW')
        except Exception: pass

    def _cleanup(self, context=None):
        if self._draw_handler_key:
            unregister_draw_handler(self._draw_handler_key); self._draw_handler_key = None
        if hasattr(self, "_vcb_handle") and self._vcb_handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._vcb_handle, 'WINDOW'); self._vcb_handle = None
        hide_vcb()

    def _finish(self, context):
        _clear_header(context); self._cleanup(context); hide_vcb()

    def _update_preview(self, context):
        if not self._target_vert or not self._edge1 or not self._edge2:
            self._preview_points = []; return
        if not self._target_vert.is_valid or not self._edge1.is_valid or not self._edge2.is_valid:
            self._preview_points = []; return
        pn = _get_plane_normal(context.scene.su_draw_plane)
        mw = self._obj.matrix_world
        vw = mw@self._target_vert.co
        v1w = mw@self._edge1.other_vert(self._target_vert).co
        v2w = mw@self._edge2.other_vert(self._target_vert).co
        prefs = _get_prefs(context)
        seg = prefs.arc_segments if prefs else DEFAULT_ARC_SEGMENTS
        pts, _, _ = _tangent_arc_pts(vw, v1w, v2w, self._radius, pn, seg)
        self._preview_points = pts

    def _commit_fillet(self, context):
        if not self._target_vert or not self._edge1 or not self._edge2: return
        if not self._target_vert.is_valid: self.report({'WARNING'}, "Vertex invalid"); return
        pn = _get_plane_normal(context.scene.su_draw_plane)
        mw = self._obj.matrix_world; inv = mw.inverted()
        vw = mw@self._target_vert.co
        v1w = mw@self._edge1.other_vert(self._target_vert).co
        v2w = mw@self._edge2.other_vert(self._target_vert).co
        prefs = _get_prefs(context)
        seg = prefs.arc_segments if prefs else DEFAULT_ARC_SEGMENTS
        pts, pt1, pt2 = _tangent_arc_pts(vw, v1w, v2w, self._radius, pn, seg)
        if len(pts) < 2 or pt1 is None or pt2 is None:
            self.report({'WARNING'}, "Cannot fillet"); return
        bm = bmesh.from_edit_mesh(self._obj.data)
        bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table()
        pt1_l = inv @ pt1; pt2_l = inv @ pt2
        nv1 = None; nv2 = None
        if self._edge1.is_valid:
            ev = self._edge1.verts[1].co - self._edge1.verts[0].co
            if ev.length > 1e-9:
                t = (pt1_l - self._edge1.verts[0].co).dot(ev) / ev.length_squared
                t = max(0.01, min(0.99, t))
                if 0.01 < t < 0.99:
                    try:
                        ne, nv1 = bmesh.utils.edge_split(self._edge1, self._edge1.verts[0], t)
                    except Exception: pass
        bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table()
        if self._edge2.is_valid:
            ev = self._edge2.verts[1].co - self._edge2.verts[0].co
            if ev.length > 1e-9:
                t = (pt2_l - self._edge2.verts[0].co).dot(ev) / ev.length_squared
                t = max(0.01, min(0.99, t))
                if 0.01 < t < 0.99:
                    try:
                        ne, nv2 = bmesh.utils.edge_split(self._edge2, self._edge2.verts[0], t)
                    except Exception: pass
        bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table()
        bmesh.update_edit_mesh(self._obj.data, loop_triangles=False, destructive=False)
        merge = prefs.merge_distance if prefs else MERGE_DISTANCE
        ei = [(i,i+1) for i in range(len(pts)-1)]
        commit_geometry_to_target(context, verts_world=pts, edges_idx=ei, merge_dist=merge)
        # Remove old corner edges and vertex
        try:
            bm2 = bmesh.from_edit_mesh(self._obj.data)
            bm2.verts.ensure_lookup_table(); bm2.edges.ensure_lookup_table()
            # Find the original corner vert by proximity
            corner_local = inv @ vw
            corner_v = None
            for v in bm2.verts:
                if v.is_valid and (v.co - corner_local).length < merge * 10:
                    corner_v = v
                    break
            if corner_v and corner_v.is_valid:
                # Remove edges from corner to the tangent points
                edges_to_kill = []
                for e in list(corner_v.link_edges):
                    if e.is_valid:
                        ov = e.other_vert(corner_v)
                        ov_w = self._obj.matrix_world @ ov.co
                        if pt1 and (ov_w - pt1).length < merge * 10:
                            edges_to_kill.append(e)
                        elif pt2 and (ov_w - pt2).length < merge * 10:
                            edges_to_kill.append(e)
                for e in edges_to_kill:
                    if e.is_valid:
                        try: bm2.edges.remove(e)
                        except Exception: pass
                bm2.verts.ensure_lookup_table()
                if corner_v.is_valid and len(corner_v.link_edges) == 0:
                    try: bm2.verts.remove(corner_v)
                    except Exception: pass
            bmesh.update_edit_mesh(self._obj.data, loop_triangles=True, destructive=True)
        except Exception:
            pass
        _kdtree_cache.invalidate()
        al = sum((pts[i+1]-pts[i]).length for i in range(len(pts)-1))
        self.report({'INFO'}, f"Fillet: R={self._radius:.4f} L={al:.4f}")

    def modal(self, context, event):
        if context.area: context.area.tag_redraw()
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)
        if event.type in _NAV_EVENTS: return {'PASS_THROUGH'}
        if event.type in {'RIGHTMOUSE','ESC'}: self._finish(context); return {'CANCELLED'}
        if event.value=='PRESS' and event.type in {'BACK_SPACE','DEL'}:
            self._typing = self._typing[:-1]
            _set_header(context, f"Fillet | Type: {self._typing} | {self._get_fillet_arc_measurement(context)}")
            return {'RUNNING_MODAL'}
        if event.value=='PRESS' and event.type=='RET':
            if self._typing:
                r = _parse_length(context, self._typing)
                if r and r > 0: self._radius = float(r); self._update_preview(context)
                self._typing = ""
                _set_header(context, f"Fillet | {self._get_fillet_arc_measurement(context)} | LMB: Confirm")
                return {'RUNNING_MODAL'}
            self._commit_fillet(context); self._finish(context); return {'FINISHED'}
        if event.value=='PRESS' and len(event.unicode)==1:
            ch = event.unicode
            if ch.isdigit() or ch in ".-+mMcCiInNfFtT\"\' ":
                self._typing += ch
                _set_header(context, f"Fillet | Type: {self._typing} | {self._get_fillet_arc_measurement(context)}")
                return {'RUNNING_MODAL'}
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            self._commit_fillet(context); self._finish(context); return {'FINISHED'}
        try:
            update_vcb(label="Fillet R", value=_format_length(context, self._radius), typed=self._typing, visible=True)
        except Exception: pass
        if event.type=='MOUSEMOVE' and not self._typing:
            hit = get_plane_hit(context, event, plane=context.scene.su_draw_plane, offset=context.scene.su_plane_offset)
            if hit and self._target_vert and self._target_vert.is_valid:
                vw = self._obj.matrix_world @ self._target_vert.co
                nr = (hit-vw).length
                if nr > 1e-9:
                    self._radius = nr; self._update_preview(context)
                    _set_header(context, f"Fillet | {self._get_fillet_arc_measurement(context)} | LMB/Enter: Confirm")
        return {'RUNNING_MODAL'}


# =============================================================================
# PUSH/PULL TOOL  (works on target mesh faces)
# =============================================================================
def _pp_header_status(axis_lock, typing, snap_info, typing_mode=False):
    axis = axis_lock if axis_lock and axis_lock != "FREE" else "FREE"
    snap = " [SNAP]" if snap_info else ""
    if typing_mode and typing:
        return f"Push/Pull | Type distance: {typing} (Enter to apply) | Esc: Cancel"
    elif typing:
        return f"Push/Pull | Axis: {axis}{snap} | Typed: {typing} (Enter to apply)"
    else:
        return f"Push/Pull | Axis: {axis}{snap} | LMB: Start/Finish | X/Y/Z: Lock | Type number | Esc/RMB: Cancel"


class SU_OT_pushpull_modal(bpy.types.Operator):
    """Extrude faces along their normal (Push/Pull) – SketchUp style"""
    bl_idname = "su.pushpull_modal"
    bl_label = "Push/Pull (SketchUp)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj and obj.type == 'MESH':
            return context.mode in {'EDIT_MESH', 'OBJECT'}
        return False

    def _raycast_face_enter_edit(self, context, event):
        """Object Mode: raycast under mouse, find face, switch to edit, select it."""
        region = context.region
        rv3d = context.region_data
        if not region or not rv3d:
            return False
        coord = (event.mouse_region_x, event.mouse_region_y)
        try:
            origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
            direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        except Exception:
            return False
        depsgraph = context.evaluated_depsgraph_get()
        try:
            result, loc, normal, face_idx, hit_obj, matrix = context.scene.ray_cast(
                depsgraph, origin, direction)
        except Exception:
            return False
        if not result or not hit_obj or hit_obj.type != 'MESH':
            self.report({'WARNING'}, "No mesh face under cursor")
            return False
        # Get original (non-evaluated) object
        orig_obj = hit_obj
        if hasattr(hit_obj, 'original') and hit_obj.original:
            orig_obj = hit_obj.original
        # Ensure it's in view_layer
        if orig_obj.name not in context.view_layer.objects:
            self.report({'WARNING'}, "Hit object not in view layer")
            return False
        # Switch active object and enter edit mode
        try:
            bpy.ops.object.select_all(action='DESELECT')
        except Exception:
            pass
        orig_obj.select_set(True)
        context.view_layer.objects.active = orig_obj
        bpy.ops.object.mode_set(mode='EDIT')
        # Select the face
        bpy.ops.mesh.select_all(action='DESELECT')
        context.tool_settings.mesh_select_mode = (False, False, True)
        bm = bmesh.from_edit_mesh(orig_obj.data)
        bm.faces.ensure_lookup_table()
        if face_idx < len(bm.faces):
            bm.faces[face_idx].select_set(True)
            bmesh.update_edit_mesh(orig_obj.data, loop_triangles=False, destructive=False)
            self._entered_edit_from_object = True
            return True
        self.report({'WARNING'}, "Face index out of range")
        return False

    def invoke(self, context, event):
        self._obj = None
        self._mesh = None
        self._bm = None
        self._backup_mesh = None
        self._state = "READY"
        self._typing = ""
        self._typing_mode = False
        self._axis_lock = "FREE"
        self._snap_info = None
        self._base_point_w = None
        self._dir_w = None
        self._dir_l = None
        self._mouse_start = None
        self._screen_base = None
        self._screen_vec = None
        self._px_per_unit = 1.0
        self._extruded_verts = []
        self._extruded_faces = []
        self._orig_co = {}
        self._kd = None
        self._meta = None
        self._depsgraph = None
        self._scalar = 0.0
        self._frame_count = 0
        self._entered_edit_from_object = False

        obj = context.active_object
        # --- OBJECT MODE: raycast → edit mode → select face ---
        if context.mode == 'OBJECT':
            if not obj or obj.type != 'MESH':
                self.report({'WARNING'}, "Select a mesh object first")
                return {'CANCELLED'}
            if not self._raycast_face_enter_edit(context, event):
                return {'CANCELLED'}
            obj = context.active_object

        if not obj or obj.type != 'MESH' or context.mode != 'EDIT_MESH':
            self.report({'WARNING'}, "Push/Pull requires Edit Mode on a mesh")
            return {'CANCELLED'}
        try:
            self._obj = obj
            self._mesh = obj.data
            self._bm = bmesh.from_edit_mesh(self._mesh)
            faces = [f for f in self._bm.faces if f.select]
            if not faces:
                self.report({'WARNING'}, "Select one or more faces")
                return {'CANCELLED'}
            self._backup_mesh = bpy.data.meshes.new("SU_PP_BACKUP")
            bm_copy = self._bm.copy()
            bm_copy.to_mesh(self._backup_mesh)
            bm_copy.free()
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)
            context.window_manager.modal_handler_add(self)
            _set_header(context, _pp_header_status(self._axis_lock, self._typing, None))
            return {'RUNNING_MODAL'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start Push/Pull: {e}")
            self._cleanup()
            return {'CANCELLED'}

    def _compute_base_and_direction(self, faces):
        mw = self._obj.matrix_world
        c = Vector((0, 0, 0))
        for f in faces:
            c += mw @ f.calc_center_median()
        c /= max(1, len(faces))
        self._base_point_w = c
        n = Vector((0, 0, 0))
        for f in faces:
            n += (mw.to_3x3() @ f.normal)
        if n.length < 1e-9:
            n = Vector((0, 0, 1))
        self._dir_w = n.normalized()

    def _apply_axis_lock_dir(self):
        if self._axis_lock == "X":
            self._dir_w = Vector((1, 0, 0))
        elif self._axis_lock == "Y":
            self._dir_w = Vector((0, 1, 0))
        elif self._axis_lock == "Z":
            self._dir_w = Vector((0, 0, 1))
        inv3 = self._obj.matrix_world.inverted().to_3x3()
        self._dir_l = (inv3 @ self._dir_w).normalized()

    def _start_extrude(self, context, event):
        try:
            faces = [f for f in self._bm.faces if f.select]
            self._compute_base_and_direction(faces)
            self._apply_axis_lock_dir()
            region = context.region
            rv3d = context.region_data
            base2d = view3d_utils.location_3d_to_region_2d(region, rv3d, self._base_point_w)
            dir2d = view3d_utils.location_3d_to_region_2d(
                region, rv3d, self._base_point_w + self._dir_w)
            if base2d and dir2d:
                self._screen_base = Vector((base2d.x, base2d.y))
                self._screen_vec = Vector((dir2d.x, dir2d.y)) - self._screen_base
                self._px_per_unit = max(1e-6, self._screen_vec.length)
            else:
                self._screen_base = Vector((event.mouse_region_x, event.mouse_region_y))
                self._screen_vec = Vector((1.0, 0.0))
                self._px_per_unit = 100.0
            self._mouse_start = Vector((event.mouse_region_x, event.mouse_region_y))
            individual = bool(context.scene.su_pushpull_individual)
            self._extruded_verts.clear()
            self._extruded_faces.clear()
            self._orig_co.clear()
            if individual:
                res = bmesh.ops.extrude_discrete_faces(self._bm, faces=faces)
                self._extruded_faces = list(res.get("faces", []))
                verts = set()
                for f in self._extruded_faces:
                    for v in f.verts:
                        verts.add(v)
                self._extruded_verts = list(verts)
            else:
                res = bmesh.ops.extrude_face_region(self._bm, geom=faces)
                geom = res.get("geom", [])
                self._extruded_verts = [e for e in geom if isinstance(e, bmesh.types.BMVert)]
                self._extruded_faces = [e for e in geom if isinstance(e, bmesh.types.BMFace)]
            for v in self._extruded_verts:
                self._orig_co[v] = v.co.copy()
            bmesh.update_edit_mesh(self._mesh, loop_triangles=False, destructive=True)
            self._state = "DRAGGING"
            self._scalar = 0.0
        except Exception as e:
            self.report({'ERROR'}, f"Extrusion failed: {e}")
            self._restore_backup()
            self._state = "READY"

    def _restore_backup(self):
        if not self._backup_mesh or not self._mesh:
            return
        try:
            if self._bm:
                self._bm.clear()
                self._bm.from_mesh(self._backup_mesh)
                bmesh.update_edit_mesh(self._mesh, loop_triangles=True, destructive=True)
            else:
                self._bm = bmesh.from_edit_mesh(self._mesh)
                self._bm.clear()
                self._bm.from_mesh(self._backup_mesh)
                bmesh.update_edit_mesh(self._mesh, loop_triangles=True, destructive=True)
        except Exception as e:
            print(f"[SU] Backup restore error: {e}")

    def _compute_scalar_from_mouse(self, event):
        cur = Vector((event.mouse_region_x, event.mouse_region_y))
        dp = cur - self._mouse_start
        if self._screen_vec.length < 1e-9:
            return 0.0
        sdir = self._screen_vec.normalized()
        return float(dp.dot(sdir) / self._px_per_unit)

    def _move_extruded(self, scalar, context):
        try:
            individual = bool(context.scene.su_pushpull_individual)
            if individual and self._extruded_faces:
                mw = self._obj.matrix_world
                inv3 = mw.inverted().to_3x3()
                for f in self._extruded_faces:
                    n_w = mw.to_3x3() @ f.normal
                    if n_w.length < 1e-9:
                        continue
                    d_l = inv3 @ n_w.normalized()
                    for v in f.verts:
                        v.co = self._orig_co.get(v, v.co) + d_l * scalar
            else:
                d_l = self._dir_l if self._dir_l else Vector((0, 0, 1))
                for v in self._extruded_verts:
                    v.co = self._orig_co.get(v, v.co) + d_l * scalar
            bmesh.update_edit_mesh(self._mesh, loop_triangles=False, destructive=True)
        except Exception as e:
            print(f"[SU] Move extruded error: {e}")

    def _cleanup(self):
        if self._backup_mesh and self._backup_mesh.name in bpy.data.meshes:
            try:
                bpy.data.meshes.remove(self._backup_mesh)
            except Exception:
                pass
            self._backup_mesh = None
        self._bm = None

    def _finish(self, context):
        _clear_header(context)
        self._cleanup()
        # Invalidate cache after push/pull modifies geometry
        _kdtree_cache.invalidate()

    def _cancel(self, context):
        self._restore_backup()
        try:
            if self._mesh:
                bmesh.update_edit_mesh(self._mesh, loop_triangles=True, destructive=True)
            self._bm = bmesh.from_edit_mesh(self._mesh) if self._mesh else None
        except Exception:
            pass
        self._finish(context)

    def _apply_typed_scalar(self, context):
        if not self._typing:
            return None
        v = _parse_length(context, self._typing)
        if v is None:
            self.report({'WARNING'}, f"Invalid input: {self._typing}")
            return None
        return float(v)

    def modal(self, context, event):
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)

        axis_key = _detect_axis_key(event)
        if axis_key:
            self._axis_lock = "FREE" if self._axis_lock == axis_key else axis_key
            if self._state == "DRAGGING":
                self._apply_axis_lock_dir()
            _set_header(context, _pp_header_status(
                self._axis_lock, self._typing, self._snap_info, self._typing_mode))
            return {'RUNNING_MODAL'}

        if event.type in _NAV_EVENTS and event.type not in {'X', 'Y', 'Z'}:
            return {'PASS_THROUGH'}
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._cancel(context)
            return {'CANCELLED'}
        if event.value == 'PRESS' and event.type in {'BACK_SPACE', 'DEL'}:
            self._typing = self._typing[:-1]
            _set_header(context, _pp_header_status(
                self._axis_lock, self._typing, self._snap_info, self._typing_mode))
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and len(event.unicode) == 1:
            ch = event.unicode
            if ch.isdigit() or ch in ".-+mMcCiInNfFtT\"\' ":
                self._typing += ch
                self._typing_mode = True
                _set_header(context, _pp_header_status(
                    self._axis_lock, self._typing, self._snap_info, self._typing_mode))
                return {'RUNNING_MODAL'}
        if event.type == 'RET' and event.value == 'PRESS':
            if self._state == "DRAGGING" and self._typing:
                typed = self._apply_typed_scalar(context)
                if typed is not None:
                    self._scalar = typed
                    self._move_extruded(self._scalar, context)
                    self._typing = ""
                    self._typing_mode = False
            _set_header(context, _pp_header_status(
                self._axis_lock, self._typing, self._snap_info, self._typing_mode))
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._state == "READY":
                self._start_extrude(context, event)
                _set_header(context, _pp_header_status(
                    self._axis_lock, self._typing, None, False))
                return {'RUNNING_MODAL'}
            else:
                self._finish(context)
                return {'FINISHED'}
        if self._state == "DRAGGING":
            if self._typing_mode:
                _set_header(context, _pp_header_status(
                    self._axis_lock, self._typing, self._snap_info, True))
                return {'RUNNING_MODAL'}
            self._scalar = self._compute_scalar_from_mouse(event)
            self._snap_info = None
            if context.scene.su_enable_vertex_snap and self._kd:
                if self._base_point_w and self._dir_w:
                    probe = self._base_point_w + self._dir_w * self._scalar
                    target_name = _get_target_name(context)
                    snap_co, snap_info = find_vertex_snap(
                        context, event, probe, self._kd, self._meta, self._depsgraph,
                        max_px=float(context.scene.su_snap_px), max_world=MAX_SNAP_DISTANCE_WORLD,
                        target_name=target_name)
                    if snap_co is not None:
                        self._scalar = float((snap_co - self._base_point_w).dot(self._dir_w))
                        self._snap_info = snap_info
            self._move_extruded(self._scalar, context)
            _set_header(context, _pp_header_status(
                self._axis_lock, self._typing, self._snap_info, False))
        return {'RUNNING_MODAL'}


# =============================================================================
# UI PANELS
# =============================================================================
class SU_PT_main(bpy.types.Panel):
    bl_label = "SketchUp Tools V5.0"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SketchUp"

    def draw(self, context):
        layout = self.layout
        scn = context.scene

        # SketchUp behaviour section
        box = layout.box()
        box.label(text="SketchUp Behaviour", icon='MESH_DATA')
        prefs = _get_prefs(context)
        if prefs:
            col = box.column(align=True)
            col.prop(prefs, "merge_distance", text="Weld Dist")
            col.prop(prefs, "auto_face", text="Auto Face")
            col.prop(prefs, "continuous_draw", text="Continuous Draw")
            col.prop(prefs, "keep_axis_lock_in_continuous", text="Keep Axis Lock")

        # Target mesh info
        target = None
        obj = context.active_object
        if obj and obj.type == 'MESH' and not obj.name.startswith("SU_TMP"):
            target = obj
        elif TARGET_MESH_NAME in bpy.data.objects:
            target = bpy.data.objects[TARGET_MESH_NAME]
        if target:
            box.label(text=f"Target: {target.name}", icon='OBJECT_DATA')
        else:
            box.label(text="Target: (will create SU_Model)", icon='ADD')

        layout.separator()

        box = layout.box()
        box.label(text="Drawing Settings", icon='SETTINGS')
        col = box.column(align=True)
        col.prop(scn, "su_draw_plane", text="Plane")
        col.prop(scn, "su_plane_offset", text="Offset")
        col.separator()
        col.prop(scn, "su_enable_vertex_snap", text="Vertex Snap")
        sub = col.row(align=True)
        sub.enabled = scn.su_enable_vertex_snap
        sub.prop(scn, "su_snap_px", text="Snap px")

        layout.separator()

        box = layout.box()
        box.label(text="Draw Tools", icon='GREASEPENCIL')
        col = box.column(align=True)
        col.operator("su.line_grid", text="Line Tool (L)", icon='IPO_LINEAR')
        col.operator("su.rect_grid", text="Rectangle Tool (R)", icon='MESH_PLANE')
        col.operator("su.arc_grid", text="Arc Tool (A)", icon='MESH_CIRCLE')

        layout.separator()

        box = layout.box()
        box.label(text="Advanced Arc Tools", icon='MESH_CIRCLE')
        col = box.column(align=True)
        col.prop(scn, "su_arc_mode", text="Mode")
        col.separator()
        col.operator("su.arc_three_point", text="Three-Point Arc", icon='SPHERECURVE')
        col.operator("su.arc_center_point", text="Center-Point Arc", icon='PIVOT_CURSOR')
        col.operator("su.arc_tangent", text="Tangent Arc", icon='MOD_CURVE')
        col.operator("su.arc_edge_connect", text="Edge-Connect Arc", icon='MOD_EDGESPLIT')
        col.operator("su.arc_fillet", text="Auto-Fillet Arc", icon='MOD_BEVEL')

        layout.separator()

        box = layout.box()
        box.label(text="Model Tools", icon='EDITMODE_HLT')
        col = box.column(align=True)
        obj = context.active_object
        if obj and obj.type == 'MESH':
            col.operator("su.pushpull_modal", text="Push/Pull (P)", icon='MOD_SOLIDIFY')
            row = col.row(align=True)
            row.prop(scn, "su_pushpull_individual", text="Individual Faces", toggle=True)
            if context.mode == 'OBJECT':
                col.label(text="Object Mode: click face to push/pull", icon='INFO')
        else:
            col.label(text="Select a mesh object", icon='INFO')

        layout.separator()

        box = layout.box()
        box.label(text="Shortcuts", icon='KEYINGSET')
        col = box.column(align=True)
        col.label(text="L: Line  |  R: Rectangle  |  A: Arc  |  P: Push/Pull")
        col.label(text="Arc modes: 3P, Center, Tangent, EC, Fillet")
        col.label(text="1/2/3: XY/XZ/YZ  |  X/Y/Z: Lock axis")
        col.label(text="Shift: Auto axis  |  Type+Enter: Precise")
        col.label(text="Backspace: Step back  |  Esc: Cancel")
        
        # Debug section (only in development)
        if context.preferences.view.show_developer_ui:
            box = layout.box()
            box.label(text="Debug Tools", icon='CONSOLE')
            col = box.column(align=True)
            col.operator("su.debug_face_split_tests", text="Run Face Split Tests")


class SU_MT_add_menu(bpy.types.Menu):
    bl_label = "SketchUp"
    bl_idname = "SU_MT_add_menu"

    def draw(self, context):
        layout = self.layout
        layout.operator_context = 'INVOKE_DEFAULT'
        layout.operator("su.line_grid", text="Line (SketchUp)", icon='IPO_LINEAR')
        layout.operator("su.line_engine", text="Line (Engine – Phase 1)", icon='IPO_LINEAR')
        layout.operator("su.rect_grid", text="Rectangle (SketchUp)", icon='MESH_PLANE')
        layout.operator("su.arc_grid", text="Arc (SketchUp)", icon='MESH_CIRCLE')
        layout.operator("su.arc_three_point", text="Three-Point Arc", icon='SPHERECURVE')
        layout.operator("su.arc_center_point", text="Center-Point Arc", icon='PIVOT_CURSOR')
        layout.operator("su.arc_tangent", text="Tangent Arc", icon='MOD_CURVE')
        layout.operator("su.arc_edge_connect", text="Edge-Connect Arc", icon='MOD_EDGESPLIT')
        layout.operator("su.arc_fillet", text="Auto-Fillet Arc", icon='MOD_BEVEL')
        layout.separator()
        layout.operator("su.pushpull_modal", text="Push/Pull (SketchUp)", icon='MOD_SOLIDIFY')


def _draw_add_menu(self, context):
    self.layout.separator()
    self.layout.menu("SU_MT_add_menu", icon='OUTLINER_OB_EMPTY')


# =============================================================================
# KEYMAPS
# =============================================================================
def _register_keymaps():
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon if wm else None
    if not kc:
        return
    try:
        addon = bpy.context.preferences.addons.get(ADDON_ID)
        prefs = addon.preferences if addon else None
    except Exception:
        prefs = None
    if prefs and not prefs.enable_hotkeys:
        return
    use_shift = bool(prefs.hotkeys_shift) if prefs else False
    km = kc.keymaps.new(name="3D View", space_type="VIEW_3D")
    kmi = km.keymap_items.new("su.line_grid", type="L", value="PRESS", shift=use_shift)
    _addon_keymaps.append((km, kmi))
    kmi = km.keymap_items.new("su.rect_grid", type="R", value="PRESS", shift=use_shift)
    _addon_keymaps.append((km, kmi))
    kmi = km.keymap_items.new("su.arc_grid", type="A", value="PRESS", shift=use_shift)
    _addon_keymaps.append((km, kmi))
    kmi = km.keymap_items.new("su.pushpull_modal", type="P", value="PRESS", shift=use_shift)
    _addon_keymaps.append((km, kmi))


def _unregister_keymaps():
    for km, kmi in _addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    _addon_keymaps.clear()


# =============================================================================
# DEBUG / SELF-TEST OPERATOR
# =============================================================================
class SU_OT_debug_face_split_tests(bpy.types.Operator):
    """Run internal face-split self-tests (SketchUp behavior verification)"""
    bl_idname = "su.debug_face_split_tests"
    bl_label = "Debug: Face Split Tests"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import sys
        import io
        
        # Redirect stdout to capture test output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            self._run_tests(context)
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        
        # Print to console
        print("=" * 60)
        print("SketchUp Face Split Self-Tests")
        print("=" * 60)
        print(output)
        print("=" * 60)
        
        # Show report
        if "FAIL" in output:
            self.report({'WARNING'}, "Some tests failed - check console")
        else:
            self.report({'INFO'}, "All tests passed")
        
        return {'FINISHED'}
    
    def _run_tests(self, context):
        print("Test 1: Create target mesh and large planar face...")
        
        # Ensure we have a clean target mesh
        target_name = "SU_Model"
        if target_name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[target_name], do_unlink=True)
        if target_name in bpy.data.meshes:
            bpy.data.meshes.remove(bpy.data.meshes[target_name])
        
        # Create fresh target
        mesh = bpy.data.meshes.new(target_name)
        obj = bpy.data.objects.new(target_name, mesh)
        safe_link_object(context, obj)
        context.view_layer.objects.active = obj
        
        # Create a large planar quad (10x10 on XY plane at Z=0)
        bm = bmesh.new()
        bm.from_mesh(mesh)
        
        verts = [
            bm.verts.new((-5, -5, 0)),
            bm.verts.new((5, -5, 0)),
            bm.verts.new((5, 5, 0)),
            bm.verts.new((-5, 5, 0))
        ]
        bm.verts.ensure_lookup_table()
        
        # Create face
        try:
            face = bm.faces.new(verts)
        except Exception as e:
            print(f"  FAIL: Could not create test face: {e}")
            bm.free()
            return
        
        bm.to_mesh(mesh)
        mesh.update()
        bm.free()
        
        initial_faces = len(mesh.polygons)
        print(f"  Created {initial_faces} face(s)")
        
        # Test 2: Draw a line across the face (diagonal)
        print("\nTest 2: Diagonal line across face...")
        verts_world = [Vector((-3, -3, 0)), Vector((3, 3, 0))]
        edges_idx = [(0, 1)]
        
        target = commit_geometry_to_target(context, verts_world, edges_idx, merge_dist=0.001)
        
        after_faces = len(target.data.polygons)
        after_edges = len(target.data.edges)
        
        print(f"  Faces before: {initial_faces}, after: {after_faces}")
        print(f"  Total edges: {after_edges}")
        
        if after_faces > initial_faces:
            print("  PASS: Face was split by diagonal line")
        else:
            print("  FAIL: Face was NOT split by diagonal line")
        
        # Test 3: Draw line from edge midpoint to opposite edge midpoint
        print("\nTest 3: Edge-midpoint to opposite edge-midpoint line...")
        # Get current mesh state
        bm = bmesh.new()
        bm.from_mesh(target.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        # Find a face to split (should be two faces now)
        if len(bm.faces) >= 1:
            # Pick first face
            face = bm.faces[0]
            face_verts = list(face.verts)
            if len(face_verts) >= 4:
                # Get midpoints of opposite edges
                v0 = face_verts[0].co
                v1 = face_verts[1].co
                v2 = face_verts[2].co
                v3 = face_verts[3].co
                
                mid1 = (v0 + v1) * 0.5
                mid2 = (v2 + v3) * 0.5
                
                bm.free()
                bm = None
                
                # Commit the crossing line
                verts_world = [mid1, mid2]
                edges_idx = [(0, 1)]
                
                target = commit_geometry_to_target(context, verts_world, edges_idx, merge_dist=0.001)
                
                faces_now = len(target.data.polygons)
                print(f"  Faces before: {after_faces}, after: {faces_now}")
                
                if faces_now > after_faces:
                    print("  PASS: Face split by edge-midpoint line")
                else:
                    print("  FAIL: Face NOT split by edge-midpoint line")
            else:
                bm.free()
                bm = None
                print("  SKIP: Face doesn't have 4 vertices")
        else:
            bm.free()
            bm = None
            print("  SKIP: No faces found")
        
        # Test 4: Polyline chain (arc-like) across face
        print("\nTest 4: Polyline chain (arc simulation) across face...")
        
        # Get current state
        faces_before = len(target.data.polygons)
        
        # Create a chain of 3 segments (arc-like)
        verts_world = [
            Vector((-4, 0, 0)),
            Vector((-2, 2, 0)),
            Vector((0, 2.5, 0)),
            Vector((2, 2, 0)),
            Vector((4, 0, 0))
        ]
        edges_idx = [(0, 1), (1, 2), (2, 3), (3, 4)]
        
        target = commit_geometry_to_target(context, verts_world, edges_idx, merge_dist=0.001)
        
        faces_after = len(target.data.polygons)
        print(f"  Faces before: {faces_before}, after: {faces_after}")
        
        if faces_after > faces_before:
            print("  PASS: Face(s) split by polyline chain")
        else:
            print("  FAIL: Face NOT split by polyline chain")
        
        # Test 5: Verify no duplicate edges
        print("\nTest 5: Checking for duplicate edges...")
        bm = bmesh.new()
        bm.from_mesh(target.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        
        edge_pairs = set()
        has_duplicates = False
        
        for edge in bm.edges:
            if not edge.is_valid:
                continue
            v1, v2 = edge.verts
            key = tuple(sorted((v1.index, v2.index)))
            if key in edge_pairs:
                has_duplicates = True
                print(f"  Found duplicate edge between verts {v1.index} and {v2.index}")
            edge_pairs.add(key)
        
        bm.free()
        
        if not has_duplicates:
            print("  PASS: No duplicate edges found")
        else:
            print("  FAIL: Duplicate edges found")
        
        # Test 6: Verify manifold geometry
        print("\nTest 6: Checking manifold condition...")
        bm = bmesh.new()
        bm.from_mesh(target.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        non_manifold_edges = 0
        for edge in bm.edges:
            if not edge.is_valid:
                continue
            if len(edge.link_faces) > 2:
                non_manifold_edges += 1
                print(f"  Non-manifold edge {edge.index} has {len(edge.link_faces)} faces")
        
        bm.free()
        
        if non_manifold_edges == 0:
            print("  PASS: All edges are manifold (0-2 faces)")
        else:
            print(f"  FAIL: Found {non_manifold_edges} non-manifold edges")
        
        print("\n" + "=" * 60)
        print("Face Split Self-Tests Complete")
        print("=" * 60)


# =============================================================================
# REGISTRATION
# =============================================================================
# =============================================================================
# PHASE 1/2 (FOUNDATION): Logic-only Tools + Single SUModalEngine-style Operator
# -----------------------------------------------------------------------------
# Goal: Introduce a single modal wrapper that centralizes:
#   - navigation pass-through
#   - typing (VCB)
#   - plane/axis
#   - snapping
# and delegates tool logic to lightweight "logic-only" tool classes.
#
# This is intentionally minimal and does NOT replace the existing geometry core.
# It simply funnels commits through a single kernel entrypoint (Phase 2).
# =============================================================================

class SUKernel:
    """Phase 2 entrypoint: single commit path over the existing geometry core."""
    def commit(self, context, verts_world, edges_idx):
        if not verts_world or not edges_idx:
            return
        prefs = _get_prefs(context)
        merge = prefs.merge_distance if prefs else MERGE_DISTANCE
        commit_geometry_to_target(
            context,
            verts_world=[v.copy() for v in verts_world],
            edges_idx=list(edges_idx),
            merge_dist=float(merge),
        )


class SUToolBase:
    """Logic-only tool protocol (Phase 1). No modal boilerplate here."""
    name = "Tool"

    def __init__(self):
        self.reset()

    def reset(self):
        self.drawing = False
        self.start = None
        self.end = None

    def on_click(self, point):
        """Return (did_commit, verts_world, edges_idx)."""
        raise NotImplementedError

    def on_move(self, point):
        raise NotImplementedError

    def on_enter(self, context, typed_text):
        """Optional: apply typed input then commit. Return (did_commit, verts, edges)."""
        return (False, None, None)

    def get_measurement(self, context, typed_text=""):
        return ""

    def get_vcb_label(self):
        return ""

    def get_vcb_value(self, context):
        return ""


class SULineTool(SUToolBase):
    name = "Line"

    def on_click(self, point):
        if not self.drawing:
            self.start = point.copy()
            self.end = point.copy()
            self.drawing = True
            return (False, None, None)
        self.end = point.copy()
        if (self.end - self.start).length < 1e-9:
            return (False, None, None)
        return (True, [self.start, self.end], [(0, 1)])

    def on_move(self, point):
        if not self.drawing or self.start is None:
            return
        self.end = point.copy()

    def on_enter(self, context, typed_text):
        if not self.drawing or self.start is None or self.end is None:
            return (False, None, None)
        if typed_text:
            length = _parse_length(context, typed_text)
            if length is not None:
                direction = (self.end - self.start)
                if direction.length > 1e-9:
                    direction.normalize()
                    self.end = self.start + direction * float(length)
        if (self.end - self.start).length < 1e-9:
            return (False, None, None)
        return (True, [self.start, self.end], [(0, 1)])

    def get_measurement(self, context, typed_text=""):
        if not self.drawing or self.start is None or self.end is None:
            return ""
        length = (self.end - self.start).length
        if length < 1e-9:
            return ""
        unit_system = context.scene.unit_settings.system
        return f"Length: {_format_measurement(length, unit_system)}"

    def get_vcb_label(self):
        return "Length"

    def get_vcb_value(self, context):
        if self.start is None or self.end is None:
            return ""
        return _format_length(context, (self.end - self.start).length)


class _EngineModalOperator(_BaseDrawOperator):
    """
    Single modal wrapper (Phase 1) that centralizes input handling and delegates
    geometry intent to a logic-only tool instance.

    Subclasses MUST set:
      - _tool_cls: a SUToolBase subclass
    """
    _tool_cls = None

    def _tool_name(self):
        return getattr(self._tool, "name", "Tool")

    def _get_measurement(self, context):
        try:
            return self._tool.get_measurement(context, self._typing)
        except Exception:
            return ""

    def _get_vcb_label(self):
        try:
            return self._tool.get_vcb_label()
        except Exception:
            return ""

    def _get_vcb_value(self, context):
        try:
            return self._tool.get_vcb_value(context)
        except Exception:
            return ""

    def invoke(self, context, event):
        self._kernel = SUKernel()
        self._tool = self._tool_cls() if self._tool_cls else SUToolBase()
        return self._invoke_common(context, event)

    def _update_preview(self, context):
        # Preview is handled by existing draw handlers (GPU).
        # Tools only maintain state (start/end/etc).
        pass

    def _commit_geometry(self, context, verts_world, edges_idx):
        self._kernel.commit(context, verts_world, edges_idx)

    def _commit(self, context):
        # Not used: commits are routed from modal via tool callbacks.
        return

    def modal(self, context, event):
        res = self._modal_common(context, event)
        if res is not None:
            return res

        # Enter -> tool may commit based on typed value
        if event.value == 'PRESS' and event.type == 'RET':
            did, verts, edges = self._tool.on_enter(context, self._typing)
            if did:
                self._typing = ""
                self._commit_geometry(context, verts, edges)
                prefs = _get_prefs(context)
                if prefs and prefs.continuous_draw:
                    # rebuild KDTree so new vert can be snapped to
                    self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(
                        context, force_rebuild=True)
                    # continue from last end
                    self._tool.start = self._tool.end.copy()
                    self._tool.end = self._tool.start.copy()
                    self._tool.drawing = True
                    if not (prefs.keep_axis_lock_in_continuous):
                        self._axis_lock = "FREE"
                    self._mouse_start = Vector((event.mouse_region_x, event.mouse_region_y))
                    self._update_header(context)
                    return {'RUNNING_MODAL'}
                self._finish(context)
                return {'FINISHED'}
            return {'RUNNING_MODAL'}

        plane = context.scene.su_draw_plane
        offset = context.scene.su_plane_offset
        hit = get_plane_hit(context, event, plane=plane, offset=offset)
        if hit is None:
            return {'RUNNING_MODAL'}

        hit, self._snap_info = self._snap_pipeline(context, event, hit)

        # Apply axis lock / shift inference if drawing
        if getattr(self._tool, "drawing", False) and getattr(self._tool, "start", None) is not None:
            if self._shift_down and self._mouse_start is not None:
                delta_px = Vector((event.mouse_region_x, event.mouse_region_y)) - self._mouse_start
                if delta_px.length > 6.0 and self._axis_lock == "FREE":
                    self._inferred_axis = _auto_axis_from_shift(self._tool.start, hit, plane)
            axis = self._axis_lock if self._axis_lock != "FREE" else getattr(self, "_inferred_axis", "FREE")
            if axis and axis != "FREE":
                hit = _apply_axis_lock(self._tool.start, hit, plane, axis)

        # Left click stages
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if not getattr(self._tool, "drawing", False):
                self._mouse_start = Vector((event.mouse_region_x, event.mouse_region_y))
            did, verts, edges = self._tool.on_click(hit)
            if did:
                self._commit_geometry(context, verts, edges)
                prefs = _get_prefs(context)
                if prefs and prefs.continuous_draw:
                    self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(
                        context, force_rebuild=True)
                    self._tool.start = self._tool.end.copy()
                    self._tool.end = self._tool.start.copy()
                    self._tool.drawing = True
                    self._typing = ""
                    if not (prefs.keep_axis_lock_in_continuous):
                        self._axis_lock = "FREE"
                    self._mouse_start = Vector((event.mouse_region_x, event.mouse_region_y))
                    self._update_header(context)
                    return {'RUNNING_MODAL'}
                self._finish(context)
                return {'FINISHED'}
            self._update_header(context)
            return {'RUNNING_MODAL'}

        # Mouse move updates
        if getattr(self._tool, "drawing", False):
            self._tool.on_move(hit)
            self._end = getattr(self._tool, "end", None)
            self._start = getattr(self._tool, "start", None)
            self._update_header(context)
        return {'RUNNING_MODAL'}


class SU_OT_line_engine(_EngineModalOperator, bpy.types.Operator):
    """Line tool via SUModalEngine wrapper (Phase 1)."""
    bl_idname = "su.line_engine"
    bl_label = "Line (Engine – Phase 1)"
    bl_options = {'REGISTER', 'UNDO'}
    _tool_cls = SULineTool



_CLASSES = [
    SU_Preferences,
    SU_OT_line_grid,
    SU_OT_line_engine,
    SU_OT_rectangle_grid,
    SU_OT_arc_grid,
    SU_OT_arc_three_point,
    SU_OT_arc_center_point,
    SU_OT_arc_tangent,
    SU_OT_arc_edge_connect,
    SU_OT_arc_fillet,
    SU_OT_pushpull_modal,
    SU_OT_debug_face_split_tests,
    SU_PT_main,
    SU_MT_add_menu,
    SU_PT_error_panel,
]


def register():
    global _registered, _IMPORT_ERROR_TEXT
    if _registered:
        return
    _IMPORT_ERROR_TEXT = ""
    try:
        _register_scene_props()
        for cls in _CLASSES:
            bpy.utils.register_class(cls)
        bpy.types.VIEW3D_MT_add.append(_draw_add_menu)
        _register_keymaps()
        _schedule_apply_defaults()
        _registered = True
        print("[SU] SketchUp Tools V5.0 registered successfully")
    except Exception as e:
        _IMPORT_ERROR_TEXT = traceback.format_exc()
        print(f"[SU] Registration failed:\n{_IMPORT_ERROR_TEXT}")
        try:
            bpy.utils.register_class(SU_PT_error_panel)
        except Exception:
            pass
        raise


def unregister():
    global _registered, _IMPORT_ERROR_TEXT, _timer_scheduled
    for key in list(_draw_handlers.keys()):
        unregister_draw_handler(key)
    _unregister_keymaps()
    try:
        bpy.types.VIEW3D_MT_add.remove(_draw_add_menu)
    except Exception:
        pass
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    _unregister_scene_props()
    _IMPORT_ERROR_TEXT = ""
    _registered = False
    _timer_scheduled = False
    print("[SU] SketchUp Tools V5.0 unregistered")


if __name__ == "__main__":
    register()
