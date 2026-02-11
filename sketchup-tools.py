# SPDX-License-Identifier: MIT
# SketchUp Tools (Grid Mode) - Enhanced & Fixed for Blender 4.5 LTS
bl_info = {
 "name":  "SketchUp Tools (Grid Mode) - Enhanced V2.5 ",
 "author":  "Sultan + OpenClaw + AI Enhancement ",
 "version": (2, 5, 0),
 "blender": (4, 5, 0),
 "location":  "View3D > Sidebar > SketchUp | Add > SketchUp ",
 "description":  "SketchUp-like tools with axis guides, snap markers, arc tool, and smart origins ",
 "category":  "3D View ",
 "doc_url":  " ",
 "tracker_url":  " ",
}
import bpy
import bmesh
import traceback
import time
import math
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree
from bpy_extras import view3d_utils
try:
    import gpu
    from gpu_extras.batch import batch_for_shader
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False
#=============================================================================
# GLOBALS & CONSTANTS
#=============================================================================
# FIX (BUG-5): Standard one-line ADDON_ID — works for single-file and packages.
ADDON_ID = __package__ or __name__
_IMPORT_ERROR_TEXT = ""
_registered = False
_addon_keymaps = []
_timer_scheduled = False
# Performance
KDTREE_CACHE_TIME = 0.5       # seconds
MAX_SNAP_DISTANCE_WORLD = 50.0 # Blender units
MAX_INPUT_LENGTH = 100         # characters
MAX_VALUE_MAGNITUDE = 1e6
# Visual feedback colors (RGBA)
SNAP_COLOR = (0.2, 1.0, 0.2, 1.0)
PREVIEW_COLOR = (1.0, 0.7, 0.0, 0.8)
AXIS_LOCK_COLOR = (0.0, 0.5, 1.0, 1.0)
# NEW: Axis colors (SketchUp style)
AXIS_X_COLOR = (1.0, 0.0, 0.0, 1.0)  # Red - full opacity
AXIS_Y_COLOR = (0.0, 1.0, 0.0, 1.0)  # Green - full opacity
AXIS_Z_COLOR = (0.0, 0.5, 1.0, 1.0)  # Blue - full opacity
AXIS_LOCKED_COLOR = (1.0, 1.0, 0.0, 1.0)  # Yellow
AXIS_DIM_ALPHA = 0.25  # Dimmed alpha for non-locked axes
# NEW: Arc tool defaults
DEFAULT_ARC_SEGMENTS = 24
MIN_ARC_SEGMENTS = 6
MAX_ARC_SEGMENTS = 256
# FIX (BUG-4): Navigation events that modals must pass through.
_NAV_EVENTS = {
    'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
    'WHEELINMOUSE', 'WHEELOUTMOUSE',
    'TRACKPADPAN', 'TRACKPADZOOM',
}
# NEW: GPU draw handler storage
_draw_handlers = {}
#=============================================================================
# SAFE OBJECT LINKING
#=============================================================================
def safe_link_object(context, obj):
    """Link object to scene with fallback for locked/overridden collections."""
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
#=============================================================================
# KDTREE CACHE
#=============================================================================
class VertexKDTreeCache:
    """KDTree caching system for vertex snapping performance."""
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
#=============================================================================
# PREFERENCES
#=============================================================================
class SU_Preferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID
    default_plane: bpy.props.EnumProperty(
        name= "Default Drawing Plane ",
        items=[
            ( "XY ",  "XY ",  "Draw on XY plane "),
            ( "XZ ",  "XZ ",  "Draw on XZ plane "),
            ( "YZ ",  "YZ ",  "Draw on YZ plane "),
        ],
        default= "XY ",
    )
    default_snap: bpy.props.BoolProperty(
        name= "Vertex Snapping Enabled ", default=True,
    )
    snap_pixel_radius: bpy.props.IntProperty(
        name= "Snap Pixel Radius ", default=12, min=2, max=64,
    )
    default_individual_faces: bpy.props.BoolProperty(
        name= "Default Individual Faces ", default=False,
    )
    enable_hotkeys: bpy.props.BoolProperty(
        name= "Enable Hotkeys (L/R/P/A) ", default=True,
    )
    hotkeys_shift: bpy.props.BoolProperty(
        name= "Use Shift+L/R/P/A ", default=False,
    )
    show_visual_feedback: bpy.props.BoolProperty(
        name= "Show Visual Feedback ", default=True,
    )
    grid_snap_enabled: bpy.props.BoolProperty(
        name= "Grid Snapping ", default=False,
    )
    grid_size: bpy.props.FloatProperty(
        name= "Grid Size ", default=0.1, min=0.001, max=10.0,
    )

    # NEW: A2 - Smart Origin Placement
    origin_mode: bpy.props.EnumProperty(
        name= "Origin Placement ",
        items=[
            ( "DEFAULT ",  "Default ",  "No origin adjustment "),
            ( "CENTER ",  "Center ",  "Place origin at geometry center "),
            ( "FIRST_CORNER ",  "First Corner ",  "Place origin at start point (SketchUp style) "),
        ],
        default= "DEFAULT ",
    )

    # NEW: A3 - Snap Visual Marker
    show_snap_marker: bpy.props.BoolProperty(
        name= "Show Snap Marker ", default=True,
    )
    snap_marker_size: bpy.props.IntProperty(
        name= "Snap Marker Size ", default=8, min=4, max=32,
    )

    # NEW: A4 - Axis Guides
    show_axis_guides: bpy.props.BoolProperty(
        name= "Show Axis Guides ", default=True,
    )
    axis_guide_length_mode: bpy.props.EnumProperty(
        name= "Axis Guide Length ",
        items=[
            ( "WORLD ",  "World Space ",  "Fixed world-space length "),
            ( "SCREEN ",  "Screen Space ",  "SketchUp-style screen-space length "),
        ],
        default= "SCREEN ",
    )
    axis_guide_world_length: bpy.props.FloatProperty(
        name= "World Length ", default=5.0, min=0.1, max=100.0,
    )
    axis_guide_screen_length: bpy.props.IntProperty(
        name= "Screen Length ", default=80, min=20, max=300,
    )

    # NEW: B - Arc Tool
    arc_segments: bpy.props.IntProperty(
        name= "Arc Segments ", default=DEFAULT_ARC_SEGMENTS,
        min=MIN_ARC_SEGMENTS, max=MAX_ARC_SEGMENTS,
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text= "SketchUp Tools V2.5 - Preferences ", icon='SETTINGS')

        box = layout.box()
        box.label(text= "Drawing Settings ", icon='GREASEPENCIL')
        col = box.column(align=True)
        col.prop(self,  "default_plane ")
        col.prop(self,  "default_snap ")
        col.prop(self,  "snap_pixel_radius ")
        col.separator()
        col.prop(self,  "origin_mode ")

        box = layout.box()
        box.label(text= "Grid Settings ", icon='GRID')
        col = box.column(align=True)
        col.prop(self,  "grid_snap_enabled ")
        sub = col.row()
        sub.enabled = self.grid_snap_enabled
        sub.prop(self,  "grid_size ")

        box = layout.box()
        box.label(text= "Push/Pull Settings ", icon='MOD_SOLIDIFY')
        box.prop(self,  "default_individual_faces ")

        box = layout.box()
        box.label(text= "Visual Feedback ", icon='HIDE_OFF')
        if not GPU_AVAILABLE:
            box.label(text= "GPU module unavailable ", icon='ERROR')
        col = box.column(align=True)
        col.prop(self,  "show_visual_feedback ")
        col.separator()
        col.prop(self,  "show_snap_marker ")
        sub = col.row()
        sub.enabled = self.show_snap_marker
        sub.prop(self,  "snap_marker_size ")
        col.separator()
        col.prop(self,  "show_axis_guides ")
        if self.show_axis_guides:
            sub = col.column(align=True)
            sub.prop(self,  "axis_guide_length_mode ", text= "Mode ")
            if self.axis_guide_length_mode ==  "WORLD ":
                sub.prop(self,  "axis_guide_world_length ")
            else:
                sub.prop(self,  "axis_guide_screen_length ")
        
        box = layout.box()
        box.label(text= "Arc Tool Settings ", icon='MESH_CIRCLE')
        box.prop(self,  "arc_segments ")

        box = layout.box()
        box.label(text= "Keyboard Shortcuts ", icon='KEYINGSET')
        col = box.column(align=True)
        col.prop(self,  "enable_hotkeys ")
        sub = col.row()
        sub.enabled = self.enable_hotkeys
        sub.prop(self,  "hotkeys_shift ")
class SU_PT_error_panel(bpy.types.Panel):
    bl_label = "SketchUp Tools - Error Report"
    bl_idname = "SU_PT_error_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SketchUp"
    @classmethod
    def poll(cls, context):
        # Only show when there is an error to report.
        return bool(_IMPORT_ERROR_TEXT)

    def draw(self, context):
        layout = self.layout
        layout.alert = True
        layout.label(text="Add-on Error Detected:", icon='ERROR')
        box = layout.box()
        for line in _IMPORT_ERROR_TEXT.splitlines()[:60]:
            box.label(text=line[:220])
#=============================================================================
# SCENE PROPERTIES
#=============================================================================
def _register_scene_props():
    bpy.types.Scene.su_draw_plane = bpy.props.EnumProperty(
        name= "Drawing Plane ",
        items=[
            ( "XY ",  "XY ",  "Draw on XY plane "),
            ( "XZ ",  "XZ ",  "Draw on XZ plane "),
            ( "YZ ",  "YZ ",  "Draw on YZ plane "),
        ],
        default= "XY ",
    )
    bpy.types.Scene.su_plane_offset = bpy.props.FloatProperty(
        name= "Plane Offset ", default=0.0, precision=4, step=10,
    )
    bpy.types.Scene.su_enable_vertex_snap = bpy.props.BoolProperty(
        name= "Vertex Snapping ", default=True,
    )
    bpy.types.Scene.su_snap_px = bpy.props.IntProperty(
        name= "Snap Radius (px) ", default=12, min=2, max=64,
    )
    bpy.types.Scene.su_pushpull_individual = bpy.props.BoolProperty(
        name= "Individual Faces ", default=False,
    )
    bpy.types.Scene.su_prefs_applied = bpy.props.BoolProperty(
        name= "Preferences Applied ", default=False, options={'HIDDEN'},
    )
    # NEW: Scene override for origin mode
    bpy.types.Scene.su_origin_mode_override = bpy.props.EnumProperty(
        name= "Origin Override ",
        items=[
            ( "USE_PREF ",  "Use Preference ",  "Use addon preference setting "),
            ( "DEFAULT ",  "Default ",  "No origin adjustment "),
            ( "CENTER ",  "Center ",  "Place origin at geometry center "),
            ( "FIRST_CORNER ",  "First Corner ",  "Place origin at start point "),
        ],
        default= "USE_PREF ",
    )
def _unregister_scene_props():
    props = [
        "su_draw_plane", "su_plane_offset", "su_enable_vertex_snap",
        "su_snap_px", "su_pushpull_individual", "su_prefs_applied",
        "su_origin_mode_override",
    ]
    for prop_name in props:
        try:
            delattr(bpy.types.Scene, prop_name)
        except Exception:
            pass
#=============================================================================
# PREFERENCES APPLICATION TIMER
#=============================================================================
def _apply_defaults_timer():
    try:
        ctx = bpy.context
        scn = getattr(ctx, "scene", None)
        if scn is None:
            return 0.25
        addon = ctx.preferences.addons.get(ADDON_ID)
        prefs = addon.preferences if addon else None
        if prefs is None:
            return 0.25
        if not getattr(scn, "su_prefs_applied", False):
            scn.su_draw_plane = prefs.default_plane
            scn.su_enable_vertex_snap = bool(prefs.default_snap)
            scn.su_snap_px = int(prefs.snap_pixel_radius)
            scn.su_pushpull_individual = bool(prefs.default_individual_faces)
            scn.su_prefs_applied = True
        else:
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
#=============================================================================
# UTILITY FUNCTIONS
#=============================================================================
def _set_header(context, text):
    if context.area:
        context.area.header_text_set(text)
def _clear_header(context):
    if context.area:
        context.area.header_text_set(None)
def _parse_length(context, s: str):
    s = (s or "").strip()
    if not s or len(s) > MAX_INPUT_LENGTH:
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
            if abs(val) > MAX_VALUE_MAGNITUDE or abs(val) < 1e-9:
                return None
            return val
        except Exception:
            return None
def _parse_rect_two(context, text: str):
    t = (text or "").strip().replace(" ", ",")
    if "," not in t:
        return None, None
    parts = t.split(",", 1)
    if len(parts) != 2:
        return None, None
    w = _parse_length(context, parts[0])
    h = _parse_length(context, parts[1])
    return w, h
def snap_to_grid(point, grid_size=0.1):
    if grid_size <= 0:
        return point
    return Vector((
        round(point.x / grid_size) * grid_size,
        round(point.y / grid_size) * grid_size,
        round(point.z / grid_size) * grid_size
    ))
# NEW: A1 - Reset object rotation to zero
def reset_object_rotation(obj):
    """Reset object world rotation to zero while preserving geometry position."""
    if not obj or not obj.data:
        return
    try:
        mw = obj.matrix_world.copy()
        loc, rot, scale = mw.decompose()
        # Apply inverse rotation to mesh data
        if hasattr(obj.data, 'transform'):
            rot_mat = rot.to_matrix().to_4x4()
            obj.data.transform(rot_mat)
            obj.data.update()
        
        # Set object matrix to location + scale only (no rotation)
        mat_loc = Matrix.Translation(loc)
        mat_scale = Matrix.Diagonal(scale.to_4d())
        obj.matrix_world = mat_loc @ mat_scale
    except Exception as e:
        print(f"[SU] Warning: Could not reset rotation: {e}")
# NEW: A2 - Apply smart origin placement
def apply_origin_placement(context, obj, mode, first_point=None):
    """Apply origin placement according to mode."""
    if not obj or not obj.data or mode ==  "DEFAULT ":
        return
    try:
        mesh = obj.data
        if mode ==  "CENTER ":
            if not mesh.vertices:
                return
            center = Vector((0, 0, 0))
            for v in mesh.vertices:
                center += v.co
            center /= len(mesh.vertices)
            for v in mesh.vertices:
                v.co -= center
            obj.location += obj.matrix_world.to_3x3() @ center
            mesh.update()
        elif mode ==  "FIRST_CORNER " and first_point is not None:
            offset = -first_point
            for v in mesh.vertices:
                v.co += offset
            obj.location += obj.matrix_world.to_3x3() @ first_point
            mesh.update()
    except Exception as e:
        print(f"[SU] Warning: Origin placement failed: {e}")
#=============================================================================
# PLANE & RAY CASTING
#=============================================================================
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
    if plane == 'XY':
        n = Vector((0.0, 0.0, 1.0))
        d = float(offset)
    elif plane == 'XZ':
        n = Vector((0.0, 1.0, 0.0))
        d = float(offset)
    elif plane == 'YZ':
        n = Vector((1.0, 0.0, 0.0))
        d = float(offset)
    else:
        n = Vector((0.0, 0.0, 1.0))
        d = float(offset)
    denom = n.dot(direction)
    if abs(denom) < 1e-9:
        return None
    t = (d - n.dot(origin)) / denom
    if t < 0.0:
        return None
    return origin + direction * t
#=============================================================================
# TEMP OBJECT MANAGEMENT
#=============================================================================
def create_temp_drawing_object(context, prefix="SU_TMP"):
    i = 1
    while True:
        name = f"{prefix}_{i:03d}"
        if name not in bpy.data.objects and name not in bpy.data.meshes:
            break
        i += 1
        if i > 999:
            name = f"{prefix}_TEMP"
            break
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    safe_link_object(context, obj)
    obj.hide_select = True
    obj.show_in_front = True
    return obj
def remove_object_safe(obj):
    try:
        if not obj:
            return
        mesh = obj.data if hasattr(obj, "data") else None
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        if mesh and mesh.name in bpy.data.meshes and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    except Exception as e:
        print(f"[SU] Warning: Failed to remove object: {e}")
#=============================================================================
# KDTREE & VERTEX SNAPPING
#=============================================================================
def build_vertex_kdtree(context):
    depsgraph = context.evaluated_depsgraph_get()
    verts = []
    for obj in context.view_layer.objects:
        if obj.type != 'MESH' or not obj.visible_get():
            continue
        if obj.name.startswith("SU_TMP"):
            continue
        try:
            obj_eval = obj.evaluated_get(depsgraph)
            try:
                me_eval = obj_eval.to_mesh(
                    preserve_all_data_layers=False, depsgraph=depsgraph
                )
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
def _is_vertex_visible(context, depsgraph, world_co, max_dist, event=None):
    region = context.region
    rv3d = context.region_data
    if not region or not rv3d:
        return True
    try:
        if event is not None:
            coord = (event.mouse_region_x, event.mouse_region_y)
            origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        else:
            cx = int(region.width * 0.5)
            cy = int(region.height * 0.5)
            origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, (cx, cy))
        vec = world_co - origin
        dist = vec.length
        if dist <= 1e-9:
            return True
        direction = vec / dist
        hit, loc, _norm, _face, _obj, _mat = depsgraph.scene_ray_cast(
            depsgraph, origin, direction, distance=min(dist, max_dist)
        )
        if not hit:
            return True
        return (loc - world_co).length < 1e-3
    except Exception:
        return True
def find_vertex_snap(context, event, hit_point, kd, meta, depsgraph, max_px=12.0, max_world=50.0):
    if kd is None or not meta or hit_point is None:
        return None, None
    region = context.region
    rv3d = context.region_data
    if not region or not rv3d:
        return None, None
    try:
        co, idx, dist = kd.find(hit_point)
    except Exception:
        return None, None
    if idx is None or dist > max_world:
        return None, None
    oname, vidx, wco = meta[idx]
    try:
        p2d = view3d_utils.location_3d_to_region_2d(region, rv3d, wco)
    except Exception:
        return None, None
    if p2d is None:
        return None, None
    dx = p2d.x - event.mouse_region_x
    dy = p2d.y - event.mouse_region_y
    px_dist = (dx * dx + dy * dy) ** 0.5
    if px_dist > max_px:
        return None, None
    if not _is_vertex_visible(context, depsgraph, wco, max_dist=max_world, event=event):
        return None, None
    snap_info = {
        "object": oname,
        "vert_index": int(vidx),
        "px_dist": float(px_dist),
        "world_dist": float(dist),
        "screen_pos": p2d,
    }
    return wco.copy(), snap_info
#=============================================================================
# PLANE & AXIS HELPERS
#=============================================================================
def _plane_axes(plane: str):
    if plane == "XY":
        return Vector((1, 0, 0)), Vector((0, 1, 0))
    elif plane == "XZ":
        return Vector((1, 0, 0)), Vector((0, 0, 1))
    elif plane == "YZ":
        return Vector((0, 1, 0)), Vector((0, 0, 1))
    return Vector((1, 0, 0)), Vector((0, 1, 0))
def _detect_axis_key(event):
    """Detect axis key press - works across keyboard layouts"""
    if event.value != 'PRESS':
        return None
    # Check physical key first
    if event.type in {'X', 'Y', 'Z'}:
        return event.type

    # Check unicode character (case-insensitive)
    if hasattr(event, 'unicode') and event.unicode:
        char = event.unicode.upper()
        if char in {'X', 'Y', 'Z'}:
            return char

    return None
def _infer_dominant_axis(start, end, plane, axis_lock):
    """Infer dominant axis from movement direction - SketchUp style"""
    # Explicit lock overrides everything
    if axis_lock and axis_lock != "FREE":
        return axis_lock
    # Calculate direction
    delta = end - start
    if delta.length < 1e-6:
        return "FREE"

    # Get plane axes
    plane_axes = {
        "XY": [("X", abs(delta.x)), ("Y", abs(delta.y))],
        "XZ": [("X", abs(delta.x)), ("Z", abs(delta.z))],
        "YZ": [("Y", abs(delta.y)), ("Z", abs(delta.z))]
    }

    axes = plane_axes.get(plane, [("X", abs(delta.x)), ("Y", abs(delta.y))])

    # Return dominant axis
    if not axes:
        return "FREE"

    dominant = max(axes, key=lambda x: x[1])
    if dominant[1] < 1e-6:
        return "FREE"

    return dominant[0]
def _get_axis_color(axis, locked=False):
    """Get color for axis - SketchUp style"""
    base_colors = {
        "X": (1.0, 0.0, 0.0),  # Red
        "Y": (0.0, 1.0, 0.0),  # Green
        "Z": (0.0, 0.5, 1.0),  # Blue
    }
    color = base_colors.get(axis, (0.7, 0.7, 0.7))

    if locked:
        # Brighter when locked
        alpha = 1.0
    else:
        # Softer when inferred
        alpha = 0.8

    return (*color, alpha)
def draw_preview_line_colored(start, end, axis, locked):
    """Draw colored preview line based on axis"""
    if not GPU_AVAILABLE:
        return
    try:
        color = _get_axis_color(axis, locked)
        vertices = [start, end]
        shader = gpu.shader.from_builtin('FLAT_COLOR')
        colors = [color, color]
        batch = batch_for_shader(
            shader, 'LINES',
            {"pos": vertices, "color": colors}
        )
        
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
    """Draw colored preview rectangle based on axis"""
    if not GPU_AVAILABLE:
        return
    try:
        ax_a, ax_b = _plane_axes(plane)
        d = end - start
        w = d.dot(ax_a)
        h = d.dot(ax_b)
        # Four corners
        p1 = start
        p2 = start + ax_a * w
        p3 = start + ax_a * w + ax_b * h
        p4 = start + ax_b * h
        
        # Draw 4 edges
        edges = [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]
        color = _get_axis_color(axis, locked)
        
        vertices = []
        colors = []
        for edge_start, edge_end in edges:
            vertices.extend([edge_start, edge_end])
            colors.extend([color, color])
        
        shader = gpu.shader.from_builtin('FLAT_COLOR')
        batch = batch_for_shader(
            shader, 'LINES',
            {"pos": vertices, "color": colors}
        )
        
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
def constrain_to_plane(point, start, plane):
    """Hard geometric constraint to plane"""
    result = point.copy()
    if plane == "XY":
        result.z = start.z
    elif plane == "XZ":
        result.y = start.y
    elif plane == "YZ":
        result.x = start.x
    return result
def constrain_to_axis(point, start, axis_name, plane):
    """Hard geometric constraint to axis with plane enforcement"""
    if not axis_name or axis_name == "FREE":
        return constrain_to_plane(point, start, plane)
    axis_vectors = {
        "X": Vector((1, 0, 0)),
        "Y": Vector((0, 1, 0)),
        "Z": Vector((0, 0, 1))
    }

    axis = axis_vectors.get(axis_name)
    if not axis:
        return constrain_to_plane(point, start, plane)

    delta = point - start
    projection_length = delta.dot(axis)
    projected_point = start + axis * projection_length

    return constrain_to_plane(projected_point, start, plane)
def _compute_perpendicular_axis_movement(context, event, start, axis_name, last_mouse_pos=None):
    """
    Compute movement along an axis perpendicular to the drawing plane.
    Uses screen-space mouse delta to determine the amount of movement.
    Returns the new endpoint.
    """
    region = context.region
    rv3d = context.region_data
    if not region or not rv3d:
        return start.copy()
    # Get axis vector
    axis_vectors = {
        "X": Vector((1, 0, 0)),
        "Y": Vector((0, 1, 0)),
        "Z": Vector((0, 0, 1))
    }
    axis = axis_vectors.get(axis_name)
    if not axis:
        return start.copy()

    # Project start point to screen
    try:
        start_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, start)
        if not start_2d:
            # Fallback: project mouse to 3D at start depth
            mx, my = event.mouse_region_x, event.mouse_region_y
            projected = view3d_utils.region_2d_to_location_3d(region, rv3d, (mx, my), start)
            delta = projected - start
            world_distance = delta.dot(axis)
            return start + axis * world_distance
        
        # Project a point along the axis to screen
        test_point = start + axis * 1.0
        test_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, test_point)
        if not test_2d:
            # Fallback
            mx, my = event.mouse_region_x, event.mouse_region_y
            projected = view3d_utils.region_2d_to_location_3d(region, rv3d, (mx, my), start)
            delta = projected - start
            world_distance = delta.dot(axis)
            return start + axis * world_distance
        
        # Calculate screen direction of the axis
        screen_axis = test_2d - start_2d
        if screen_axis.length < 1e-6:
            # Fallback when axis is nearly parallel to view direction
            mx, my = event.mouse_region_x, event.mouse_region_y
            projected = view3d_utils.region_2d_to_location_3d(region, rv3d, (mx, my), start)
            delta = projected - start
            world_distance = delta.dot(axis)
            return start + axis * world_distance
        
        screen_axis.normalize()
        
        # Calculate mouse delta from start
        mouse_pos = Vector((event.mouse_region_x, event.mouse_region_y))
        if last_mouse_pos:
            mouse_delta = mouse_pos - last_mouse_pos
        else:
            mouse_delta = mouse_pos - start_2d
        
        # Project mouse delta onto screen axis direction
        screen_distance = mouse_delta.dot(screen_axis)
        
        # Convert screen distance to world distance
        # Use the ratio of screen distance to world distance
        screen_length_per_unit = (test_2d - start_2d).length
        if screen_length_per_unit > 1e-6:
            world_distance = screen_distance / screen_length_per_unit
        else:
            world_distance = 0.0
        
        # Return point along axis
        return start + axis * world_distance
        
    except Exception:
        # Ultimate fallback: project mouse to 3D at start depth
        mx, my = event.mouse_region_x, event.mouse_region_y
        try:
            projected = view3d_utils.region_2d_to_location_3d(region, rv3d, (mx, my), start)
            delta = projected - start
            world_distance = delta.dot(axis)
            return start + axis * world_distance
        except Exception:
            return start.copy()
def _apply_axis_lock(start: Vector, cur: Vector, plane: str, axis_lock: str):
    """Apply axis lock constraint - SketchUp behavior"""
    return constrain_to_axis(cur, start, axis_lock, plane)
def _auto_axis_from_shift(start: Vector, cur: Vector, plane: str):
    delta = cur - start
    ax_a, ax_b = _plane_axes(plane)
    da = abs(delta.dot(ax_a))
    db = abs(delta.dot(ax_b))
    if da >= db:
        if plane in ("XY", "XZ"):
            return "X"
        elif plane == "YZ":
            return "Y"
    else:
        if plane == "XY":
            return "Y"
        elif plane in ("XZ", "YZ"):
            return "Z"
    return "FREE"
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
def _header_status(tool_name, plane, axis_lock, typing, snap_info):
    axis = axis_lock if axis_lock and axis_lock != "FREE" else "FREE"
    snap = " [SNAP]" if snap_info else ""
    t = f" | Type: {typing}" if typing else ""
    return f"{tool_name} | Plane: {plane} | Axis: {axis}{snap}{t} | Esc/RMB: Cancel"
#=============================================================================
# NEW: GPU DRAW HANDLERS (A3 & A4)
#=============================================================================
def register_draw_handler(key, handler, args, region_type='WINDOW', space='POST_PIXEL'):
    global _draw_handlers
    if not GPU_AVAILABLE:
        return
    try:
        if key in _draw_handlers:
            unregister_draw_handler(key)
        handle = bpy.types.SpaceView3D.draw_handler_add(
            handler, args, region_type, space
        )
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
            # Handle tuple format
            if isinstance(data, tuple):
                if len(data) == 2:
                    # Format: (handle_pixel, handle_view)
                    handle_pixel, handle_view = data
                    bpy.types.SpaceView3D.draw_handler_remove(handle_pixel, 'WINDOW')
                    bpy.types.SpaceView3D.draw_handler_remove(handle_view, 'WINDOW')
                elif len(data) == 3:
                    # Check if it's (handle, region, space) or (handle_pixel, handle_view, handle_preview)
                    if isinstance(data[1], str):
                        # Format: (handle, region_type, space)
                        handle, region_type, space = data
                        bpy.types.SpaceView3D.draw_handler_remove(handle, region_type)
                    else:
                        # Format: (handle_pixel, handle_view, handle_preview)
                        handle_pixel, handle_view, handle_preview = data
                        bpy.types.SpaceView3D.draw_handler_remove(handle_pixel, 'WINDOW')
                        bpy.types.SpaceView3D.draw_handler_remove(handle_view, 'WINDOW')
                        bpy.types.SpaceView3D.draw_handler_remove(handle_preview, 'WINDOW')
                else:
                    # Single handle
                    bpy.types.SpaceView3D.draw_handler_remove(data, 'WINDOW')
            else:
                # Single handle
                bpy.types.SpaceView3D.draw_handler_remove(data, 'WINDOW')
            del _draw_handlers[key]
    except Exception as e:
        print(f"[SU] Warning: Failed to unregister draw handler: {e}")
# NEW: A3 - Snap marker drawing
def draw_snap_marker_simple(snap_info, prefs):
    """Draw snap marker in 2D screen space"""
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
            vx = x + size * math.cos(angle)
            vy = y + size * math.sin(angle)
            vertices.append((vx, vy))
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
# NEW: A4 - Axis guides drawing
def draw_axis_guides_simple(context, anchor, plane, axis_lock, prefs):
    """Draw axis guides in 3D world space"""
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
                # SCREEN MODE - SketchUp style
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
                p_end_world = view3d_utils.region_2d_to_location_3d(
                    region, rv3d, p_end_2d, anchor
                )
                if not p_end_world:
                    continue
                vertices.extend([anchor, p_end_world])
                colors.extend([color, color])
            else:
                # WORLD MODE
                length = prefs.axis_guide_world_length
                p_end = anchor + axis_vec * length
                vertices.extend([anchor, p_end])
                colors.extend([color, color])
        if not vertices:
            return
        shader = gpu.shader.from_builtin('FLAT_COLOR')
        batch = batch_for_shader(
            shader, 'LINES',
            {"pos": vertices, "color": colors}
        )
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(3.0)  # Thicker for visibility
        gpu.state.depth_test_set('LESS_EQUAL')
        batch.draw(shader)
        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)
        gpu.state.depth_test_set('NONE')
    except Exception:
        pass
#=============================================================================
# BASE DRAW OPERATOR (Enhanced with GPU draw handlers)
#=============================================================================
class _BaseDrawOperator:
    """
    Shared modal infrastructure for plane-based drawing tools.
    Subclasses MUST implement:
        _tool_name()        -> str
        _update_preview(context)
        _commit(context)
    And MUST set:
        _tmp_prefix         -> str  (e.g. "SU_TMP_LINE")
    """

    _tmp_prefix = "SU_TMP"

    # -- Subclass interface (override these) ----------------------------------

    def _tool_name(self):
        return "Tool"

    def _update_preview(self, context):
        raise NotImplementedError

    def _commit(self, context):
        raise NotImplementedError

    # -- Shared invoke --------------------------------------------------------

    def _invoke_common(self, context, event):
        """Shared initialization. Returns operator result set."""
        self._tmp_obj = None
        self._bm = None
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
        self._draw_handler_key = None  # NEW
        self._inferred_axis = "FREE"  # NEW: for live preview coloring
        self._mouse_start = None  # NEW: for perpendicular axis movement

        if not context.area or context.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "Must be in 3D View area")
            return {'CANCELLED'}

        try:
            self._tmp_obj = create_temp_drawing_object(context, prefix=self._tmp_prefix)
            self._bm = bmesh.new()
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)
            
            # NEW: Register draw handlers
            self._register_draw_handlers(context)
            
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

    # NEW: Register GPU draw handlers
    def _register_draw_handlers(self, context):
        if not GPU_AVAILABLE:
            return
        addon = context.preferences.addons.get(ADDON_ID)
        prefs = addon.preferences if addon else None
        if not prefs:
            return
        
        # Store reference to self for callback access
        op_ref = self
        self._draw_handler_key = f"{self._tool_name().lower()}_{id(self)}"
        
        # POST_PIXEL callback for snap marker
        def draw_snap_marker():
            snap_info = getattr(op_ref, '_snap_info', None)
            if snap_info and prefs.show_snap_marker:
                draw_snap_marker_simple(snap_info, prefs)
        
        # POST_VIEW callback for axis guides
        def draw_axis_guides():
            start = getattr(op_ref, '_start', None)
            axis_lock = getattr(op_ref, '_axis_lock', 'FREE')
            if start and prefs.show_axis_guides:
                draw_axis_guides_simple(
                    context, start,
                    context.scene.su_draw_plane,
                    axis_lock, prefs
                )
        
        # POST_VIEW callback for colored preview (SketchUp-style inference)
        def draw_preview_colored():
            start = getattr(op_ref, '_start', None)
            end = getattr(op_ref, '_end', None)
            drawing = getattr(op_ref, '_drawing', False)
            axis_lock = getattr(op_ref, '_axis_lock', 'FREE')
            inferred_axis = getattr(op_ref, '_inferred_axis', 'FREE')
            tool_name = getattr(op_ref, '_tool_name', lambda: " ")()
            
            if start and end and drawing:
                # Prioritize axis_lock when locked for coloring (FIX for Z axis blue preview)
                axis = axis_lock if axis_lock != "FREE" else inferred_axis
                locked = (axis_lock != "FREE")
                if axis != "FREE":
                    if tool_name == "Rectangle":
                        plane = context.scene.su_draw_plane
                        draw_preview_rectangle_colored(start, end, plane, axis, locked)
                    else:
                        # Line and other tools
                        draw_preview_line_colored(start, end, axis, locked)
        
        # Register all handlers
        try:
            handle_pixel = bpy.types.SpaceView3D.draw_handler_add(
                draw_snap_marker, (), 'WINDOW', 'POST_PIXEL'
            )
            handle_view = bpy.types.SpaceView3D.draw_handler_add(
                draw_axis_guides, (), 'WINDOW', 'POST_VIEW'
            )
            handle_preview = bpy.types.SpaceView3D.draw_handler_add(
                draw_preview_colored, (), 'WINDOW', 'POST_VIEW'
            )
            _draw_handlers[self._draw_handler_key] = (handle_pixel, handle_view, handle_preview)
        except Exception as e:
            print(f"[SU] Failed to register draw handlers: {e}")

    # -- Shared cleanup -------------------------------------------------------

    def _cleanup(self, context=None):
        if self._bm:
            try:
                self._bm.free()
            except Exception:
                pass
        self._bm = None
        if self._tmp_obj:
            remove_object_safe(self._tmp_obj)
        self._tmp_obj = None
        # NEW: Unregister draw handlers 
        if self._draw_handler_key:
            unregister_draw_handler(self._draw_handler_key)
            self._draw_handler_key = None

    def _finish(self, context):
        _clear_header(context)
        self._cleanup(context)

    # -- Shared snap + grid pipeline ------------------------------------------

    def _snap_pipeline(self, context, event, hit):
        """Apply vertex snap, then grid snap. Returns (snapped_hit, snap_info)."""
        snap_info = None
        if context.scene.su_enable_vertex_snap:
            snap_co, sinfo = find_vertex_snap(
                context, event, hit, self._kd, self._meta, self._depsgraph,
                max_px=float(context.scene.su_snap_px),
                max_world=MAX_SNAP_DISTANCE_WORLD,
            )
            if snap_co is not None:
                return snap_co, sinfo

        addon = context.preferences.addons.get(ADDON_ID)
        prefs = addon.preferences if addon else None
        if prefs and prefs.grid_snap_enabled:
            hit = snap_to_grid(hit, prefs.grid_size)
        return hit, snap_info

    # -- Shared modal dispatch ------------------------------------------------

    def _modal_common(self, context, event):
        """
        Handle events shared between drawing tools.

        Returns:
            A set like {'RUNNING_MODAL'} if the event was consumed,
            or None if the subclass should handle it.
        """
        # NEW: Force redraw for visual feedback
        if context.area:
            context.area.tag_redraw()
        
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)

        # Axis lock - MUST be before navigation pass-through (supports all keyboard layouts)
        axis_key = _detect_axis_key(event)
        if axis_key:
            self._axis_lock = "FREE" if self._axis_lock == axis_key else axis_key
            self._update_header(context)
            return {'RUNNING_MODAL'}

        # FIX (BUG-4): Pass through viewport navigation events (exclude axis keys)
        if event.type in _NAV_EVENTS and event.type not in {'X', 'Y', 'Z'}:
            return {'PASS_THROUGH'}

        # Plane switching (1/2/3)
        if _handle_plane_keys(context, event):
            self._update_header(context)
            return {'RUNNING_MODAL'}

        # Cancel
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._finish(context)
            return {'CANCELLED'}

        # Shift tracking
        if event.type in {'LEFT_SHIFT', 'RIGHT_SHIFT'}:
            self._shift_down = event.value in {'PRESS', 'CLICK_DRAG'}
            return {'RUNNING_MODAL'}

        # Backspace (typing)
        if event.value == 'PRESS' and event.type in {'BACK_SPACE', 'DEL'}:
            self._typing = self._typing[:-1]
            self._update_header(context)
            return {'RUNNING_MODAL'}

        # Character input (typing)
        if event.value == 'PRESS' and len(event.unicode) == 1:
            ch = event.unicode
            if ch.isdigit() or ch in ".,-+mMcCiInNfFtT ":
                self._typing += ch
                self._update_header(context)
                return {'RUNNING_MODAL'}

        return None  # Not consumed — let subclass handle

    def _update_header(self, context):
        _set_header(context, _header_status(
            self._tool_name(), context.scene.su_draw_plane,
            self._axis_lock, self._typing, self._snap_info,
        ))
#=============================================================================
# LINE TOOL (Enhanced with A1, A2)
#=============================================================================
class SU_OT_line_grid(_BaseDrawOperator, bpy.types.Operator):
    bl_idname = "su.line_grid"
    bl_label = "Line (Grid)"
    bl_options = {'REGISTER', 'UNDO'}
    _tmp_prefix = "SU_TMP_LINE"

    def _tool_name(self):
        return "Line"

    def invoke(self, context, event):
        return self._invoke_common(context, event)

    def _update_preview(self, context):
        if not self._tmp_obj or not self._bm:
            return
        try:
            self._bm.clear()
            if self._start is None or self._end is None:
                self._tmp_obj.data.clear_geometry()
                return
            v1 = self._bm.verts.new(self._start)
            v2 = self._bm.verts.new(self._end)
            self._bm.edges.new((v1, v2))
            me = self._tmp_obj.data
            self._bm.to_mesh(me)
            me.update()
        except Exception:
            pass

    def _commit(self, context):
        try:
            mesh = bpy.data.meshes.new("SU_Line")
            obj = bpy.data.objects.new("SU_Line", mesh)
            safe_link_object(context, obj)
            bm = bmesh.new()
            v1 = bm.verts.new(self._start)
            v2 = bm.verts.new(self._end)
            bm.edges.new((v1, v2))
            bm.to_mesh(mesh)
            mesh.update()
            bm.free()
            
            # NEW: A1 - Reset rotation
            reset_object_rotation(obj)
            
            # NEW: A2 - Apply origin placement
            addon = context.preferences.addons.get(ADDON_ID)
            prefs = addon.preferences if addon else None
            origin_mode = context.scene.su_origin_mode_override
            if origin_mode == "USE_PREF" and prefs:
                origin_mode = prefs.origin_mode
            apply_origin_placement(context, obj, origin_mode, self._start)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create line: {e}")

    def modal(self, context, event):
        res = self._modal_common(context, event)
        if res is not None:
            return res

        # Enter - apply typed length
        if event.value == 'PRESS' and event.type == 'RET':
            if self._drawing and self._start is not None and self._typing:
                length = _parse_length(context, self._typing)
                if length is not None:
                    direction = (self._end - self._start)
                    if direction.length > 1e-9:
                        direction.normalize()
                        self._end = self._start + direction * float(length)
                        self._update_preview(context)
                    self._typing = ""
                self._update_header(context)
            return {'RUNNING_MODAL'}

        plane = context.scene.su_draw_plane
        offset = context.scene.su_plane_offset
        hit = get_plane_hit(context, event, plane=plane, offset=offset)
        if hit is None:
            return {'RUNNING_MODAL'}

        hit, self._snap_info = self._snap_pipeline(context, event, hit)

        # Left mouse - start/finish
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

        # Update during drawing
        if self._drawing:
            # Check if axis lock is perpendicular to plane
            perpendicular = False
            if self._axis_lock == "Z" and plane == "XY":
                perpendicular = True
            elif self._axis_lock == "Y" and plane == "XZ":
                perpendicular = True
            elif self._axis_lock == "X" and plane == "YZ":
                perpendicular = True
            
            if perpendicular:
                # Use perpendicular axis movement calculation (now robust for Z axis)
                cur = _compute_perpendicular_axis_movement(
                    context, event, self._start, self._axis_lock, self._mouse_start
                )
            else:
                # Use standard constraint
                cur = hit.copy()
                if self._shift_down and self._axis_lock == "FREE":
                    lock = _auto_axis_from_shift(self._start, cur, plane)
                    cur = _apply_axis_lock(self._start, cur, plane, lock)
                else:
                    cur = _apply_axis_lock(self._start, cur, plane, self._axis_lock)
            
            self._end = cur
            
            # Calculate inferred axis for preview coloring
            self._inferred_axis = _infer_dominant_axis(self._start, self._end, plane, self._axis_lock)
            
            self._update_preview(context)
            self._update_header(context)

        return {'RUNNING_MODAL'}
#=============================================================================
# RECTANGLE TOOL (Enhanced with A1, A2)
#=============================================================================
class SU_OT_rectangle_grid(_BaseDrawOperator, bpy.types.Operator):
    bl_idname = "su.rect_grid"
    bl_label = "Rectangle (Grid)"
    bl_options = {'REGISTER', 'UNDO'}
    _tmp_prefix = "SU_TMP_RECT"

    def _tool_name(self):
        return "Rectangle"

    def invoke(self, context, event):
        return self._invoke_common(context, event)

    def _update_preview(self, context):
        if not self._tmp_obj or not self._bm:
            return
        try:
            self._bm.clear()
            if self._start is None or self._end is None:
                self._tmp_obj.data.clear_geometry()
                return
            plane = context.scene.su_draw_plane
            ax_a, ax_b = _plane_axes(plane)
            delta = self._end - self._start
            w = float(delta.dot(ax_a))
            h = float(delta.dot(ax_b))
            if abs(w) < 1e-9 or abs(h) < 1e-9:
                self._tmp_obj.data.clear_geometry()
                return
            p0 = self._start
            p1 = p0 + ax_a * w
            p2 = p1 + ax_b * h
            p3 = p0 + ax_b * h
            v0 = self._bm.verts.new(p0)
            v1 = self._bm.verts.new(p1)
            v2 = self._bm.verts.new(p2)
            v3 = self._bm.verts.new(p3)
            try:
                self._bm.faces.new((v0, v1, v2, v3))
            except ValueError:
                pass
            me = self._tmp_obj.data
            self._bm.to_mesh(me)
            me.update()
        except Exception:
            pass

    def _commit(self, context):
        try:
            plane = context.scene.su_draw_plane
            ax_a, ax_b = _plane_axes(plane)
            delta = self._end - self._start
            w = float(delta.dot(ax_a))
            h = float(delta.dot(ax_b))
            if abs(w) < 1e-9 or abs(h) < 1e-9:
                self.report({'WARNING'}, "Rectangle too small")
                return
            mesh = bpy.data.meshes.new("SU_Rectangle")
            obj = bpy.data.objects.new("SU_Rectangle", mesh)
            safe_link_object(context, obj)
            p0 = self._start
            p1 = p0 + ax_a * w
            p2 = p1 + ax_b * h
            p3 = p0 + ax_b * h
            bm = bmesh.new()
            v0 = bm.verts.new(p0)
            v1 = bm.verts.new(p1)
            v2 = bm.verts.new(p2)
            v3 = bm.verts.new(p3)
            try:
                bm.faces.new((v0, v1, v2, v3))
            except ValueError:
                pass
            bm.to_mesh(mesh)
            mesh.update()
            bm.free()
            
            # NEW: A1 - Reset rotation
            reset_object_rotation(obj)
            
            # NEW: A2 - Apply origin placement
            addon = context.preferences.addons.get(ADDON_ID)
            prefs = addon.preferences if addon else None
            origin_mode = context.scene.su_origin_mode_override
            if origin_mode == "USE_PREF" and prefs:
                origin_mode = prefs.origin_mode
            apply_origin_placement(context, obj, origin_mode, self._start)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create rectangle: {e}")

    def modal(self, context, event):
        res = self._modal_common(context, event)
        if res is not None:
            return res

        # Enter - apply typed dimensions
        if event.value == 'PRESS' and event.type == 'RET':
            if self._drawing and self._start is not None and self._typing:
                w, h = _parse_rect_two(context, self._typing)
                if w is not None and h is not None:
                    plane = context.scene.su_draw_plane
                    ax_a, ax_b = _plane_axes(plane)
                    self._end = self._start + ax_a * float(w) + ax_b * float(h)
                    self._update_preview(context)
                self._typing = ""
                self._update_header(context)
            return {'RUNNING_MODAL'}

        plane = context.scene.su_draw_plane
        offset = context.scene.su_plane_offset
        hit = get_plane_hit(context, event, plane=plane, offset=offset)
        if hit is None:
            return {'RUNNING_MODAL'}

        hit, self._snap_info = self._snap_pipeline(context, event, hit)

        # Left mouse - start/finish
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

        # Update during drawing
        if self._drawing:
            # Check if axis lock is perpendicular to plane
            perpendicular = False
            if self._axis_lock == "Z" and plane == "XY":
                perpendicular = True
            elif self._axis_lock == "Y" and plane == "XZ":
                perpendicular = True
            elif self._axis_lock == "X" and plane == "YZ":
                perpendicular = True
            
            if perpendicular:
                # Use perpendicular axis movement calculation (now robust for Z axis)
                cur = _compute_perpendicular_axis_movement(
                    context, event, self._start, self._axis_lock, self._mouse_start
                )
            else:
                # Use standard constraint
                cur = hit.copy()
                cur = _apply_axis_lock(self._start, cur, plane, self._axis_lock)
                if self._shift_down:
                    ax_a, ax_b = _plane_axes(plane)
                    d = cur - self._start
                    w = d.dot(ax_a)
                    h = d.dot(ax_b)
                    s = w if abs(w) >= abs(h) else h
                    cur = self._start + ax_a * s + ax_b * s
            
            self._end = cur
            
            # Calculate inferred axis for preview coloring
            self._inferred_axis = _infer_dominant_axis(self._start, self._end, plane, self._axis_lock)
            
            self._update_preview(context)
            self._update_header(context)

        return {'RUNNING_MODAL'}
#=============================================================================
# NEW: B - ARC TOOL (2-Point)
#=============================================================================
def _arc_header_status(stage, plane, typing, snap_info):
    stages = {
        "START": "Arc: Click start point | 1/2/3: Plane | Esc: Cancel",
        "END": "Arc: Click end point | 1/2/3: Plane | Esc: Cancel",
        "BULGE": f"Arc: Set bulge/radius | Type radius + Enter | LMB: Confirm | Esc: Cancel",
    }
    base = stages.get(stage, "Arc Tool")
    snap = " [SNAP]" if snap_info else ""
    t = f" | Type: {typing}" if typing else ""
    return f"{base} | Plane: {plane}{snap}{t}"
class SU_OT_arc_grid(bpy.types.Operator):
    """Draw an arc on the current drawing plane (2-Point method)"""
    bl_idname = "su.arc_grid"
    bl_label = "Arc (2-Point)"
    bl_options = {'REGISTER', 'UNDO'}
    def invoke(self, context, event):
        # Initialize instance variables
        self._tmp_obj = None
        self._bm = None
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
            self._tmp_obj = create_temp_drawing_object(context, prefix="SU_TMP_ARC")
            self._bm = bmesh.new()
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)
            self._register_draw_handlers(context)
            context.window_manager.modal_handler_add(self)
            _set_header(context, _arc_header_status(
                self._stage, context.scene.su_draw_plane, self._typing, None
            ))
            return {'RUNNING_MODAL'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start Arc tool: {e}")
            self._cleanup(context)
            return {'CANCELLED'}

    def _register_draw_handlers(self, context):
        if not GPU_AVAILABLE:
            return
        addon = context.preferences.addons.get(ADDON_ID)
        prefs = addon.preferences if addon else None
        if not prefs:
            return
        
        # Store reference to self for callback access
        op_ref = self
        self._draw_handler_key = f"arc_{id(self)}"
        
        # POST_PIXEL callback for snap marker
        def draw_snap_marker():
            snap_info = getattr(op_ref, '_snap_info', None)
            if snap_info and prefs.show_snap_marker:
                draw_snap_marker_simple(snap_info, prefs)
        
        # POST_VIEW callback for axis guides
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
                        context, anchor, context.scene.su_draw_plane, "FREE", prefs
                    )
        
        # POST_VIEW callback for colored preview (prioritize axis_lock for coloring)
        def draw_preview_colored():
            stage = getattr(op_ref, '_stage', 'START')
            start = getattr(op_ref, '_start', None)
            end = getattr(op_ref, '_end', None)
            axis_lock = getattr(op_ref, '_axis_lock', 'FREE')
            inferred_axis = getattr(op_ref, '_inferred_axis', 'FREE')
            
            if stage == "END" and start and end:
                # Prioritize axis_lock when locked for coloring (FIX for Z axis blue preview)
                axis = axis_lock if axis_lock != "FREE" else inferred_axis
                locked = (axis_lock != "FREE")
                if axis != "FREE":
                    draw_preview_line_colored(start, end, axis, locked)
        
        # Register all handlers
        try:
            handle_pixel = bpy.types.SpaceView3D.draw_handler_add(
                draw_snap_marker, (), 'WINDOW', 'POST_PIXEL'
            )
            handle_view = bpy.types.SpaceView3D.draw_handler_add(
                draw_axis_guides, (), 'WINDOW', 'POST_VIEW'
            )
            handle_preview = bpy.types.SpaceView3D.draw_handler_add(
                draw_preview_colored, (), 'WINDOW', 'POST_VIEW'
            )
            _draw_handlers[self._draw_handler_key] = (handle_pixel, handle_view, handle_preview)
        except Exception as e:
            print(f"[SU] Failed to register draw handlers: {e}")

    def _calculate_arc_geometry(self, context):
        if not self._start or not self._end:
            return []
        plane = context.scene.su_draw_plane
        ax_a, ax_b = _plane_axes(plane)
        chord = self._end - self._start
        chord_len = chord.length
        if chord_len < 1e-6:
            return []
        mid = (self._start + self._end) * 0.5
        if self._bulge_point:
            bulge_vec = self._bulge_point - mid
            if plane == "XY":
                chord_norm_in_plane = chord.cross(Vector((0, 0, 1)))
            elif plane == "XZ":
                chord_norm_in_plane = chord.cross(Vector((0, 1, 0)))
            else:
                chord_norm_in_plane = chord.cross(Vector((1, 0, 0)))
            if chord_norm_in_plane.length > 1e-6:
                chord_norm_in_plane.normalize()
                bulge_dist = bulge_vec.dot(chord_norm_in_plane)
            else:
                bulge_dist = bulge_vec.length
        else:
            bulge_dist = self._radius
        if abs(bulge_dist) < 1e-6:
            return [self._start, self._end]
        radius = (chord_len * chord_len) / (8.0 * abs(bulge_dist)) + abs(bulge_dist) / 2.0
        if plane == "XY":
            perp = Vector((-chord.y, chord.x, 0.0))
        elif plane == "XZ":
            perp = Vector((-chord.z, 0.0, chord.x))
        else:
            perp = Vector((0.0, -chord.z, chord.y))
        if perp.length > 1e-6:
            perp.normalize()
        else:
            return [self._start, self._end]
        center_dist = math.sqrt(max(0, radius * radius - (chord_len * 0.5) ** 2))
        center = mid + perp * center_dist * (1.0 if bulge_dist > 0 else -1.0)
        v1 = self._start - center
        v2 = self._end - center
        cross = v1.cross(v2)
        dot = v1.dot(v2)
        angle = math.atan2(cross.length, dot)
        if plane == "XY":
            if cross.z * bulge_dist < 0:
                angle = -angle
        elif plane == "XZ":
            if cross.y * bulge_dist > 0:
                angle = -angle
        else:
            if cross.x * bulge_dist < 0:
                angle = -angle
        addon = context.preferences.addons.get(ADDON_ID)
        prefs = addon.preferences if addon else None
        segments = prefs.arc_segments if prefs else DEFAULT_ARC_SEGMENTS
        segments = max(MIN_ARC_SEGMENTS, min(MAX_ARC_SEGMENTS, segments))
        points = []
        for i in range(segments + 1):
            t = i / segments
            current_angle = t * angle
            if plane == "XY":
                axis = Vector((0, 0, 1))
            elif plane == "XZ":
                axis = Vector((0, 1, 0))
            else:
                axis = Vector((1, 0, 0))
            cos_a = math.cos(current_angle)
            sin_a = math.sin(current_angle)
            rotated = (
                v1 * cos_a +
                axis.cross(v1) * sin_a +
                axis * axis.dot(v1) * (1 - cos_a)
            )
            points.append(center + rotated)
        return points

    def _update_preview(self, context):
        if not self._tmp_obj or not self._bm:
            return
        try:
            self._bm.clear()
            if self._stage == "START":
                self._tmp_obj.data.clear_geometry()
                return
            elif self._stage == "END":
                if self._start and self._end:
                    v1 = self._bm.verts.new(self._start)
                    v2 = self._bm.verts.new(self._end)
                    self._bm.edges.new((v1, v2))
            elif self._stage == "BULGE":
                points = self._calculate_arc_geometry(context)
                if len(points) >= 2:
                    prev_v = None
                    for p in points:
                        v = self._bm.verts.new(p)
                        if prev_v:
                            self._bm.edges.new((prev_v, v))
                        prev_v = v
            me = self._tmp_obj.data
            self._bm.to_mesh(me)
            me.update()
        except Exception:
            pass

    def _commit_arc(self, context):
        try:
            points = self._calculate_arc_geometry(context)
            if len(points) < 2:
                self.report({'WARNING'}, "Arc too small")
                return
            mesh = bpy.data.meshes.new("SU_Arc")
            obj = bpy.data.objects.new("SU_Arc", mesh)
            safe_link_object(context, obj)
            bm = bmesh.new()
            prev_v = None
            for p in points:
                v = bm.verts.new(p)
                if prev_v:
                    bm.edges.new((prev_v, v))
                prev_v = v
            bm.to_mesh(mesh)
            mesh.update()
            bm.free()
            reset_object_rotation(obj)
            addon = context.preferences.addons.get(ADDON_ID)
            prefs = addon.preferences if addon else None
            origin_mode = context.scene.su_origin_mode_override
            if origin_mode == "USE_PREF" and prefs:
                origin_mode = prefs.origin_mode
            apply_origin_placement(context, obj, origin_mode, self._start)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create arc: {e}")

    def _cleanup(self, context=None):
        if self._bm:
            try:
                self._bm.free()
            except Exception:
                pass
        self._bm = None
        if self._tmp_obj:
            remove_object_safe(self._tmp_obj)
        self._tmp_obj = None
        if self._draw_handler_key:
            unregister_draw_handler(self._draw_handler_key)
            self._draw_handler_key = None

    def _finish(self, context):
        _clear_header(context)
        self._cleanup(context)

    def modal(self, context, event):
        if context.area:
            context.area.tag_redraw()
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self._kd, self._meta, self._depsgraph = _kdtree_cache.get_or_build(context)
        
        # Axis lock - check before navigation (supports all keyboard layouts)
        axis_key = _detect_axis_key(event)
        if axis_key:
            # Arc doesn't use axis lock directly but should not pass through
            return {'RUNNING_MODAL'}
        
        # Pass through navigation (exclude axis keys)
        if event.type in _NAV_EVENTS and event.type not in {'X', 'Y', 'Z'}:
            return {'PASS_THROUGH'}
        
        if _handle_plane_keys(context, event):
            _set_header(context, _arc_header_status(
                self._stage, context.scene.su_draw_plane, self._typing, self._snap_info
            ))
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
            self._update_preview(context)
            _set_header(context, _arc_header_status(
                self._stage, context.scene.su_draw_plane, self._typing, self._snap_info
            ))
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and event.type == 'RET':
            if self._stage == "BULGE" and self._typing:
                radius = _parse_length(context, self._typing)
                if radius is not None and radius > 0:
                    self._radius = float(radius)
                    self._update_preview(context)
                self._typing = ""
                _set_header(context, _arc_header_status(
                    self._stage, context.scene.su_draw_plane, self._typing, self._snap_info
                ))
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and len(event.unicode) == 1:
            ch = event.unicode
            if ch.isdigit() or ch in ".-+mMcCiInNfFtT ":
                self._typing += ch
                _set_header(context, _arc_header_status(
                    self._stage, context.scene.su_draw_plane, self._typing, self._snap_info
                ))
            return {'RUNNING_MODAL'}
        plane = context.scene.su_draw_plane
        offset = context.scene.su_plane_offset
        hit = get_plane_hit(context, event, plane=plane, offset=offset)
        if hit is None:
            return {'RUNNING_MODAL'}
        self._snap_info = None
        if context.scene.su_enable_vertex_snap:
            snap_co, snap_info = find_vertex_snap(
                context, event, hit, self._kd, self._meta, self._depsgraph,
                max_px=float(context.scene.su_snap_px),
                max_world=MAX_SNAP_DISTANCE_WORLD
            )
            if snap_co is not None:
                hit = snap_co
                self._snap_info = snap_info
        if self._snap_info is None:
            addon = context.preferences.addons.get(ADDON_ID)
            prefs = addon.preferences if addon else None
            if prefs and prefs.grid_snap_enabled:
                hit = snap_to_grid(hit, prefs.grid_size)
        if self._stage == "BULGE":
            self._bulge_point = hit.copy()
            if self._start and self._end:
                mid = (self._start + self._end) * 0.5
                bulge_vec = self._bulge_point - mid
                self._radius = bulge_vec.length
            self._update_preview(context)
        elif self._stage == "END" and self._start:
            # Update temporary end point for preview coloring
            plane = context.scene.su_draw_plane
            
            # Check if axis lock is perpendicular to plane (Arc doesn't use axis lock but check anyway)
            perpendicular = False
            axis_lock = "FREE"  # Arc doesn't have axis lock, but keep for future
            if axis_lock == "Z" and plane == "XY":
                perpendicular = True
            elif axis_lock == "Y" and plane == "XZ":
                perpendicular = True
            elif axis_lock == "X" and plane == "YZ":
                perpendicular = True
            
            if perpendicular and self._mouse_start:
                self._end = _compute_perpendicular_axis_movement(
                    context, event, self._start, axis_lock, self._mouse_start
                )
            else:
                self._end = hit.copy()
            
            self._inferred_axis = _infer_dominant_axis(self._start, self._end, plane, "FREE")
            
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
            self._update_preview(context)
            _set_header(context, _arc_header_status(
                self._stage, context.scene.su_draw_plane, self._typing, self._snap_info
            ))
        return {'RUNNING_MODAL'}
#=============================================================================
# PUSH/PULL TOOL (Preserved from source)
#=============================================================================
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
    """Extrude faces along their normal (Push/Pull)"""
    bl_idname = "su.pushpull_modal"
    bl_label = "Push/Pull (Modal)"
    bl_options = {'REGISTER', 'UNDO'}
    @classmethod
    def poll(cls, context):
        return (
            context.active_object is not None
            and context.active_object.type == 'MESH'
            and context.mode == 'EDIT_MESH' 
        )

    def invoke(self, context, event):
        # Initialize instance variables
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
            _set_header(context, _pp_header_status(
                self._axis_lock, self._typing, None
            ))
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
            base2d = view3d_utils.location_3d_to_region_2d(
                region, rv3d, self._base_point_w
            )
            dir2d = view3d_utils.location_3d_to_region_2d(
                region, rv3d, self._base_point_w + self._dir_w
            )
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
                self._extruded_verts = [
                    ele for ele in geom if isinstance(ele, bmesh.types.BMVert)
                ]
                self._extruded_faces = [
                    ele for ele in geom if isinstance(ele, bmesh.types.BMFace)
                ]
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
            try:
                self._bm = bmesh.from_edit_mesh(self._mesh)
            except Exception:
                pass

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
                    n_w = (mw.to_3x3() @ f.normal)
                    if n_w.length < 1e-9:
                        continue
                    d_l = (inv3 @ n_w.normalized())
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

    def _cancel(self, context):
        self._restore_backup()
        try:
            if self._mesh:
                bmesh.update_edit_mesh(self._mesh, loop_triangles=True, destructive=True)
            self._bm = bmesh.from_edit_mesh(self._mesh) if self._mesh else None
        except Exception as e:
            print(f"[SU] Cancel update error: {e}")
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
        
        # Axis lock - MUST be before navigation pass-through (supports all keyboard layouts)
        axis_key = _detect_axis_key(event)
        if axis_key:
            self._axis_lock = "FREE" if self._axis_lock == axis_key else axis_key
            if self._state == "DRAGGING":
                self._apply_axis_lock_dir()
            _set_header(context, _pp_header_status(
                self._axis_lock, self._typing, self._snap_info, self._typing_mode
            ))
            return {'RUNNING_MODAL'}
        
        # Pass through navigation (exclude axis keys)
        if event.type in _NAV_EVENTS and event.type not in {'X', 'Y', 'Z'}:
            return {'PASS_THROUGH'}
        
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self._cancel(context)
            return {'CANCELLED'}
        if event.value == 'PRESS' and event.type in {'BACK_SPACE', 'DEL'}:
            self._typing = self._typing[:-1]
            _set_header(context, _pp_header_status(
                self._axis_lock, self._typing, self._snap_info, self._typing_mode
            ))
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and len(event.unicode) == 1:
            ch = event.unicode
            if ch.isdigit() or ch in ".-+mMcCiInNfFtT ":
                self._typing += ch
                self._typing_mode = True
                _set_header(context, _pp_header_status(
                    self._axis_lock, self._typing, self._snap_info, self._typing_mode
                ))
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
                self._axis_lock, self._typing, self._snap_info, self._typing_mode
            ))
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._state == "READY":
                self._start_extrude(context, event)
                _set_header(context, _pp_header_status(
                    self._axis_lock, self._typing, None, False
                ))
                return {'RUNNING_MODAL'}
            else:
                self._finish(context)
                return {'FINISHED'}
        if self._state == "DRAGGING":
            if self._typing_mode:
                _set_header(context, _pp_header_status(
                    self._axis_lock, self._typing, self._snap_info, True
                ))
                return {'RUNNING_MODAL'}
            self._scalar = self._compute_scalar_from_mouse(event)
            self._snap_info = None
            if context.scene.su_enable_vertex_snap and self._kd:
                if self._base_point_w and self._dir_w:
                    probe = self._base_point_w + self._dir_w * self._scalar
                    snap_co, snap_info = find_vertex_snap(
                        context, event, probe, self._kd, self._meta, self._depsgraph,
                        max_px=float(context.scene.su_snap_px),
                        max_world=MAX_SNAP_DISTANCE_WORLD
                    )
                    if snap_co is not None:
                        self._scalar = float((snap_co - self._base_point_w).dot(self._dir_w))
                        self._snap_info = snap_info
            self._move_extruded(self._scalar, context)
            _set_header(context, _pp_header_status(
                self._axis_lock, self._typing, self._snap_info, False
            ))
        return {'RUNNING_MODAL'}
#=============================================================================
# UI PANELS
#=============================================================================
class SU_PT_main(bpy.types.Panel):
    bl_label = "SketchUp Tools V2.5"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SketchUp"
    def draw(self, context):
        layout = self.layout
        scn = context.scene
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
        col.separator()
        col.prop(scn, "su_origin_mode_override", text="Origin")
        layout.separator()
        box = layout.box()
        box.label(text="Draw Tools", icon='GREASEPENCIL')
        col = box.column(align=True)
        col.operator("su.line_grid", text="Line Tool (L)", icon='IPO_LINEAR')
        col.operator("su.rect_grid", text="Rectangle Tool (R)", icon='MESH_PLANE')
        col.operator("su.arc_grid", text="Arc Tool (A)", icon='MESH_CIRCLE')
        layout.separator()
        box = layout.box()
        box.label(text="Model Tools (Edit Mode)", icon='EDITMODE_HLT')
        col = box.column(align=True)
        obj = context.active_object
        if obj and obj.type == 'MESH' and context.mode == 'EDIT_MESH':
            col.operator("su.pushpull_modal", text="Push/Pull (P)", icon='MOD_SOLIDIFY')
            row = col.row(align=True)
            row.prop(scn, "su_pushpull_individual", text="Individual Faces", toggle=True)
        else:
            col.label(text="⚠ Enter Edit Mode & select faces", icon='INFO')
        layout.separator()
        box = layout.box()
        box.label(text="Keyboard Shortcuts", icon='KEYINGSET')
        col = box.column(align=True)
        col.label(text="Global:")
        col.label(text="  • L - Line tool")
        col.label(text="  • R - Rectangle tool")
        col.label(text="  • A - Arc tool")
        col.label(text="  • P - Push/Pull (Edit Mode)")
        col.separator()
        col.label(text="During Tool:")
        col.label(text="  • 1/2/3 - Switch plane (XY/XZ/YZ)")
        col.label(text="  • X/Y/Z - Lock to axis")
        col.label(text="  • Shift - Auto axis lock (Line/Rect)")
        col.label(text="  • Type number + Enter - Precise input")
        col.label(text="  • Backspace - Step back / Clear input")
        col.label(text="  • Esc / RMB - Cancel")
class SU_MT_add_menu(bpy.types.Menu):
    bl_label = "SketchUp"
    bl_idname = "SU_MT_add_menu"
    def draw(self, context):
        layout = self.layout
        layout.operator_context = 'INVOKE_DEFAULT'
        layout.operator("su.line_grid", text="Line (Grid)", icon='IPO_LINEAR')
        layout.operator("su.rect_grid", text="Rectangle (Grid)", icon='MESH_PLANE')
        layout.operator("su.arc_grid", text="Arc (2-Point)", icon='MESH_CIRCLE')
        layout.separator()
        layout.operator("su.pushpull_modal", text="Push/Pull (Modal)", icon='MOD_SOLIDIFY')
def _draw_add_menu(self, context):
    self.layout.separator()
    self.layout.menu("SU_MT_add_menu", icon='OUTLINER_OB_EMPTY')
#=============================================================================
# KEYMAPS
#=============================================================================
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
#=============================================================================
# REGISTRATION
#=============================================================================
_CLASSES = [
    SU_Preferences,
    SU_OT_line_grid,
    SU_OT_rectangle_grid,
    SU_OT_arc_grid,
    SU_OT_pushpull_modal,
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
        print("[SU] SketchUp Tools V2.5 registered successfully")
    except Exception as e:
        _IMPORT_ERROR_TEXT = traceback.format_exc()
        print(f"[SU] Registration failed:\n{_IMPORT_ERROR_TEXT}")
        try:
            bpy.utils.register_class(SU_PT_error_panel)
        except Exception:
            pass
        raise
def unregister():
    global _registered, _IMPORT_ERROR_TEXT, _timer_scheduled, _draw_handlers
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
    print("[SU] SketchUp Tools V2.5 unregistered")
if __name__ == "__main__":
    register()