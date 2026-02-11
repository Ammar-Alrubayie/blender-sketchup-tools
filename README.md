# Blender SketchUp Tools (Grid Mode) – Enhanced

SketchUp-like modeling tools for Blender 4.5 LTS  
Designed for fast architectural and technical modeling.

---

# 🚀 Installation

## Method 1 – Direct Install (Recommended)

1. Open **Blender 4.5**
2. Go to:

   Edit → Preferences → Add-ons

3. Click **Install**
4. Select:

   sketchup-tools.py

5. Enable the add-on

You will now find the tools in:

3D View → Sidebar (Press N) → SketchUp

---

## Method 2 – Manual Folder Installation

1. Create a folder:

   sketchup_tools_grid

2. Rename the file to:

   __init__.py

3. Place it inside:

   Windows:
   C:\Users\YourName\AppData\Roaming\Blender Foundation\Blender\4.5\scripts\addons\

4. Restart Blender
5. Enable the add-on in Preferences

---

# 📌 How to Use

Open 3D View → Press **N** → Go to **SketchUp tab**

Available tools:

• Line  
• Rectangle  
• Arc  
• Plane Selection (XY / XZ / YZ)

---

# ✏️ Line Tool

## Usage

1. Click Line
2. Click start point
3. Move mouse
4. Click to confirm

## Features

• Type exact length while drawing:
  
  2.35m

• Axis lock:

  Press X, Y, or Z

• Shift = auto axis lock (SketchUp style)

• Smart vertex snapping

---

# ▭ Rectangle Tool

## Usage

1. Click Rectangle
2. Click first corner
3. Drag
4. Click to confirm

## Enter Dimensions

While dragging, type:

  2.5m , 1.2m

Press Enter.

## Features

• Automatic square with Shift  
• Axis locking  
• Works on XY / XZ / YZ planes  
• Smart origin placement  

---

# ◯ Arc Tool (2-Point)

## Usage

1. Select start point
2. Select end point
3. Move mouse to define curvature
4. Click to confirm

## Features

• Adjustable bulge  
• Custom segment count  
• Smart snapping  

---

# 🧭 Plane Selection

Press:

1 → XY Plane  
2 → XZ Plane  
3 → YZ Plane  

All drawing tools respect selected plane.

---

# 🎯 Snapping System

• Vertex snapping via KDTree  
• Visual snap marker  
• Fast and optimized  

---

# ⌨️ Shortcuts

X / Y / Z → Axis Lock  
Shift → Auto axis constraint  
Enter → Confirm typed value  
Esc → Cancel  
1 / 2 / 3 → Change plane  

---

# ⚙️ Add-on Settings

Edit → Preferences → Add-ons → SketchUp Tools

Options include:

• Snap strength  
• Arc segments  
• Grid size  
• Axis guide visibility  
• Hotkey enable/disable  

---

# 🏗 Technical Overview

Built using:

• Blender bmesh API  
• GPU draw handlers  
• KDTree snapping system  
• Non-destructive temporary geometry  
• Smart origin logic  

Optimized for architectural modeling workflows.

---

# 📦 Blender Version

Tested on:

Blender 4.5 LTS

---

# 📜 License

MIT License
