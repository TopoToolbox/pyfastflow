# 3D Terrain Demo with PyFastFlow

A comprehensive demonstration of Modern OpenGL 3D terrain rendering using PyFastFlow's GPU-accelerated Perlin noise generation.

## Features

- **Real-time 3D Terrain Rendering**: High-performance mesh rendering with ModernGL
- **GPU-Accelerated Noise**: Uses PyFastFlow's Taichi-based Perlin noise for realistic terrain generation
- **Interactive Camera**: RTS-style camera controls with orbit, pan, and zoom
- **Dynamic Parameters**: Real-time ImGui interface for adjusting noise generation parameters
- **Normal-based Lighting**: Simple but effective lighting system with height-based coloring

## Controls

- **Left Mouse Button**: Rotate camera around center point
- **Right Mouse Button**: Pan the center of rotation (relative to current view)
- **Mouse Wheel**: Zoom in/out
- **ImGui Panel**: Adjust Perlin noise parameters in real-time

## Technical Details

- **Mesh Resolution**: 512 nodes on the longest dimension, maintains aspect ratio
- **Shaders**: Custom GLSL vertex/fragment shaders for heightmap displacement and lighting
- **Height Displacement**: GPU-based vertex displacement using heightmap texture
- **Normal Calculation**: Real-time normal calculation using finite differences on heightmap
- **Color Mapping**: Height-based terrain coloring (valleys to peaks)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure PyFastFlow is installed (from parent directory):
```bash
pip install -e ../..
```

## Usage

Run the demo:
```bash
python terrain_demo.py
```

## Perlin Noise Parameters

- **Frequency**: Controls the scale of terrain features (higher = more detail)
- **Octaves**: Number of noise layers combined (more = finer details)
- **Persistence**: How much each octave contributes (affects roughness)
- **Amplitude**: Overall height scale of the terrain
- **Seed**: Random seed for reproducible terrain generation

## Requirements

- Python 3.7+
- Modern OpenGL 3.3+ compatible graphics card
- PyFastFlow with Taichi backend

## Architecture

The demo consists of several key components:

1. **TerrainRenderer**: Manages mesh generation, shader compilation, and rendering
2. **Camera**: RTS-style camera with orbit controls around a target point  
3. **Shader Pipeline**: Vertex shader for heightmap displacement, fragment shader for lighting
4. **ImGui Integration**: Real-time parameter adjustment interface
5. **PyFastFlow Integration**: GPU-accelerated Perlin noise generation