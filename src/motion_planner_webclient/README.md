# Motion Planner — Interactive Web Demo

A fully self-contained browser demo that runs the `motion_planner_core` library via [Pyodide](https://pyodide.org/) (Python compiled to WebAssembly). No server-side Python needed.

## Usage

```bash
cd src/motion_planner_webclient
./serve.sh
# Open http://localhost:8080
```

Or for GitHub Pages: just push this folder — it's static HTML + Python files.

## Features

- Click to place **waypoints** (numbered, connected by dashed lines)
- Click to place **obstacles** (red circles, configurable radius)
- Hit **Plan & Simulate** to run the full pipeline in-browser:
  - Costmap generation from obstacles
  - A* path planning between waypoints
  - Cubic spline smoothing
  - Trapezoidal velocity profile
  - Pure Pursuit kinematic simulation
- See planned path (pink), smoothed path (blue), robot trajectory (green)
- Adjust parameters: velocity, acceleration, smoothing samples, lookahead distance
- Undo, clear, switch between waypoint/obstacle modes

## How It Works

Pyodide loads NumPy + SciPy as WASM, then fetches the `motion_planner_core` Python modules from the `core/` symlink. The entire pipeline runs client-side in the browser.

## For GitHub Pages Deployment

Replace the `core/` symlink with actual copies of the Python files:

```bash
rm core
cp -r ../motion_planner_core/motion_planner_core core
```

Then push to a `gh-pages` branch or enable Pages on the repo.
