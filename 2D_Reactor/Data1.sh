#!/bin/bash
# Run OpenFOAM case from the latest generated geometry

# Load OpenFOAM environment
source /usr/lib/openfoam/openfoam2506/etc/bashrc

# Find the most recent generated mesh folder
baseDir=$(ls -td "$HOME/ResearchProject/4th-Year-Research-Project/2D_Reactor/generate_mesh_1"/*/ | head -1)

timestamp=$(date +"%Y-%m-%d_%H-%M")
runDir="$HOME/ResearchProject/4th-Year-Research-Project/2D_Reactor/Results_$timestamp"
mkdir -p "$runDir"

# Copy all files (including test.foam) from generated mesh into Results folder
cp -r "$baseDir"* "$runDir/"
cp "$baseDir/test.foam" "$runDir/"

# Run Simulation using pimpleFoam solver
cd "$runDir" || exit 1
blockMesh
pimpleFoam

# Ensure ParaView has a .foam file in the results folder
touch "$runDir/test.foam"

# Open in ParaView (WSL path translation)
"/mnt/c/Program Files/ParaView 6.0.0/bin/paraview.exe" \
  "\\wsl.localhost\Ubuntu\home\ts322\ResearchProject\4th-Year-Research-Project\2D_Reactor\Results_$timestamp\test.foam"
