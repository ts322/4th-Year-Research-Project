#!/bin/bash
# Run OpenFOAM case from the latest generated geometry and open in ParaView

# Load OpenFOAM environment
source /usr/lib/openfoam/openfoam2506/etc/bashrc

# Find the most recent generated mesh folder from Shape_Generation.py
baseDir=$(ls -td "$HOME/ResearchProject/4th-Year-Research-Project/2D_Reactor/generate_mesh_1"/*/ | head -1)

# Create a timestamped results folder
timestamp=$(date +"%Y-%m-%d_%H-%M")
runDir="$HOME/ResearchProject/4th-Year-Research-Project/2D_Reactor/Results_$timestamp"
mkdir -p "$runDir"

echo "Base case (generated folder): $baseDir"
echo "Results will be stored in: $runDir"

# Copy the generated case into Results folder
cp -r "$baseDir"/* "$runDir/"

# Ensure ParaView can recognise this as a case
touch "$runDir/test.foam"

# Move into results folder
cd "$runDir" || exit 1

# Always remove any old mesh to avoid stale geometry
echo "Cleaning mesh..."
rm -rf constant/polyMesh || true

# Rebuild mesh from blockMeshDict
echo "Running blockMesh..."
blockMesh | tee log.blockMesh

# Check mesh quality
echo "Checking mesh..."
checkMesh | tee log.checkMesh

# Run solver
echo "Running pimpleFoam..."
pimpleFoam | tee log.pimpleFoam

# Convert Linux path to Windows path for ParaView
winPath=$(wslpath -w "$runDir/test.foam")
echo "Opening in ParaView: $winPath"

# Open results in ParaView
"/mnt/c/Program Files/ParaView 6.0.0/bin/paraview.exe" "$winPath"
