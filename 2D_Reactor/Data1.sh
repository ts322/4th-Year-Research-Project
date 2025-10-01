#!/bin/bash
# Run swakless test case in Results_Date_Time folder

# Load OpenFOAM environment
source /usr/lib/openfoam/openfoam2506/etc/bashrc

# Base case: hardcoded generated folder
baseDir="$HOME/ResearchProject/4th-Year-Research-Project/2D_Reactor/generate_mesh_1/2025_10_01_12_29_47_2fef194b42f3495b8d4feaad721b61bf"

# Results directory with timestamp
timestamp=$(date +"%Y-%m-%d_%H-%M")
runDir="$HOME/ResearchProject/4th-Year-Research-Project/2D_Reactor/Results_$timestamp"
mkdir -p "$runDir"

echo "Copying from $baseDir to $runDir"
cp -r "$baseDir/"* "$runDir/"

# Run simulation
cd "$runDir" || exit 1
blockMesh
checkMesh -allGeometry -allTopology   # add this single guard
pimpleFoam

# Ensure test.foam exists for ParaView
touch "$runDir/test.foam"

# Convert runDir path to Windows format for ParaView
winPath=$(wslpath -w "$runDir/test.foam")
echo "Opening in ParaView: $winPath"

# Open results in ParaView
"/mnt/c/Program Files/ParaView 6.0.0/bin/paraview.exe" "$winPath"
