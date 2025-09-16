#Create a folder called Results_Date_Time and pastes initial conditions in it
source /usr/lib/openfoam/openfoam2506/etc/bashrc

baseDir="$HOME/ResearchProject/2D_Reactor/swakless-1"
timestamp=$(date +"%Y-%m-%d_%H-%M")
runDir="$HOME/ResearchProject/2D_Reactor/Results_$timestamp"
mkdir -p "$runDir"

#Copy all files from swakless-1 to Results_Date_Time
cp -r "$baseDir/"* "$runDir/"

#Run Simulation using pimpleFoam solver
cd "$runDir" || exit 1
blockMesh
pimpleFoam

#Opens paraview with the results file
"/mnt/c/Program Files/ParaView 6.0.0/bin/paraview.exe" \
  "\\wsl.localhost\Ubuntu\home\ts322\ResearchProject\2D_Reactor\Results_$timestamp\test.foam"
