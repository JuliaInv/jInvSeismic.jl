using  jInv.Mesh
using DelimitedFiles
using  jInv.Utils
using jInvSeismic.FWI
using jInvSeismic.Utils

function readModelAndGenerateMeshMrefOverthrust(readModelFolder::String,modelFilename::String,dim::Int64,pad::Int64,domain::Vector{Float64},newSize::Vector=[],velBottom::Float64=1.75,velHigh::Float64=2.9)
########################## m,mref are in Velocity here. ###################################

if dim==2
	# SEGmodel2Deasy.dat
	m = readdlm(string(readModelFolder,"/",modelFilename));
	m = m*1e-3;
	m = Matrix(m');
	mref = getSimilarLinearModel(m,velBottom,velHigh);
else
	# 3D SEG slowness model
	file = matopen(string(readModelFolder,"/",modelFilename)); DICT = read(file); close(file);
	m = DICT["VELs"];
	m = m*1e-3;
	mref = getSimilarLinearModel(m,velBottom,velHigh);
end

sea_level = 1.5;
sea = abs.(m[:] .- 1.5) .< 1e-3;
mref[sea] = m[sea];
if newSize!=[]
	m    = expandModelNearest(m,   collect(size(m)),newSize);
	mref = expandModelNearest(mref,collect(size(mref)),newSize);
end

Minv = getRegularMesh(domain,collect(size(m)));


(mPadded,MinvPadded) = addAbsorbingLayer(m,Minv,pad);
(mrefPadded,MinvPadded) = addAbsorbingLayer(mref,Minv,pad);

mPaddedNoSalt = copy(mPadded);
mPaddedNoSalt[mPaddedNoSalt .> 5.5] .= 5.5;

sea_level = 1.5;
sea = abs.(mPadded[:] .- 1.5) .< 1e-2;

file = matopen(string("examples/mrefOverthrust3D.mat")); DICT = read(file); close(file);
mrefPadded = DICT["VELs"];

N = prod(MinvPadded.n);
boundsLow  = minimum(mPadded);
boundsHigh = maximum(mPadded);

boundsLow  = ones(N)*boundsLow;
boundsLow = convert(Array{Float32},boundsLow);
boundsHigh = ones(N)*boundsHigh;
boundsHigh = convert(Array{Float32},boundsHigh);

return (mPadded,MinvPadded,mrefPadded,boundsHigh,boundsLow);
end


dim     = 3;
pad     = 16;
jumpSrc    = 10;
jumpRcv = 4;
offset  = 1000;
domain = [0.0,7.5,0.0,7.5,0.0,4.65]; # without the pad for Marmousi 1
newSize = [172,172,108];
modelDir = pwd();

(m,Minv,mref,boundsHigh,boundsLow) = readModelAndGenerateMeshMrefOverthrust(modelDir,"examples/overthrust3D_small.mat",dim,pad,domain,newSize,1.25,4.0);
