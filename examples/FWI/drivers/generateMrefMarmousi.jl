using  jInv.Mesh
using DelimitedFiles
using  jInv.Utils
using jInvSeismic.FWI
using jInvSeismic.Utils

function readModelAndGenerateMeshMrefMarmousi(readModelFolder::String,modelFilename::String,dim::Int64,pad::Int64,domain::Vector{Float64},newSize::Vector=[],velBottom::Float64=1.75,velHigh::Float64=2.9)
########################## m,mref are in Velocity here. ###################################

if dim==2
	# SEGmodel2Deasy.dat
	m = readdlm(string(readModelFolder,"/",modelFilename));
	m = m*1e-3;
	m = Matrix(m');
	mref = getSimilarLinearModel(m,velBottom,velHigh);
else
	# 3D SEG slowness model
	# modelFilename = 3Dseg256256128.mat
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
mPaddedNoSalt[mPaddedNoSalt .> 3.8] .= 3.8;

mrefPadded = smoothModel(mPaddedNoSalt,[],250);
sea_level = 1.5;
sea = abs.(mPadded[:] .- 1.5) .< 1e-2;
mrefPadded[sea] = mPadded[sea];

mrefRowsNum = size(mrefPadded, 1)
mrefPadded = repeat(mean(mrefPadded,dims=1), outer=(mrefRowsNum, 1))

N = prod(MinvPadded.n);
boundsLow  = minimum(mPadded);
boundsHigh = maximum(mPadded);

boundsLow  = ones(N)*boundsLow;
boundsLow = convert(Array{Float32},boundsLow);
boundsHigh = ones(N)*boundsHigh;
boundsHigh = convert(Array{Float32},boundsHigh);

return (mPadded,MinvPadded,mrefPadded,boundsHigh,boundsLow);
end




dim     = 2;
pad     = 50;
jumpSrc    = 4;
jumpRcv = 1
offset  = 1000;
domain = [0.0,9.192,0.0,2.904]; # without the pad for Marmousi 1
newSize = [550,200];
modelDir = pwd();
(m,Minv,mref,boundsHigh,boundsLow) = readModelAndGenerateMeshMrefMarmousi(modelDir,"examples/MarmousiModelWithoutPad.dat",dim,pad,domain,newSize,1.25,4.0);
