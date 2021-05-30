export expandModelNearest, getSimilarLinearModel, addAbsorbingLayer
export addAbsorbingLayer, smoothModel, smooth3
export velocityToSlowSquared,slowSquaredToVelocity,velocityToSlow,slowToSlowSquared,slowSquaredToSlow
export slowToLeveledSlowSquared,getModelInvNewton
using Statistics
using jInv.Mesh


function slowToLeveledSlowSquared(s,mid::Float64 = 0.32,a::Float64 = 0.0,b::Float64 = 0.05)
d = (b-a)./2.0;
dinv = 200;
tt = dinv.*(mid-s);
t = -d.*(tanh(tt).+1) + a;
dt = (dinv*d)*(sech(tt)).^2 .+ 1;
# up until here it's just the slowness
dt = spdiagm(2.0.*(t+s).*dt);
t = (t + s).^2;
return t,dt
end


function getModelInvNewton(m,modFun::Function,m0 = copy(m))
# m0: initial guess for the model inverse
# modFun: the model function to invert.
err_prev = Inf;
s_prev = copy(m0);
s = m0;
for k=1:50
    k
    (fs,dfs) = modFun(s);
	err = vecnorm(fs - m,Inf);
    println(err)
	if err < 1e-5
        break;
    end
    err_prev = err;
	s_prev[:] = s;
	s = s - 0.4*(dfs\(fs - m));
end
return s;
end

function velocityToSlowSquared(v::Array)
s = (1.0./(v.+1e-16)).^2
ds = sparse(Diagonal((-2.0)./(v[:].^3)));
return s,ds
end

function slowSquaredToVelocity(s::Array)
m = 1.0./sqrt.(s.+1e-16);
dm = sparse(Diagonal((-0.5*(1.0/(s[:].^(3.0/2.0))))));
return m,dm
end

function velocityToSlow(v::Array)
s = (1.0./(v.+1e-16))
ds = sparse(Diagonal(((-1.0)./(v[:].^2))));
return s,ds
end


function slowToSlowSquared(v::Array)
s = v.^2;
ds = sparse(Diagonal((2.0.*v[:])));
return s,ds
end

function slowSquaredToSlow(v::Array)
s = sqrt.(v);
ds = sparse(Diagonal((0.5./s[:])));
return s,ds
end


function expandModelNearest(m,n,ntarget)
if length(size(m))==2
	mnew = zeros(Float64,ntarget[1],ntarget[2]);
	for j=1:ntarget[2]
		for i=1:ntarget[1]
			jorig = convert(Int64,ceil((j/ntarget[2])*n[2]));
			iorig = convert(Int64,ceil((i/ntarget[1])*n[1]));
			mnew[i,j] = m[iorig,jorig];
		end
	end
elseif length(size(m))==3
	mnew = zeros(Float64,ntarget[1],ntarget[2],ntarget[3]);
	for k=1:ntarget[3]
		for j=1:ntarget[2]
			for i=1:ntarget[1]
				korig = convert(Int64,floor((k/ntarget[3])*n[3]));
				jorig = convert(Int64,floor((j/ntarget[2])*n[2]));
				iorig = convert(Int64,floor((i/ntarget[1])*n[1]));
				mnew[i,j,k] = m[iorig,jorig,korig];
			end
		end
	end
end
return mnew
end

function getSimilarLinearModel(m::Array{Float64},mtop::Float64=0.0,mbottom::Float64=0.0)
# m here is assumed to be a velocity model.

if length(size(m))==2
	(nx,nz) = size(m);
	m_vel = copy(m) ;
	if mtop==0.0
		mtop = m_vel[1:10,5:6];
		mtop = Statistics.mean(mtop[:]);
		println("Mref top = ",mtop);
	end
	if mbottom==0.0
		mbottom = m_vel[1:10,end-10:end];
		mbottom = Statistics.mean(mbottom[:]);
		println("Mref bottom = ",mbottom);
	end
	m_vel = ones(nx)*range(mtop,stop=mbottom,length=nz)';
	mref = m_vel;
elseif length(size(m))==3
	(nx,ny,nz) = size(m);
	m_vel = copy(m);
	if mtop==0.0
		mtop = m_vel[1:10,:,5:15];
		mtop = Statistics.mean(mtop[:]);
	end
	if mbottom==0.0
		mbottom = m_vel[1:10,:,end-10:end];
		mbottom = Statistics.mean(mbottom[:]);
	end
	lin = range(mtop,stop=mbottom,length=nz);
	m_vel = copy(m);
	Oplane = ones(nx,ny);
	for k=1:nz
		m_vel[:,:,k] = lin[k]*Oplane;
	end
	mref = m_vel;
else
	error("Unhandled Dimensions");
end
return mref;
end


function addAbsorbingLayer2D(m::Array{Float64},pad::Int64)
if pad<=0
	return m;
end
mnew = zeros(size(m,1)+2*pad,size(m,2)+pad);
mnew[pad+1:end-pad,1:end-pad] = m;
mnew[1:pad,1:end-pad] = repeat(m[[1],:],pad,1);
mnew[end-pad+1:end,1:end-pad] = repeat(m[[end],:],pad,1);
mnew[:,end-pad+1:end] = repeat(mnew[:,end-pad],1,pad);
return mnew;
end


function addAbsorbingLayer(m::Array{Float64},Msh::RegularMesh,pad::Int64)
if pad<=0
	return m,Msh;
end
Omega = Msh.domain;

if length(size(m))==2
	mnew = addAbsorbingLayer2D(m,pad);
	MshNew = getRegularMesh([Omega[1],Omega[2] + 2*pad*Msh.h[1],Omega[3],Omega[4]+pad*Msh.h[2]],Msh.n.+[2*pad,pad]);
elseif length(size(m))==3
	mnew = zeros(size(m,1)+2*pad,size(m,2)+2*pad,size(m,3)+pad);
	mnew[pad+1:end-pad,pad+1:end-pad,1:end-pad] = m;

	extendedPlane1 = addAbsorbingLayer2D(reshape(m[1,:,:],size(m,2),size(m,3)),pad);
	extendedPlaneEnd = addAbsorbingLayer2D(reshape(m[end,:,:],size(m,2),size(m,3)),pad);

	for k=1:pad
		mnew[k,:,:] = extendedPlane1;
		mnew[end-k+1,:,:] = extendedPlaneEnd;
		mnew[pad+1:end-pad,end-k+1,1:end-pad] = m[:,end,:];
		mnew[pad+1:end-pad,k,1:end-pad] = m[:,1,:];
	end
	t = mnew[:,:,end-pad];
	for k=1:pad
		mnew[:,:,end-pad+k] = t;
	end
	MshNew = getRegularMesh([Omega[1],Omega[2] + 2*pad*Msh.h[1],Omega[3],Omega[4] + 2*pad*Msh.h[2],Omega[5],Omega[6]+pad*Msh.h[2]],Msh.n.+[2*pad,2*pad,pad]);
end

return mnew,MshNew;
end



function smoothModel(m,Mesh,times = 0)
	ms = addAbsorbingLayer2D(m,times);
	for k=1:times
		for j = 2:size(ms,2)-1
			for i = 2:size(ms,1)-1
				@inbounds ms[i,j] = (2*ms[i,j] + (ms[i-1,j-1]+ms[i-1,j]+ms[i-1,j+1]+ms[i,j-1]+ms[i,j+1]+ms[i+1,j-1]+ms[i+1,j]+ms[i,j+1]))/10.0;
			end
		end
	end
	return ms[(times+1):(end-times),1:end-times];
end

function smooth3(m,Mesh,times = 0)
	pad = 50
	println("Smoothing ", times," times");
	ms, = addAbsorbingLayer(m, Mesh, pad)
	for k=1:times
		for l = 2:size(ms,3)-1
			for j = 2:size(ms,2)-1
				for i = 2:size(ms,1)-1
					@inbounds ms[i,j,l] = (2*ms[i,j,l] +
					(ms[i,j,l+1] + ms[i,j,l-1] + ms[i,j-1,l] + ms[i,j-1,l-1] + ms[i,j-1,l+1] + ms[i,j+1,l] + ms[i,j+1,l-1] + ms[i,j+1,l+1] +
					ms[i-1,j,l] + ms[i-1,j,l+1] + ms[i-1,j,l-1] + ms[i-1,j-1,l] + ms[i-1,j-1,l-1] + ms[i-1,j-1,l+1] + ms[i-1,j+1,l] + ms[i-1,j+1,l-1] + ms[i-1,j+1,l+1] +
					ms[i+1,j,l] + ms[i+1,j,l+1] + ms[i+1,j,l-1] + ms[i+1,j-1,l] +
					 ms[i+1,j-1,l-1] + ms[i+1,j-1,l+1] + ms[i+1,j+1,l] + ms[i+1,j+1,l-1] + ms[i+1,j+1,l+1]))/28.0;
				end
			end
		end


	end
	return ms[(pad+1):(end-pad),(pad+1):(end-pad),1:end-pad];
end
