export updateWd,multWd,setSources,setSourcesSame,setDobs,setWd
function updateWd(pMis::Array{RemoteChannel},Dc::Array{RemoteChannel})
@sync begin
		for k=1:length(pMis)
	@async begin
			pMis[k] = remotecall_fetch(updateWd,pMis[k].where,pMis[k],Dc[k]);
		end
	end
end
return pMis;
end

function updateWd(pMisRF::RemoteChannel,Dc::RemoteChannel)
pMis  = take!(pMisRF)
Dc = fetch(Dc);
pMis.Wd = 1.0./(real(Dc - pMis.dobs) + 1e-3*mean(abs(pMis.dobs[:]))) + 1im./(imag(Dc - pMis.dobs) + 1e-3*mean(abs(pMis.dobs[:])));
put!(pMisRF,pMis)
return pMisRF;
end


function multWd(pMis::Array{RemoteChannel},beta::Float64)
@sync begin
		for k=1:length(pMis)
	@async begin
			pMis[k] = remotecall_fetch(multWd,pMis[k].where,pMis[k],beta);
		end
	end
end
return pMis;
end

function multWd(pMisRF::RemoteChannel,beta::Float64)
pMis  = take!(pMisRF)
pMis.Wd *= beta;
put!(pMisRF,pMis)
return pMisRF;
end

export setSources, setDobs

function setWd(pMis::Array{RemoteChannel},newWd::Array)
@sync begin
		for k=1:length(pMis)
	@async begin
			## TODO: make sure sources are replicated for different frequencies
			pMis[k] = remotecall_fetch(setWd,pMis[k].where,pMis[k],newWd[k]);
		end
	end
end
return pMis;
end

function setWd(pMisRF::RemoteChannel,newWd)
pMis  = take!(pMisRF)
pMis.Wd = newWd;
put!(pMisRF,pMis)
return pMisRF;
end
function setDobs(pMis::Array{RemoteChannel},newDobs::Array)
@sync begin
		for k=1:length(pMis)
	@async begin
			## TODO: make sure sources are replicated for different frequencies
			pMis[k] = remotecall_fetch(setDobs,pMis[k].where,pMis[k],newDobs[k]);
		end
	end
end
return pMis;
end

function setDobs(pMisRF::RemoteChannel,newDobs)
pMis  = take!(pMisRF)
pMis.dobs = newDobs;
put!(pMisRF,pMis)
return pMisRF;
end

function setSourcesSame(pMis::Array{RemoteChannel},newSources)
@sync begin
		for k=1:length(pMis)
	@async begin
			## TODO: make sure sources are replicated for different frequencies
			pMis[k] = remotecall_fetch(setSources,pMis[k].where,pMis[k],newSources);
		end
	end
end
return pMis;
end

function setSources(pMis::Array{RemoteChannel},newSources)
@sync begin
		for k=1:length(pMis)
	@async begin
			## TODO: make sure sources are replicated for different frequencies
			pMis[k] = remotecall_fetch(setSources,pMis[k].where,pMis[k],newSources[k]);
		end
	end
end
return pMis;
end

function setSources(pMisRF::RemoteChannel,newSources)
pMis  = take!(pMisRF)
pMis.pFor.Sources = newSources;
put!(pMisRF,pMis)
return pMisRF;
end


export getWd
function getWd(pMis::Array{RemoteChannel})
Wd = Array{Array{ComplexF64}}(undef,length(pMis))
@sync begin
		for k=1:length(pMis)
	@async begin
			Wd[k] = remotecall_fetch(getWd,pMis[k].where,pMis[k]);
		end
	end
end
return Wd;
end

function getWd(pMisRF::RemoteChannel)
pMis  = fetch(pMisRF)
return pMis.Wd;
end

export getDobs
function getDobs(pMis::Array{RemoteChannel})
Dobs = Array{Array{ComplexF64}}(undef,length(pMis))
@sync begin
		for k=1:length(pMis)
	@async begin
			Dobs[k] = remotecall_fetch(getDobs,pMis[k].where,pMis[k]);
		end
	end
end
return Dobs;
end

function getDobs(pMisRF::RemoteChannel)
pMis  = fetch(pMisRF)
return pMis.dobs;
end
