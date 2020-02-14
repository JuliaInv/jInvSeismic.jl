export updateWd,multWd,setSources
function updateWd(pMis::Array{RemoteChannel},Dc::Array{RemoteChannel})
@sync begin
	@async begin
		for k=1:length(pMis)
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
	@async begin
		for k=1:length(pMis)
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

export setSources

function setSources(pMis::Array{RemoteChannel},newSources::Array)
@sync begin
	@async begin
		for k=1:length(pMis)
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
	@async begin
		for k=1:length(pMis)
			Wd[k] = remotecall_fetch(getWd,pMis[k].where,pMis[k]);
		end
	end
end
return pMis;
end

function getWd(pMisRF::RemoteChannel)
pMis  = fetch(pMisRF)
return pMis.Wd;
end

export getDobs
function getDobs(pMis::Array{RemoteChannel})
Dobs = Array{Array{ComplexF64}}(undef,length(pMis))
@sync begin
	@async begin
		for k=1:length(pMis)
			Dobs[k] = remotecall_fetch(getDobs,pMis[k].where,pMis[k]);
		end
	end
end
return pMis;
end

function getDobs(pMisRF::RemoteChannel)
pMis  = fetch(pMisRF)
return pMis.dobs;
end



