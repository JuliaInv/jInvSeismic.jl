export computeHinvTRec

function computeHinvTRec(pMis::Array{RemoteChannel})
HinvTRec = Array{Array{ComplexF64}}(undef,length(pMis))
@sync begin
	@async begin
		for k=1:length(pMis)
			HinvTRec[k] = remotecall_fetch(computeHinvTRec,pMis[k].where,pMis[k]);
		end
	end
end
return HinvTRec;
end


function computeHinvTRec(pMisRF::RemoteChannel)
pMis = fetch(pMisRF)
HinvTP, = solveLinearSystem(spzeros(ComplexF64,0,0),complex(Matrix(pMis.pFor.Receivers)),pMis.pFor.ForwardSolver,1);
return HinvTP;
end
