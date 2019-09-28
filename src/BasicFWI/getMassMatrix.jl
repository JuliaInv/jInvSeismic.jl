export getMassMatrix

"""
	M = getMassMatrix(m::Vector,gamma::Vector,Mesh::RegularMesh)

	Input:
		m     - model
		gamma - attenuation
		Mesh  - regular mesh

	Output
		M::SparseMatrixCSC
"""
function getMassMatrix(m::Vector,gamma::Vector,Mesh::RegularMesh)

    An2cc = getNodalAverageMatrix(Mesh)
    # println("AN2cc size")
    # println(size(An2cc))
    # println(size(m))
    # println(size(gamma))
    M     = sdiag(An2cc'*(m.*(1 .- 1im*gamma)))

    return M
end
