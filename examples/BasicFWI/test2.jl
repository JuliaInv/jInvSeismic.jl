using JLD
bb = SourcesSubIndFWI[1:2]

qq= Q[:,bb[1]]

qq2 = Q[:,bb[2]]

ans = zeros(size(Q))

ans[:,bb[1]] = qq

ans[:,bb[2]] = qq2

ans == Q

dobs = load("Merged.jld", "dobs")
wd = load("Merged.jld", "wd")
src = load("Merged.jld", "src")
