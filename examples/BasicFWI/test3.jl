
n = 20

newSrcInd = [Int[] for i=1:10]

for i=1:n
	append!(newSrcInd[(i % 10) + 1], i)
end
