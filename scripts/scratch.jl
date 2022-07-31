using MLDatasets, MLUtils

traindir(args...) = datadir("exp_raw", "train", args...)
labeldir(args...) = datadir("exp_raw", "masks", args...)

# cases = readdir(datadir("exp_raw", "train"))

train = []
labels = []

# for case in cases
#     for day in days

#     days = vcat(days, readdir(traindir(case)))
# end

train = []
for (root, dirs, files) in walkdir(traindir(); topdown=false)
    for file in files
        #println(joinpath(root, file))
        train = vcat(train, joinpath(root, file)) # path to files
    end
end

labels = []
for (root, dirs, files) in walkdir(labeldir(); topdown=false)
    for file in files
        #println(joinpath(root, file))
        train = vcat(labels, joinpath(root, file)) # path to files
    end
end

X_files = FileDataset(identity, traindir(), "*/*/*/*")
Y_files = FileDataset(identity, labeldir(), "*/*/*/*")
files = (X_files, Y_files,)

using FileIO
function loadimagemaskpair((x, y,))
    return (
        FileIO.load(x),
        FileIO.load(y),
    )
end


# function loadimagemaskpair2((x, y,))
#     return (
#         reshape(channelview(FileIO.load(x)), 266, 266, 1, 1),
#         reshape(channelview(FileIO.load(y)), 266, 266, 3, 1),
#     )
# end

data = mapobs(loadimagemaskpair, files)

data_loader = DataLoader(data; batchsize=2)

reinterpret(Float32, X[1])

using FileIO, Images
for (x, y) in data_loader
   @assert size(cat(reshape.(channelview.(x), 266, 266, 1); dims=4)) == (266, 266, 3) || size(x) == (28, 28, 1, 96)
   @assert size(cat(reshape.(channelview.(y), 266, 266, 3); dims=4)) == (266, 266, 3) || size(y) == (10, 96)
end

for (x, y) in data_loader
   @assert size(x) == (266, 266, 1) || size(x) == (28, 28, 1, 96)
   @assert size(y) == (266, 266, 3) || size(y) == (10, 96)
end

for (x, y) in data_loader
    println(size(cat(imresize.(x, 100, 100); dims=3)))
    println(size(cat(imresize.(y, 100, 100); dims=3)))
end


for (x, y) in BatchView(data; batchsize=2)
    println(size.(imresize.(x, 100, 100)))
    println(size.(imresize.(y, 100, 100)))
end

### ---- THIS ONE ---- ###
function b2a(a)
    return cat(a[1], a[2]; dims=4)
end

function transform256(array, channels)
    return float(b2a(reshape.(channelview.(imresize.(array, 256, 256)), 256, 256, channels)))
end

for (x, y) in data_loader
    println(size.((transform256(x, 1), transform256(y, 3),)))
end

println(size.((transform256(first(data_loader)[1], 1), transform256(first(data_loader)[2], 3),)))
println(typeof((transform256(first(data_loader)[1], 1), transform256(first(data_loader)[2], 3),)))

for (x, y) in BatchView(data; batchsize=2)
    println(size(b2a(reshape.(channelview.(imresize.(x, 256, 256)), 256, 256, 1))))
    println(size(b2a(reshape.(channelview.(imresize.(y, 256, 256)), 256, 256, 3))))
end
### ---- END ---- ####
### ACTUAL DATA LOADER
for (x, y) in data_loader
    println(size.((transform256(x, 1), transform256(y, 3),)))
end

u = Unet(1, 3)
u = gpu(u)
function loss(x, y)
         op = clamp.(u(x), 0.001f0, 1.f0)
         mean(bce(op, y))
end
opt = Momentum()

function mytrain!(model, datafunc)
    function loss(x, y)
         op = clamp.(model(x), 0.001f0, 1.f0)
         mean(bce(op, y))
    end
    for (x, y) in datafunc
        x = gpu(transform256(x, 1))
        y = gpu(transform256(y, 3))
        data = (x, y)
        # rep = gpu((transform256(x, 1), transform256(y, 3)))
        Flux.train!(loss, Flux.params(model), data, Momentum());
        @show loss(x, y)
    end
end

for (x, y) in data_loader
        x = gpu(transform256(x, 1))
        y = gpu(transform256(y, 3))
    @show loss(x, y)
end


function batch_to_array(b, batchsize=batchsize)
    mybatch = [ ]
    for i in range(1, batchsize)
        mybatch = cat(mybatch, b[i]; dims=4)
    end
    return mybatch
end


function mytrain!(model, datafunc)
    for (x, y) in datafunc
        x = gpu(transform256(x, 1))
        y = gpu(transform256(y, 3))
        rep = Iterators.repeated((x, y), 10)
        Flux.train!(loss, Flux.params(model), rep, opt);
        @show loss(x, y)
    end
end


for (x, y) in BatchView(data; batchsize=2)
    println(size(batch_to_array(reshape.(channelview.(imresize.(x, 100, 100)), 100, 100, 1))))
    println(size(batch_to_array(reshape.(channelview.(imresize.(y, 100, 100)), 100, 100, 3))))
end

# sizes, 266x266, 276x276, 310x360, 234x234

# This is useful for preserving the aspect ratio and making sure the
# smallest side is still at least however long for random crops
# function presizeimage(image, sz)
#     ratio = maximum(sz ./ size(image))
#     newsz = round.(Int, size(image) .* ratio)
#     σ = 0.25 .* ( 1 ./ (ratio, ratio))
#     k = KernelFactors.gaussian(σ)
#     return imresize(imfilter(image, k, NA()), newsz)
# end

# SZ = (160, 160)
# image = getobs(X_files, 30000)
# presizeimage(FileIO.load(image), SZ)
