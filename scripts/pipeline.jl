using DrWatson
quickactivate(@__DIR__)

using MLDatasets, MLUtils, DataAugmentation

traindir(args...) = datadir("exp_raw", "train", args...)
labeldir(args...) = datadir("exp_raw", "masks", args...)

X_files = FileDataset(identity, traindir(), "*/*/*/*")
Y_files = FileDataset(identity, labeldir(), "*/*/*/*")
files = (X_files, Y_files,)

using FileIO, Images

# Transforms
function mytransforms(image)
    tfm = Crop((128, 128), DataAugmentation.FromRandom()) |> ImageToTensor()
    return apply(tfm, image) |> itemdata
end

function gettfmimagemaskpair((x, y,))
    return (
        mytransforms(Image(FileIO.load(x))),
        mytransforms(Image(FileIO.load(y))),
    )
end

# transformed data
tfmdata = mapobs(gettfmimagemaskpair, files)

# split into training and validation sets
traind, vald = splitobs(tfmdata, at=0.7)

# create iterators
traind_loader = DataLoader(traind; batchsize=20, parallel=true, collate=true, shuffle=true)
vald_loader = DataLoader(vald; batchsize=20, parallel=true, collate=true, shuffle=true)

# trainiter, valiter = DataLoader(traindata, 128, buffered=false), DataLoader(valdata, 256, buffered=false);

# Create model
using UWmadGI
model = Unet(1, 3)
optimizer = Momentum()

## This does not work in FluxTraining.jl
# function loss(x, y)
#      op = clamp.(model(x), 0.001f0, 1.f0)
#      mean(bce(op, y))
# end
# lossfn = loss

lossfn = Flux.Losses.logitcrossentropy

# Create the learner
using FluxTraining
learner = Learner(model, lossfn; callbacks=[ToGPU(), Metrics(accuracy)], optimizer)

FluxTraining.fit!(learner, 1, (traind_loader, vald_loader))

using BenchmarkTools

@btime mytrain!(u, data_loader_serial)
@btime mytrain!(u, data_loader_serial)

@btime mytrain!(u, data_loader_parallel)
@btime mytrain!(u, data_loader_parallel)
