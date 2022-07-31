using DrWatson
quickactivate(@__DIR__)

using MLDatasets, MLUtils, DataAugmentation

traindir(args...) = datadir("exp_raw", "train", args...)
labeldir(args...) = datadir("exp_raw", "masks", args...)

# function get_training()
#     train = []
#     for (root, dirs, files) in walkdir(traindir(); topdown=false)
#         for file in files
#             train = vcat(train, joinpath(root, file)) # path to files
#         end
#     end
#     return train
# end

# function get_labels()
#     labels = []
#     for (root, dirs, files) in walkdir(labeldir(); topdown=false)
#         for file in files
#             train = vcat(labels, joinpath(root, file)) # path to files
#         end
#     end
#     return labels
# end

# get_training
# get_labels

X_files = FileDataset(identity, traindir(), "*/*/*/*")
Y_files = FileDataset(identity, labeldir(), "*/*/*/*")
files = (X_files, Y_files,)

using FileIO, Images
function loadimagemaskpair((x, y,))
    return (
        FileIO.load(x),
        FileIO.load(y),
    )
end

data = mapobs(loadimagemaskpair, files)

### CONSIDER #### FIXME do random crops and make a dataloader
### TODO Train/Val the dataloader, and use FluxTraining to train
### TODO Find mean & std of traindir samples

image = Image(first(data)[2])
tfm = ImageToTensor()
apply(tfm, image)


tfm = Crop((128, 128), DataAugmentation.FromRandom()) |> ImageToTensor()

tfms = [
    Crop((128, 128), DataAugmentation.FromRandom()),
    ImageToTensor,
    DataAugmentation.Normalize,
]

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
traind_loader = DataLoader(traind; batchsize=20, parallel=true, collate=true)
vald_loader = DataLoader(vald; batchsize=20, parallel=true, collate=true)

# trainiter, valiter = DataLoader(traindata, 128, buffered=false), DataLoader(valdata, 256, buffered=false);

# Create model
using UWmadGI
model = Unet(1, 3)
function loss(x, y)
         op = clamp.(u(x), 0.001f0, 1.f0)
         mean(bce(op, y))
end
optimizer = Momentum()
function loss(x, y)
     op = clamp.(model(x), 0.001f0, 1.f0)
     mean(bce(op, y))
end
lossfn = loss

# Create the learner
using FluxTraining
learner = Learner(model, lossfn; callbacks=[ToGPU(), Metrics(accuracy)], optimizer)

FluxTraining.fit!(learner, 10, (traind_loader, vald_loader))

## FOR TESTING pipeline speed
data_loader_serial = DataLoader(obsview(data, 1:100); batchsize=2, parallel=false)
data_loader_parallel = DataLoader(obsview(data, 1:100); batchsize=2, parallel=true)

### ---- THIS ONE ---- ###
function b2a(a)
    return cat(a[1], a[2]; dims=4)
end

function transform256(array, channels)
    return float(b2a(reshape.(channelview.(imresize.(array, 256, 256)), 256, 256, channels)))
end


using UWmadGI
u = Unet(1, 3)
u = gpu(u)
function loss(x, y)
         op = clamp.(u(x), 0.001f0, 1.f0)
         mean(bce(op, y))
end
opt = Momentum()

## FOR TESTING
# for (x, y) in data_loader
#         x = gpu(transform256(x, 1))
#         y = gpu(transform256(y, 3))
#     @show loss(x, y)
# end

## DO NOT WORK WITH DataLoader & CUDA
# function mytrain!(model, datafunc)
#     function loss(x, y)
#          op = clamp.(model(x), 0.001f0, 1.f0)
#          mean(bce(op, y))
#     end
#     for (x, y) in datafunc
#         x = gpu(transform256(x, 1))
#         y = gpu(transform256(y, 3))
#         data = (x, y)
#         # rep = gpu((transform256(x, 1), transform256(y, 3)))
#         Flux.train!(loss, Flux.params(model), data, Momentum());
#         @show loss(x, y)
#     end
# end
#
# function mytrain!(model, datafunc)
#     for (x, y) in datafunc
#         x = gpu(transform256(x, 1))
#         y = gpu(transform256(y, 3))
#         rep = Iterators.repeated((x, y), 10)
#         Flux.train!(loss, Flux.params(model), rep, opt);
#         @show loss(x, y)
#     end
# end


# function mytrain!(model, datafunc)
#     function loss(x, y)
#          op = clamp.(model(x), 0.001f0, 1.f0)
#          mean(bce(op, y))
#     end
#     rep = gpu(datafunc)
#     Flux.train!(loss, Flux.params(model), rep, opt);
#     @show loss(x, y)
# end


## Good example how to use DataLoader with Flux.gpu
## https://github.com/FluxML/model-zoo/blob/master/vision/vgg_cifar10/vgg_cifar10.jl
function mytrain!(m, train_loader; epochs=1)
    function loss(x, y)
         op = clamp.(m(x), 0.001f0, 1.f0)
         mean(bce(op, y))
    end
    opt = Momentum()
    ps = Flux.params(m)

    @info("Training....")
    # Starting to train models
    for epoch in 1:epochs
        @info "Epoch $epoch"

        for (x, y) in train_loader
            x = gpu(transform256(x, 1))
            y = gpu(transform256(y, 3))
            gs = Flux.gradient(() -> loss(x,y), ps)
            Flux.update!(opt, ps, gs)
            @show loss(x, y)
        end
    end
    return m
end

using BenchmarkTools

@btime mytrain!(u, data_loader_serial)
@btime mytrain!(u, data_loader_serial)

@btime mytrain!(u, data_loader_parallel)
@btime mytrain!(u, data_loader_parallel)
