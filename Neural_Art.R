rm(list=ls())
gc()
#Sys.getenv()
setwd("local_path")
library(mxnet)
library(imager)
mx.ctx.default()

get_model = function( input_size , ctx ) {
data = mx.symbol.Variable("data")
conv1_1 = mx.symbol.Convolution(name='conv1_1', data=data , num_filter=64, pad=c(1,1), kernel=c(3,3), stride=c(1,1), no_bias=FALSE, workspace=1024)
relu1_1 = mx.symbol.Activation(name='relu1_1', data=conv1_1 , act_type='relu')
conv1_2 = mx.symbol.Convolution(name='conv1_2', data=relu1_1 , num_filter=64, pad=c(1,1), kernel=c(3,3), stride=c(1,1), no_bias=FALSE, workspace=1024)
relu1_2 = mx.symbol.Activation(name='relu1_2', data=conv1_2 , act_type='relu')
pool1 = mx.symbol.Pooling(name='pool1', data=relu1_2 , pad=c(0,0), kernel=c(2,2), stride=c(2,2), pool_type='avg')
conv2_1 = mx.symbol.Convolution(name='conv2_1', data=pool1 , num_filter=128, pad=c(1,1), kernel=c(3,3), stride=c(1,1), no_bias=FALSE, workspace=1024)
relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1 , act_type='relu')
conv2_2 = mx.symbol.Convolution(name='conv2_2', data=relu2_1 , num_filter=128, pad=c(1,1), kernel=c(3,3), stride=c(1,1), no_bias=FALSE, workspace=1024)
relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2 , act_type='relu')
pool2 = mx.symbol.Pooling(name='pool2', data=relu2_2 , pad=c(0,0), kernel=c(2,2), stride=c(2,2), pool_type='avg')
conv3_1 = mx.symbol.Convolution(name='conv3_1', data=pool2 , num_filter=256, pad=c(1,1), kernel=c(3,3), stride=c(1,1), no_bias=FALSE, workspace=1024)
relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1 , act_type='relu')
conv3_2 = mx.symbol.Convolution(name='conv3_2', data=relu3_1 , num_filter=256, pad=c(1,1), kernel=c(3,3), stride=c(1,1), no_bias=FALSE, workspace=1024)
relu3_2 = mx.symbol.Activation(name='relu3_2', data=conv3_2 , act_type='relu')
conv3_3 = mx.symbol.Convolution(name='conv3_3', data=relu3_2 , num_filter=256, pad=c(1,1), kernel=c(3,3), stride=c(1,1), no_bias=FALSE, workspace=1024)
relu3_3 = mx.symbol.Activation(name='relu3_3', data=conv3_3 , act_type='relu')
conv3_4 = mx.symbol.Convolution(name='conv3_4', data=relu3_3 , num_filter=256, pad=c(1,1), kernel=c(3,3), stride=c(1,1), no_bias=FALSE, workspace=1024)
relu3_4 = mx.symbol.Activation(name='relu3_4', data=conv3_4 , act_type='relu')
pool3 = mx.symbol.Pooling(name='pool3', data=relu3_4 , pad=c(0,0), kernel=c(2,2), stride=c(2,2), pool_type='avg')
conv4_1 = mx.symbol.Convolution(name='conv4_1', data=pool3 , num_filter=512, pad=c(1,1), kernel=c(3,3), stride=c(1,1), no_bias=FALSE, workspace=1024)
relu4_1 = mx.symbol.Activation(name='relu4_1', data=conv4_1 , act_type='relu')
conv4_2 = mx.symbol.Convolution(name='conv4_2', data=relu4_1 , num_filter=512, pad=c(1,1), kernel=c(3,3), stride=c(1,1), no_bias=FALSE, workspace=1024)
relu4_2 = mx.symbol.Activation(name='relu4_2', data=conv4_2 , act_type='relu')
conv4_3 = mx.symbol.Convolution(name='conv4_3', data=relu4_2 , num_filter=512, pad=c(1,1), kernel=c(3,3), stride=c(1,1), no_bias=FALSE, workspace=1024)
relu4_3 = mx.symbol.Activation(name='relu4_3', data=conv4_3 , act_type='relu')
conv4_4 = mx.symbol.Convolution(name='conv4_4', data=relu4_3 , num_filter=512, pad=c(1,1), kernel=c(3,3), stride=c(1,1), no_bias=FALSE, workspace=1024)
relu4_4 = mx.symbol.Activation(name='relu4_4', data=conv4_4 , act_type='relu')
pool4 = mx.symbol.Pooling(name='pool4', data=relu4_4 , pad=c(0,0), kernel=c(2,2), stride=c(2,2), pool_type='avg')
conv5_1 = mx.symbol.Convolution(name='conv5_1', data=pool4 , num_filter=512, pad=c(1,1), kernel=c(3,3), stride=c(1,1), no_bias=FALSE, workspace=1024)
relu5_1 = mx.symbol.Activation(name='relu5_1', data=conv5_1 , act_type='relu')

# style and content layers
style = mx.symbol.Group(c(relu1_1, relu2_1, relu3_1, relu4_1, relu5_1))
content = mx.symbol.Group(c(relu4_2))
out = mx.symbol.Group(c(style, content))
#tmp = mx.symbol.infer.shape( symbol = out, data=c(600, 400, 3, 1) ) # VVVVVVVVVVVVVVVVVVVVVV
tmp = mx.symbol.infer.shape( symbol = out, data=c( input_size[1], input_size[2], 3, 1 ) )
arg_shapes    = tmp$arg.shapes
output_shapes = tmp$out.shapes
aux_shapes    = tmp$aux.shapes
arg_names = out$arguments
pretrained = mx.nd.load("./vgg19.params")
arg.arrays <- sapply( arg_shapes , function(shape) {
    mx.nd.zeros( shape, ctx=ctx )
    }, simplify = FALSE, USE.NAMES = TRUE)
aux.arrays <- sapply( arg_shapes , function(shape) {
    mx.nd.zeros( shape, ctx=ctx )
    }, simplify = FALSE, USE.NAMES = TRUE)

  for( name in arg_names ) { # name = arg_names[2]
      if (name == "data") {
        next
      }
    key = paste( "arg:" , name , sep="" )
    arg.arrays[name] = pretrained[key]
  }

grad.req = "write"
grad.reqs <- lapply(names(arg_shapes), function(nm) {
    if (!mxnet:::mx.util.str.endswith(nm, "label") && !mxnet:::mx.util.str.endswith(nm, "data")) {
      grad.req
    } else {
      "null"
    }
                                                    }
  )

executor = mxnet:::mx.symbol.bind(symbol = out, 
                                  ctx=ctx,
                                  arg.arrays=arg.arrays,
                                  aux.arrays=aux.arrays,
                                  grad.reqs = grad.reqs)
return( executor )
}

PreprocessStyleImage = function( path , shape ) {
  img <- load.image(file = path)
  new_size = c( shape[1] , shape[2] )
  resized_img = resize( img , new_size[1] , new_size[2] )
  sample  <- as.array(resized_img)
  sample[,,,1] = sample[,,,1] - 123.68
  sample[,,,2] = sample[,,,2] - 116.779
  sample[,,,3] = sample[,,,3] - 103.939
  dim(sample) = c(600,400,3,1) 
  return( sample )
}

PreprocessContentImage = function( path , long_edge ) {
  img <- load.image(file = path)
  factor = long_edge / max(dim(img)[1:2])
  new_size = c( dim(img)[1]*factor , dim(img)[2]*factor )
  resized_img = resize( img , new_size[1] , new_size[2] )
  sample  <- as.array(resized_img)
  sample[,,,1] = sample[,,,1] - 123.68
  sample[,,,2] = sample[,,,2] - 116.779
  sample[,,,3] = sample[,,,3] - 103.939
  dim(sample) = c(600,400,3,1) 
  return( sample )
}

PostprocessImage = function( img ) {
  img      = as.array(img)
  dim(img) = c(600,400,1,3)
  img[,,1,1] = img[,,1,1] + 123.68
  img[,,1,2] = img[,,1,2] + 116.779
  img[,,1,3] = img[,,1,3] + 103.939
  return( img )
}

StyleGramExecutor = function(input_shape,ctx) {
  # symbol
  data      = mx.symbol.Variable("conv")
  rs_data   = mx.symbol.Reshape( data=data , target_shape=c( prod(input_shape[c(1:(length(input_shape)-2))]) , input_shape[length(input_shape)-1] ) ) 
  weight    = mx.symbol.Variable("weight")
  rs_weight = mx.symbol.Reshape(data=weight, target_shape=c( prod(input_shape[c(1:(length(input_shape)-2))]) , input_shape[length(input_shape)-1] ) )
  fc        = mx.symbol.FullyConnected(data=rs_data, weight=rs_weight, no_bias=TRUE, num_hidden=input_shape[length(input_shape)-1])
  # executor
  conv = mx.nd.zeros(input_shape, ctx=ctx)
  grad = mx.nd.zeros(input_shape, ctx=ctx)
  args = list( "conv"   = conv ,  
               "weight" = conv
              )
  grad = list("conv"=grad)
  reqs = list("conv"   = "write",
              "weight" = "null"
              )

  executor = mxnet:::mx.symbol.bind(symbol = fc , 
                                    ctx    = mx.gpu(),
                                    arg.arrays=args,
                                    aux.arrays=grad,
                                    grad.reqs =reqs)
  return(list( 
               executor=executor
             ) 
        )
}

arg.array.func = function( executor , ctx ) {
  shape_array = sapply( executor$arg.arrays , function(style) {
    dim(style) 
  } , simplify = FALSE, USE.NAMES = TRUE)
  
  temp <- sapply( shape_array , function(shape) {
    mx.nd.zeros( shape, ctx=ctx )
  }, simplify = FALSE, USE.NAMES = TRUE)
  
  return( temp )
}

aux.array.func = function( executor , ctx ) {
  shape_array = sapply( executor$aux.arrays , function(style) {
    dim(style) 
  } , simplify = FALSE, USE.NAMES = TRUE)
  
  temp <- sapply( shape_array , function(shape) {
    mx.nd.zeros( shape, ctx=ctx )
  }, simplify = FALSE, USE.NAMES = TRUE)
  
  return( temp )
}

dev = mx.gpu()
content_np = PreprocessContentImage(path = "./IMG_4343.jpg", long_edge = 600)
style_np   = PreprocessStyleImage(path = "./starry_night.jpg", shape=dim(content_np))
size       = dim( content_np )
executor = get_model( input_size = size , ctx = dev )

gram_executor <- sapply( executor$outputs[-length(executor$outputs)] , function(style) {
  StyleGramExecutor( dim(style), ctx=dev )
} , simplify = FALSE, USE.NAMES = TRUE)


# get style representation
style_array <- sapply( gram_executor , function(x) {
  mx.nd.zeros( dim(x$executor$outputs[[1]]), ctx=dev )
}, simplify = FALSE, USE.NAMES = TRUE)


#ConvExecutor$data = style_np
arg.arrays = arg.array.func( executor = executor , ctx = dev )
  for( name in names(executor$arg.arrays) ) { # name = "data"
      if (name == "data") {
        arg.arrays[name] = list(mx.nd.array(style_np))
      } else {
        arg.arrays[name] = executor$arg.arrays[name]
      }
  }

mx.exec.update.arg.arrays(executor, arg.arrays )
mx.exec.forward(executor)

  for( i in 1:length(executor$outputs[-length(executor$outputs)]) ) { # i = 1
    arg.arrays = arg.array.func( executor = gram_executor[i][[1]]$executor , ctx = dev )
      for( name in names( gram_executor[i][[1]]$executor$arg.arrays) ) { # name = "data"
        if (name == "conv") {
          arg.arrays[name] = executor$outputs[-length(executor$outputs)][i]
        } else {
          arg.arrays[name] = executor$outputs[-length(executor$outputs)][i]
        }
      }
    mx.exec.update.arg.arrays(gram_executor[i][[1]]$executor, arg.arrays )
    mx.exec.forward(gram_executor[i][[1]]$executor)
    style_array[[i]] = gram_executor[i][[1]]$executor$outputs[[1]]
  }

# get content representation
content_array = mx.nd.zeros( shape = dim( executor$outputs[ length(executor$outputs)][[1]] ) , ctx = dev )
content_grad  = mx.nd.zeros( shape = dim( executor$outputs[ length(executor$outputs)][[1]] ) , ctx = dev )

#ConvExecutor$data = content_np
arg.arrays = arg.array.func( executor = executor , ctx = dev )
  for( name in names(executor$arg.arrays) ) { # name = "data"
    if (name == "data") {
      arg.arrays[name] = list(mx.nd.array(content_np))
    } else {
      arg.arrays[name] = executor$arg.arrays[name]
    }
  }
#executor$outputs[ length(executor$outputs)][[1]]
mx.exec.update.arg.arrays(executor, arg.arrays )
mx.exec.forward(executor)
content_array = executor$outputs[ length(executor$outputs)][[1]]
                
# train
#img = mx.nd.zeros(dim(content_np), ctx=dev)
img = mx.runif( shape = dim(content_np) ,  min = -0.1 , max = 0.1 , ctx = dev  )

lr = FactorScheduler(step = 10,factor_val = 0.9)
optimizer = mxnet:::mx.opt.sgd( learning.rate = 0.1 ,
                                momentum = 0.9 ,
                                wd = 0.005 ,
                                lr_scheduler = lr ,
                                clip_gradient = 10
                              )
optim_state = optimizer$create.state(0,img)
old_img = as.array(img)

  for( e in 1:10 ) { # e = 1
    arg.arrays = arg.array.func( executor = executor , ctx = dev )
      for( name in names(executor$arg.arrays) ) { # name = "data"
        if (name == "data") {
          arg.arrays[name] = list(img)
        } else {
          arg.arrays[name] = executor$arg.arrays[name]
        }
      }
    mx.exec.update.arg.arrays(executor, arg.arrays )
    mx.exec.forward(executor)
 
    # style gradient
      for( i in 1:length(executor$outputs[-length(executor$outputs)]) ) { # i = 1
        arg.arrays = arg.array.func( executor = gram_executor[i][[1]]$executor , ctx = dev )
          for( name in names( gram_executor[i][[1]]$executor$arg.arrays) ) { # name = "data"
              if (name == "conv") {
                arg.arrays[name] = executor$outputs[-length(executor$outputs)][i]
              } else {
                arg.arrays[name] = executor$outputs[-length(executor$outputs)][i]
              }
          }
        mx.exec.update.arg.arrays(gram_executor[i][[1]]$executor, arg.arrays )
        mx.exec.forward(gram_executor[i][[1]]$executor)
        temp = gram_executor[i][[1]]$executor$outputs[[1]] - style_array[[i]]
        gram_executor[i][[1]]$executor$backward( list( temp ) )

        aux.arrays = aux.array.func( gram_executor[i][[1]]$executor , ctx = dev )
        arg.arrays[[1]] = gram_executor[i][[1]]$executor$aux.arrays[[1]] / ( (dim( gram_executor[i][[1]]$executor$arg.arrays["conv"][[1]] )[3]^2)*prod( dim(gram_executor[i][[1]]$executor$arg.arrays["conv"][[1]])[1:2] ) )
        arg.arrays[[1]] = arg.arrays[[1]] * 1.0
        mx.exec.update.aux.arrays(gram_executor[i][[1]]$executor, arg.arrays )
        
      }
    # content gradient
    content_grad = (executor$outputs[ length(executor$outputs) ][[1]] - content_array)*10
    
    # image gradient
    grad_array = sapply( 1:length(gram_executor) , function(x) {
      gram_executor[x][[1]]$data_grad[[1]] 
    }, simplify = FALSE, USE.NAMES = TRUE)  
    grad_array[[length(grad_array)+1]] = content_grad
    executor$backward(grad_array)

    optimizer_result = optimizer$update( 0 , img , ConvExecutor$data_grad[[1]] , optim_state)
    optim_state = optimizer_result$state
    img         = optimizer_result$weight
    new_img     = as.array(img)
    
    #identical(old_img,new_img)
    
    old_img = new_img
  }  
    


im <- load.image(file = "./IMG_4343.jpg")
imager::capture.plot(im)

aa = imager::as.cimg( PostprocessImage(img) )
plot(aa)
################################################# fuck

