import numpy as np
import tensorflow as tf

DEFAULT_PADDING = 'SAME'

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path).item()
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self):
        return self.inputs[-1]

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape):
        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i/group, c_o])
            biases = self.make_var('biases', [c_o])
            if group==1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
            if relu:
                bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
                return tf.nn.relu(bias, name=scope.name)
            return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list(), name=scope.name)

    @layer
    def circular_conv(self, inputs, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding='VALID', group=1):
        if padding!='VALID':
            raise ValueError('Padding must be VALID in circular_conv')
        if k_w%2==0:
            raise ValueError('Width of kernel size must be odd in circular_conv')
        B, H, W, C = inputs.get_shape().as_list()
        pad_size = (k_w-1) / 2
        if pad_size>0:
            new_W = W + 2*pad_size
            slice_size_hz = [B,H,pad_size,C]
            slice_size_vt = [B,pad_size,new_W,C]
            if new_W%2==0:
                updown_split_size = (new_W/2,new_W/2)
            else:
                updown_split_size = (int(new_W/2),int(new_W/2)+1)
            slice_size_vt_left = [B,pad_size,updown_split_size[0],C]
            slice_size_vt_right = [B,pad_size,updown_split_size[1],C]
            # start graph-building part
            with tf.variable_scope(name) as scope:
                # slice left
                left = tf.slice(inputs, [0,0,0,0], slice_size_hz)
                # slice right
                right = tf.slice(inputs, [0,0,W-pad_size,0], slice_size_hz)
                # concat left and right along W axis
                tensor_out = tf.concat([right,inputs,left], 2)
                # slice up with new W
                up = tf.slice(tensor_out, [0,0,0,0], slice_size_vt)
                up = tf.map_fn(lambda u:tf.image.flip_up_down(u), up)
                up_left = tf.slice(up, [0,0,0,0], slice_size_vt_left)
                up_right = tf.slice(up, [0,0,updown_split_size[0],0], slice_size_vt_right)
                up = tf.concat([up_right,up_left], 2)
                # slice down with new W
                down = tf.slice(tensor_out, [0,H-pad_size,0,0], slice_size_vt)
                down = tf.map_fn(lambda d:tf.image.flip_up_down(d), down)
                down_left = tf.slice(down, [0,0,0,0], slice_size_vt_left)
                down_right = tf.slice(down, [0,0,updown_split_size[0],0], slice_size_vt_right)
                down = tf.concat([down_right,down_left], 2)
                # concat up and down along H axis
                tensor_out = tf.concat([down,tensor_out,up], 1)
                # perform convolution
                convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
                kernel = self.make_var('weights', shape=[k_h, k_w, C/group, c_o])
                biases = self.make_var('biases', [c_o])
                if group==1:
                    conv = convolve(tensor_out, kernel)
                else:
                    input_groups = tf.split(3, group, tensor_out)
                    kernel_groups = tf.split(3, group, kernel)
                    output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                    conv = tf.concat(3, output_groups)
                if relu:
                    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
                    tensor_out = tf.nn.relu(bias, name=scope.name)
                else:
                    tensor_out = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list(), name=scope.name)
        else: # 1x1 circular_conv, reduce to normal conv
            padding = 'SAME'
            with tf.variable_scope(name) as scope:
                convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
                kernel = self.make_var('weights', shape=[k_h, k_w, C/group, c_o])
                biases = self.make_var('biases', [c_o])
                if group==1:
                    conv = convolve(inputs, kernel)
                else:
                    input_groups = tf.split(3, group, inputs)
                    kernel_groups = tf.split(3, group, kernel)
                    output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                    conv = tf.concat(3, output_groups)
                if relu:
                    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
                    tensor_out = tf.nn.relu(bias, name=scope.name)
                else:
                    tensor_out = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list(), name=scope.name)
        return tensor_out

    @layer
    def roll_branching(self, inputs, name):
        with tf.variable_scope(name):
            B, H, W, C = inputs.get_shape().as_list()
            t_list = tf.split(inputs, num_or_size_splits=W, axis=2)
            indices = np.arange(W)
            out_branches = []
            for _ in range(W):
                # form a branch
                br = [t_list[idx] for idx in indices]
                br = tf.concat(br, axis=2)
                # append to branch list
                out_branches.append(br)
                # update indices, NOTE: np here is used correctly for tensorflow function
                indices = np.roll(indices, -1)

        return out_branches

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims==4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [int(input_shape[0]), dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        return tf.nn.softmax(input, name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)
