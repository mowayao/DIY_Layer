from keras.models import Sequential, Graph
from keras.layers.core import Activation, Layer
from keras.layers.convolutional import Convolution2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense

class Identity(Layer):
    def get_output(self, train):
        return self.get_input(train)


def build_residual_block(params):
    ```
    n_filter: the number of filters
    prefix: unique prefix
    input_shape: input shape
    kernel_h: kernel height
    kernel_w: kernel width

    ```

    n_filters = params.get('n_filter',96)
    if 'input_shape' not in params:
        raise ValueError("input shape is none!")
    if 'prefix' not in params:
        raise ValueError("prefix is empty")

    prefix = params['prefix']
    input_shape = params['input_shape']
    kernel_h = params.get('kernel_h',3)
    kernel_w = params.get('kernel_w',3)
    n_skip = params.get('n_skip',2)

    block = Graph()
    input_name = '%s_x'%prefix
    block.add_input(input_name, input_shape=input_shape)

    prev_output = input_name

    identity_name = "%s_i"%prefix
    block.add_node(Identity(input_shape=input_shape),name=identity_name,input=prev_output)


    for _ in xrange(n_skip):
        conv_name = "%sconv_%d"%(prefix,_+1)
        batch_norm_name = "%sbatch_norm_%d"%(prefix,_+1)
        activation_name = "%sactivation_%d"%(prefix,_+1)
        block.add_node(Convolution2D(n_filters,kernel_w,kernel_h,border_mode='same'),name=conv_name,input=prev_output)
        block.add_node(BatchNormalization(axis=1),name=batch_norm_name,input=conv_name)
        if _ < n_skip-1:
            block.add_node(Activation('relu'),name=activation_name,input=batch_norm_name)
            prev_output = activation_name
        else:
            prev_output = batch_norm_name
    concat_relu_name = "%sconcat_relu"%prefix
    block.add_node(Activation('relu'),name=concat_relu_name,inputs=[identity_name,prev_output],merge_mode='sum')
    prev_output = concat_relu_name
    output_name = "%s_output"%prefix
    block.add_output(name=output_name,input=prev_output)
    return block
