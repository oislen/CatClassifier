from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.merge import concatenate

def Inception(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    """
    function for creating a projected inception module
    """
	# 1x1 conv
	conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
	# 3x3 conv
	conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
	conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
	# 5x5 conv
	conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
	conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
	# 3x3 max pooling
	pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
	pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out

def GoogleNet(input_shape = (256, 256, 3), name = 'GoogleNet'):
	"""
    function for creating GoogleNet
    
    from keras.utils import plot_model
    model = GoogleNet()
    model.summary()
	plot_model(model, show_shapes=True, to_file='GoogleNet.png')
	"""
    # define model input
	visible = Input(shape = input_shape)
    # add preliminary layer HERE
    # stacked inception layers
	layer_1 = Inception(visible, 64, 96, 128, 16, 32, 32)
	layer_2 = Inception(layer_1, 128, 128, 192, 32, 96, 64)
    # add drop out HERE
    # add output layers HERE
	model = Model(inputs = visible, outputs = layer_2, name = name)

	return model

