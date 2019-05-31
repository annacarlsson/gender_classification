from keras import Model
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.layers import BatchNormalization, Flatten, Conv2D, Dense, Input, Activation
from keras.regularizers import l2

# TODO: add classes that provide which hyperparameters there are
# Parent class to both architectures
    


class Architecture():
    def __init__(self, name, transfer, filter_factor, l2_factor, freeze=True):
        self.name = name + f'_{filter_factor}_l2_{l2_factor:.2E}'
        if freeze:
            for layer in transfer.layers:
                layer.trainable = False
        x = Conv2D(filter_factor, kernel_size=(3,3), use_bias=False, kernel_regularizer=l2(l2_factor))(transfer.output)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filter_factor, kernel_size=(3,3), use_bias=False, kernel_regularizer=l2(l2_factor))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(2, kernel_size=(3,3), kernel_regularizer=l2(l2_factor))(x)
        x = Activation('softmax')(x)
        x = Flatten()(x)
        self.model = Model(inputs = transfer.input, outputs = x)

class ArchitectureFullyConnected(): # Might want dropout for this one
    def __init__(self, name, transfer, filter_factor, l2_factor, freeze=True):
        self.name = name + f'_{filter_factor}_l2_{l2_factor:.2E}'
        if freeze:
            for layer in transfer.layers:
                layer.trainable = False
        x = Flatten()(transfer.output)
        x = Dense(filter_factor, kernel_regularizer=l2(l2_factor))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dense(filter_factor, kernel_regularizer=l2(l2_factor))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dense(2, kernel_regularizer=l2(l2_factor))(x)
        x = Activation('softmax')(x)

        self.model = Model(inputs = transfer.input, outputs = x)

class MobileNetGenderFConnected(ArchitectureFullyConnected):
    def __init__(self, image_size, alpha, filter_factor, l2_factor):
        name = f'MobileNetFConnected_alpha_{alpha}'
        trans = alpha <= 1
        transfer = MobileNet(input_shape = (*image_size, 3), alpha=alpha, include_top=False, weights='imagenet' if trans else None)
        super().__init__(name, transfer, filter_factor, l2_factor, transfer)
class MobileNetGender(Architecture):
    def __init__(self, image_size, alpha, filter_factor, l2_factor):
        name = f'MobileNet_alpha_{alpha}'
        trans = alpha <= 1
        transfer = MobileNet(input_shape = (*image_size, 3), alpha=alpha, include_top=False, weights='imagenet' if trans else None)
        super().__init__(name, transfer, filter_factor, l2_factor, transfer)

class MobileNetGenderV2(Architecture):
    def __init__(self, image_size, alpha, filter_factor, l2_factor):
        name = f'MobileNetV2_alpha_{alpha}'
        transfer = MobileNetV2(input_shape = (*image_size, 3), alpha=alpha, include_top=False, weights='imagenet')
        super().__init__(name, transfer, filter_factor, l2_factor, transfer)

class InceptionGenderV3(ArchitectureFullyConnected): 
    # Default input size is 299x299. We might need to readjust the layers after this if we want to use the same input
    def __init__(self, image_size, filter_factor, l2_factor):
        name = f'InceptionV3'
        transfer = InceptionV3(input_shape = (*image_size, 3), include_top=False,)
        super().__init__(name, transfer, filter_factor, l2_factor)


class VGGGender(ArchitectureFullyConnected):
    def __init__(self, image_size, filter_factor, l2_factor):
        name = f'VGG16'
        transfer = VGG16(input_shape = (*image_size, 3), include_top=False, weights='imagenet')
        super().__init__(name, transfer, filter_factor, l2_factor, transfer)