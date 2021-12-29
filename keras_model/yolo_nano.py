from keras.models import Model
from keras.layers import *
import keras.backend as K
from keras.callbacks.callbacks import Callback
import warnings

def conv1x1(input_layer, output_channels, stride=1, bn=True):
    # 1x1 convolution without padding
    if bn == True:
        layer = Conv2D(output_channels, kernel_size=1, strides=stride, padding='same', bias=False)(input_layer)
        layer = BatchNormalization()(layer)
        layer = ReLU(max_value=6)(layer)
    else:
        layer = Conv2D(output_channels, kernel_size=1, strides=stride, padding='same', bias=False)(input_layer)
    return layer

def conv3x3(input_layer, output_channels, stride=1, bn=True):
    layer = None
    # 3x3 convolution with padding=1
    if bn == True:
        layer = Conv2D(output_channels, kernel_size=3, strides=stride, padding='same', bias=False)(input_layer)
        layer = BatchNormalization()(layer)
        layer = ReLU(max_value=6)(layer)
    else:
        layer = Conv2D(output_channels, kernel_size=3, strides=stride, padding='same', bias=False)(input_layer)
    return layer

def sepconv3x3(input_layer, output_channels, stride=1, expand_ratio=1):
    input_channels = input_layer.shape[3]

    layer = Conv2D(input_channels * expand_ratio, kernel_size=1, strides=1, bias=False)(input_layer)
    layer = BatchNormalization()(layer)
    layer = ReLU(max_value=6)(layer)

    layer = Conv2D(input_channels * expand_ratio, kernel_size=3, strides=stride, padding='same', bias=False)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU(max_value=6)(layer)

    layer = Conv2D(output_channels, kernel_size=1, strides=1, bias=False)(layer)
    layer = BatchNormalization()(layer)

    return layer

def EP(input_layer, output_channels, stride=1):
    input_channels = input_layer.shape[3]
    use_res_connect = stride == 1 and input_channels == output_channels
    layer = sepconv3x3(input_layer, output_channels, stride=stride)

    if use_res_connect:
        layer = Add()([layer, input_layer])
    return layer

def PEP(input_layer, output_channels, x, stride=1):
    input_channels = input_layer.shape[3]
    use_res_connect = stride == 1 and input_channels == output_channels
    layer = conv1x1(input_layer, x)
    layer = sepconv3x3(layer, output_channels, stride=stride)
    if use_res_connect:
        layer = Add()([layer, input_layer])
    return layer

def FCA(input_layer, reduction_ratio):
    input_channels = input_layer.shape[3]
    hidden_channels = input_channels // reduction_ratio

    layer = AveragePooling2D((1, 1))(input_layer)
    layer = Dense(hidden_channels, use_bias=False)(layer)
    layer = ReLU(max_value=6)(layer)
    layer = Dense(input_channels, use_bias=False, activation='sigmoid')(layer)
    layer = Multiply()([layer, input_layer])
    return layer


def YoloNano(num_classes, image_size):
    visible = Input(shape=(*image_size, 3))
    num_anchors = 3
    yolo_channels = (num_classes + 5) * num_anchors

    # image:  416x416x3
    conv1 = conv3x3(visible, 12, stride=1)  # output: 416x416x12
    conv2 = conv3x3(conv1, 24, stride=2)  # output: 208x208x24
    pep1 = PEP(conv2, 24, 7, stride=1)  # output: 208x208x24
    ep1 = EP(pep1, 70, stride=2)  # output: 104x104x70
    pep2 = PEP(ep1, 70, 25, stride=1)  # output: 104x104x70
    pep3 = PEP(pep2, 70, 24, stride=1)  # output: 104x104x70
    ep2 = EP(pep3, 150, stride=2)  # output: 52x52x150
    pep4 = PEP(ep2, 150, 56, stride=1)  # output: 52x52x150
    conv3 = conv1x1(pep4, 150, stride=1)  # output: 52x52x150
    fca1 = FCA(conv3, 8)  # output: 52x52x150
    pep5 = PEP(fca1, 150, 73, stride=1)  # output: 52x52x150
    pep6 = PEP(pep5, 150, 71, stride=1)  # output: 52x52x150

    pep7 = PEP(pep6, 150, 75, stride=1)  # output: 52x52x150
    ep3 = EP(pep7, 325, stride=2)  # output: 26x26x325
    pep8 = PEP(ep3, 325, 132, stride=1)  # output: 26x26x325
    pep9 = PEP(pep8, 325, 124, stride=1)  # output: 26x26x325
    pep10 = PEP(pep9, 325, 141, stride=1)  # output: 26x26x325
    pep11 = PEP(pep10, 325, 140, stride=1)  # output: 26x26x325
    pep12 = PEP(pep11, 325, 137, stride=1)  # output: 26x26x325
    pep13 = PEP(pep12, 325, 135, stride=1)  # output: 26x26x325
    pep14 = PEP(pep13, 325, 133, stride=1)  # output: 26x26x325

    pep15 = PEP(pep14, 325, 140, stride=1)  # output: 26x26x325
    ep4 = EP(pep15, 545, stride=2)  # output: 13x13x545
    pep16 = PEP(ep4, 545, 276, stride=1)  # output: 13x13x545
    conv4 = conv1x1(pep16, 230, stride=1)  # output: 13x13x230
    ep5 = EP(conv4, 489, stride=1)  # output: 13x13x489
    pep17 = PEP(ep5, 469, 213, stride=1)  # output: 13x13x469

    conv5 = conv1x1(pep17, 189, stride=1)  # output: 13x13x189
    conv6 = conv1x1(conv5, 105, stride=1)  # output: 13x13x105

    # upsampling conv6 to 26x26x105
    # concatenating [conv6, pep15] -> pep18 (26x26x430)
    concat_1 = concatenate([UpSampling2D(2)(conv6), pep15])
    pep18 = PEP(concat_1, 325, 113, stride=1)  # output: 26x26x325
    pep19 = PEP(pep18, 207, 99, stride=1)  # output: 26x26x325

    conv7 = conv1x1(pep19, 98, stride=1)  # output: 26x26x98
    conv8 = conv1x1(conv7, 47, stride=1)  # output: 26x26x47

    # upsampling conv8 to 52x52x47
    # concatenating [conv8, pep7] -> pep20 (52x52x197)
    concat_2 = concatenate([UpSampling2D(2)(conv8), pep7])
    pep20 = PEP(concat_2, 122, 58, stride=1)  # output: 52x52x122
    pep21 = PEP(pep20, 87, 52, stride=1)  # output: 52x52x87
    pep22 = PEP(pep21, 93, 47, stride=1)  # output: 52x52x93
    conv9 = conv1x1(pep22, yolo_channels, stride=1, bn=False)  # output: 52x52x yolo_channels

    # conv7 -> ep6
    ep6 = EP(conv7, 183, stride=1)  # output: 26x26x183
    conv10 = conv1x1(ep6, yolo_channels, stride=1, bn=False)  # output: 26x26x yolo_channels

    # conv5 -> ep7
    ep7 = EP(conv5, 462, stride=1)  # output: 13x13x462
    conv11 = conv1x1(ep7, yolo_channels, stride=1, bn=False)  # output: 13x13x yolo_channels

    model = Model(inputs=visible, outputs=[conv9, conv10, conv11], name='YoloNano')

    return model

model = YoloNano(20, (416, 416))
model.summary()

