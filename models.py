'''Original code from below link but modified.
https://github.com/MoleImg/Attention_UNet/blob/master/AttResUNet.py 
'''
import os
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Activation, Add

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def f1_score(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    actual_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


# MORE ACCURATE FOR MULTICLASS THAN BINARY SO DO NOT USE IT
@tf.autograph.experimental.do_not_convert
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = tf.constant([0.05, 0.25])  # Assuming binary classification (background, ship)

    # Calculate the focal weight based on the true labels
    focal_weight = tf.where(tf.equal(y_true, 1), alpha[1], alpha[0])

    # Calculate the focal loss
    pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    focal_loss = -focal_weight * (1 - pt) ** gamma * tf.math.log(pt + 1e-8)

    return tf.reduce_mean(focal_loss)




##############################################################


def conv_block(x, filter_size, size, dropout):
    
    conv = layers.Conv2D(size, (filter_size, filter_size), kernel_initializer='he_normal', padding="same")(x)
    conv = layers.Activation("relu")(conv)
    
        
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), kernel_initializer='he_normal', padding="same")(conv)
    conv = layers.Activation("relu")(conv)

    return conv


def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 64,64,6), lambda will return a tensor of shape 
    #(None, 64,64,12), if specified axis=3 and rep=2.

     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)

def grouped_conv_block(x, filter_size, size, groups, dropout, strides=(1, 1), padding='same', activation='relu'):
    group_list = []
    for i in range(groups):
        conv = layers.Conv2D(size, (filter_size, filter_size), strides=strides, padding=padding)(x)
        if dropout > 0.0:
            conv = layers.Dropout(dropout)(conv)
        group_list.append(conv)
    group_merge = layers.Add()(group_list)
    activation = layers.Activation(activation)(group_merge)
    return activation

def resnext_block(x, filter_size, size, groups, dropout, strides=(1, 1), padding='same', activation='relu'):
    conv1 = layers.Conv2D(size, (1, 1))(x)
    conv2 = grouped_conv_block(conv1, filter_size, size, groups, dropout, strides, padding, activation)
    conv3 = grouped_conv_block(conv2, filter_size, size, groups, dropout, padding=padding, activation='linear')
    shortcut = layers.Conv2D(size, (1, 1))(x)
    added = layers.Add()([conv3, shortcut])
    activation = layers.Activation(activation)(added)
    return activation


def res_conv_block(x, filter_size, size, dropout):

    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    conv = layers.Activation('relu')(conv)
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    #conv = layers.Activation('relu')(conv)    #Activation before addition with shortcut
    
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)

    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation('relu')(res_path)    #Activation after addition with shortcut (Original residual block)
    return res_path

def gating_signal(input, out_size):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    x = layers.Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    
    return result

def multihead_attention_block(x, gating, inter_shape, num_heads):
    # Create multiple attention heads
    head_outputs = []
    for _ in range(num_heads):
        attention_output = attention_block(x, gating, inter_shape)
        head_outputs.append(attention_output)
    
    # Combine the outputs of attention heads (you can use concatenation, summation, etc.)
    combined_output = layers.concatenate(head_outputs, axis=-1)
    
    return combined_output


def UNet(input_shape, NUM_CLASSES=1, dropout_rate=0.2):
    '''
    UNet
    
    '''
    # network structure
    FILTER_NUM = 16 # number of filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    

    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate)

   
    # Upsampling layers 
    up_16 = layers.Conv2DTranspose(8*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, conv_16])
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate)

    up_32 = layers.Conv2DTranspose(4*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, conv_32])
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate)

    up_64 = layers.Conv2DTranspose(2*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64])
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate)

    up_128 = layers.Conv2DTranspose(FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate)

    # 1*1 convolutional layers  
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel


    # Model 
    model = models.Model(inputs, conv_final, name="UNet")
    model.summary()
    return model

def Attention_UNet(input_shape, NUM_CLASSES=1, dropout_rate=0.2):
    '''
    Attention UNet
    
    '''
    # network structure
    FILTER_NUM = 16 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = layers.Conv2DTranspose(8*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16])
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate)
    
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = layers.Conv2DTranspose(4*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32])
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate)

    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = layers.Conv2DTranspose(2*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64])
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate)
    
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.Conv2DTranspose(FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate)

    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs, conv_final, name="Attention_UNet")
    model.summary()

    return model
    
def ResUNet(input_shape, NUM_CLASSES=1, dropout_rate=0.2):
    '''
    ResUNet 
    
    '''
    # network structure
    FILTER_NUM = 16 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    # input data
    # dimension of the image depth
    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_128 = res_conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = res_conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = res_conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = res_conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = res_conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate)

    # Upsampling layers
    up_16 = layers.Conv2DTranspose(8*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, conv_16])
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate)

    up_32 = layers.Conv2DTranspose(4*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, conv_32])
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate)

    up_64 = layers.Conv2DTranspose(2*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64])
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate)

    up_128 = layers.Conv2DTranspose(FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate)

    # 1*1 convolutional layers  
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs, conv_final, name="ResUNet")
    model.summary()

    return model

def Attention_ResUNet(input_shape, NUM_CLASSES=1, dropout_rate=0.2):
    '''
    Attention_ResUNet 
    
    '''
    # network structure
    FILTER_NUM = 16 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    # input data
    # dimension of the image depth
    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_128 = res_conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = res_conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = res_conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = res_conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = res_conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = layers.Conv2DTranspose(8*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16])
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate)
    
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = layers.Conv2DTranspose(4*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32])
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate)

    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = layers.Conv2DTranspose(2*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64])
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate)
    
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.Conv2DTranspose(FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate)


    # 1*1 convolutional layers
    
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs, conv_final, name="AttentionResUNet")
    model.summary()

    return model


def Multihead_Attention_UNet(input_shape, NUM_CLASSES=1, 
                             dropout_rate=0.2, num_attention_heads=4):
    
    '''
    Multi-head Attention UNet
    
    '''
    
    FILTER_NUM = 16
    FILTER_SIZE = 3
    UP_SAMP_SIZE = 2
    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate)
    
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM)
    att_16 = multihead_attention_block(conv_16, gating_16, 8*FILTER_NUM, num_attention_heads)
    up_16 = layers.Conv2DTranspose(8*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16])
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate)
    
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM)
    att_32 = multihead_attention_block(conv_32, gating_32, 4*FILTER_NUM, num_attention_heads)
    up_32 = layers.Conv2DTranspose(4*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32])
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate)
    
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM)
    att_64 = multihead_attention_block(conv_64, gating_64, 2*FILTER_NUM, num_attention_heads)
    up_64 = layers.Conv2DTranspose(2*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64])
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate)
    
    gating_128 = gating_signal(up_conv_64, FILTER_NUM)
    att_128 = multihead_attention_block(conv_128, gating_128, FILTER_NUM, num_attention_heads)
    up_128 = layers.Conv2DTranspose(FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate)
    
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.Activation('sigmoid')(conv_final)
    
    model = models.Model(inputs, conv_final, name="Multihead_Attention_UNet")
    model.summary()
    
    return model

def Multi_Attention_ResUNet(input_shape, NUM_CLASSES=1, 
                             dropout_rate=0.2, num_attention_heads=4):
    
    '''
    Multi-head Attention ResUNet
    
    '''
    
    FILTER_NUM = 16
    FILTER_SIZE = 3
    UP_SAMP_SIZE = 2
    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    
    conv_128 = res_conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    
    # DownRes 2
    conv_64 = res_conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    
    # DownRes 3
    conv_32 = res_conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    
    # DownRes 4
    conv_16 = res_conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    
    # DownRes 5, convolution only
    conv_8 = res_conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate)

    
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM)
    att_16 = multihead_attention_block(conv_16, gating_16, 8*FILTER_NUM, num_attention_heads)
    up_16 = layers.Conv2DTranspose(8*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16])
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate)
    
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM)
    att_32 = multihead_attention_block(conv_32, gating_32, 4*FILTER_NUM, num_attention_heads)
    up_32 = layers.Conv2DTranspose(4*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32])
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate)
    
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM)
    att_64 = multihead_attention_block(conv_64, gating_64, 2*FILTER_NUM, num_attention_heads)
    up_64 = layers.Conv2DTranspose(2*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64])
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate)
    
    gating_128 = gating_signal(up_conv_64, FILTER_NUM)
    att_128 = multihead_attention_block(conv_128, gating_128, FILTER_NUM, num_attention_heads)
    up_128 = layers.Conv2DTranspose(FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate)
    
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.Activation('sigmoid')(conv_final)
    
    model = models.Model(inputs, conv_final, name="Multihead_Attention_UNet")
    model.summary()
    
    return model


def Attention_ResNeXtUNet(input_shape, NUM_CLASSES=1, dropout_rate=0.2):
    '''
    ResNeXt UNet with Attention
    
    '''
    # Network structure
    FILTER_NUM = 16  # Number of basic filters for the first layer
    FILTER_SIZE = 3  # Size of the convolutional filter
    UP_SAMP_SIZE = 2  # Size of upsampling filters
    CARDINALITY = 8  # Number of convolutional groups in ResNeXt blocks

    # Input data
    inputs = layers.Input(input_shape, dtype=tf.float32)

    conv_128 = resnext_block(inputs, FILTER_SIZE, FILTER_NUM, CARDINALITY, dropout_rate)  # Use the ResNeXt block here
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)
    
    conv_64 = resnext_block(pool_64,  FILTER_SIZE, FILTER_NUM*2, CARDINALITY, dropout_rate)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)
    
    conv_32 = resnext_block(pool_32,  FILTER_SIZE, FILTER_NUM*4, CARDINALITY, dropout_rate)
    pool_16 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)
    
    conv_16 = resnext_block(pool_16, FILTER_SIZE, FILTER_NUM*8, CARDINALITY, dropout_rate)
    pool_8 = layers.MaxPooling2D(pool_size=(2, 2))(conv_16)
    conv_8 = resnext_block(pool_8,FILTER_SIZE, FILTER_NUM*16, CARDINALITY, dropout_rate)

    #Upsampling layers
    gating_16 = gating_signal(conv_8, 8 * FILTER_NUM)
    att_16 = attention_block(conv_16, gating_16, 8 * FILTER_NUM)
    up_16 = layers.Conv2DTranspose(8 * FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16])
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8 * FILTER_NUM, dropout_rate)

    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = layers.Conv2DTranspose(4*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32])
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate)
    
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = layers.Conv2DTranspose(2*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64])
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate)
    
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.Conv2DTranspose(FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate)

    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1, 1))(up_conv_128)
    conv_final = layers.Activation('sigmoid')(conv_final)  # Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs, conv_final, name="AttentionResNeXtUNet")
    model.summary()

    return model

def FINAL_UNet(input_shape, NUM_CLASSES=1, dropout_rate=0.2,
               num_attention_heads=4):
    '''
    ResNeXt UNet with Multi head Attention
    
    '''
    # Network structure
    FILTER_NUM = 16  # Number of basic filters for the first layer
    FILTER_SIZE = 3  # Size of the convolutional filter
    UP_SAMP_SIZE = 2  # Size of upsampling filters
    CARDINALITY = 8  # Number of convolutional groups in ResNeXt blocks

    # Input data
    inputs = layers.Input(input_shape, dtype=tf.float32)

    conv_128 = resnext_block(inputs, FILTER_SIZE, FILTER_NUM, CARDINALITY, dropout_rate)  # Use the ResNeXt block here
    pool_64 = layers.MaxPooling2D(pool_size=(2, 2))(conv_128)
    
    conv_64 = resnext_block(pool_64,  FILTER_SIZE, FILTER_NUM*2, CARDINALITY, dropout_rate)
    pool_32 = layers.MaxPooling2D(pool_size=(2, 2))(conv_64)
    
    conv_32 = resnext_block(pool_32,  FILTER_SIZE, FILTER_NUM*4, CARDINALITY, dropout_rate)
    pool_16 = layers.MaxPooling2D(pool_size=(2, 2))(conv_32)
    
    conv_16 = resnext_block(pool_16, FILTER_SIZE, FILTER_NUM*8, CARDINALITY, dropout_rate)
    pool_8 = layers.MaxPooling2D(pool_size=(2, 2))(conv_16)
    conv_8 = resnext_block(pool_8,FILTER_SIZE, FILTER_NUM*16, CARDINALITY, dropout_rate)

    #Upsampling layers
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM)
    att_16 = multihead_attention_block(conv_16, gating_16, 8*FILTER_NUM, num_attention_heads)
    up_16 = layers.Conv2DTranspose(8*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16])
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate)
    
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM)
    att_32 = multihead_attention_block(conv_32, gating_32, 4*FILTER_NUM, num_attention_heads)
    up_32 = layers.Conv2DTranspose(4*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32])
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate)
    
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM)
    att_64 = multihead_attention_block(conv_64, gating_64, 2*FILTER_NUM, num_attention_heads)
    up_64 = layers.Conv2DTranspose(2*FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64])
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate)
    
    gating_128 = gating_signal(up_conv_64, FILTER_NUM)
    att_128 = multihead_attention_block(conv_128, gating_128, FILTER_NUM, num_attention_heads)
    up_128 = layers.Conv2DTranspose(FILTER_NUM, kernel_size=(UP_SAMP_SIZE, UP_SAMP_SIZE), strides=(UP_SAMP_SIZE, UP_SAMP_SIZE), padding='same', data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate)
    
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.Activation('sigmoid')(conv_final)
    
    model = models.Model(inputs, conv_final, name="Final_UNet")
    model.summary()

    return model

