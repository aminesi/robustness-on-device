package com.amin.aseproject.util;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.SupportPreconditions;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;

public class ChannelOperator implements TensorOperator {
    public enum Channel {
        RGB, BGR
    }

    private Channel channel;
    private int channelCount = 3;

    public ChannelOperator(Channel channel) {
        this.channel = channel;
    }

    @Override
    public TensorBuffer apply(TensorBuffer input) {
        if (channel == Channel.RGB) {
            return input;
        }
        int[] shape = input.getShape();
        float[] values = input.getFloatArray();

        for (int i = 0; i < values.length; i += channelCount) {
            float r = values[i];
            values[i] = values[i + channelCount - 1];
            values[i + channelCount - 1] = r;
        }

        TensorBuffer output;
        if (input.isDynamic()) {
            output = TensorBufferFloat.createDynamic(DataType.FLOAT32);
        } else {
            output = TensorBufferFloat.createFixedSize(shape, DataType.FLOAT32);
        }

        output.loadArray(values, shape);
        return output;
    }
}
