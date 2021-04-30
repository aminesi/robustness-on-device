package com.amin.aseproject.classifier;

import android.content.Context;

import com.amin.aseproject.util.ChannelOperator;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class CaffeClassifier extends TFClassifier {

    private final float[] mean = {103.939f, 116.779f, 123.68f};
    private final float[] std = {1, 1, 1};


    public CaffeClassifier(Context context, String modelPath) throws IOException {
        super(context, modelPath);
    }

    @Override
    protected List<TensorOperator> getPreprocessOperations() {
        return Arrays.asList(
                new ChannelOperator(ChannelOperator.Channel.BGR),
                new NormalizeOp(mean, std)
        );
    }
}
