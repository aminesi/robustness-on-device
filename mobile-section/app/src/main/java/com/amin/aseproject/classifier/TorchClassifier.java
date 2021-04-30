package com.amin.aseproject.classifier;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.SystemClock;
import android.util.Log;

import com.amin.aseproject.util.ClassificationResult;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.IOException;

public class TorchClassifier extends Classifier {
    private final Module module;

    public TorchClassifier(Context context, String filename) throws IOException {
        super(context);
        this.filename = filename;
        module = Module.load(inputUtil.getAssetPath(filename));
    }

    @Override
    public ClassificationResult classify(File file) {
        long t = SystemClock.uptimeMillis();
        Bitmap bitmap = BitmapFactory.decodeFile(file.getAbsolutePath());
        Tensor tensorIn = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
        );
        Log.i(filename + " -> preprocess time", String.valueOf(SystemClock.uptimeMillis() - t));
        t = SystemClock.uptimeMillis();
        IValue forward = module.forward(IValue.from(tensorIn));
        Tensor tensor;
        if (forward.isTuple()) {
            tensor = forward.toTuple()[0].toTensor();
        } else if (forward.isTensor()) {
            tensor = forward.toTensor();
        } else {
            throw new IllegalStateException("model output was invalid");
        }
        float[] out = tensor.getDataAsFloatArray();
        Log.i(filename + " -> inference time", String.valueOf(SystemClock.uptimeMillis() - t));
        return getTopResult(out);
    }
}
