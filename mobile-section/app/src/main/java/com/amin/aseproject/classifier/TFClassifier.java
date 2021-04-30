package com.amin.aseproject.classifier;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.SystemClock;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;

import com.amin.aseproject.util.ClassificationResult;
import com.amin.aseproject.util.ImageNetLabel;
import com.amin.aseproject.util.InputUtil;
import com.amin.aseproject.util.Label;
import com.amin.aseproject.util.LabelDeserializer;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.DequantizeOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TFClassifier extends Classifier {
    private final Interpreter model;
    protected int imageSize;
    protected int classCount;
    protected TensorBuffer probBuffer;
    private final TensorProcessor probProcessor;
    private TensorImage tensorImage;


    public TFClassifier(Context context, String modelPath) throws IOException {
        super(context);
        this.filename = modelPath;
        Interpreter.Options tfliteOptions = new Interpreter.Options();
        tfliteOptions.setNumThreads(4);
        model = new Interpreter(FileUtil.loadMappedFile(context, modelPath), tfliteOptions);
        imageSize = model.getInputTensor(0).shape()[1];
        DataType imageDataType = model.getInputTensor(0).dataType();
        int[] probShape = model.getOutputTensor(0).shape();
        classCount = probShape[1];
        DataType probDataType = model.getOutputTensor(0).dataType();
        probBuffer = TensorBuffer.createFixedSize(probShape, probDataType);
        TensorProcessor.Builder builder = new TensorProcessor.Builder();
        for (TensorOperator postprocessOperation : getPostprocessOperations()) {
            builder.add(postprocessOperation);
        }
        probProcessor = builder.build();
        tensorImage = new TensorImage(imageDataType);

    }

    protected List<TensorOperator> getPreprocessOperations() {
        return Collections.singletonList(new NormalizeOp(127.5f, 127.5f));
    }

    protected List<TensorOperator> getPostprocessOperations() {
        return Collections.emptyList();
    }

    @Override
    public ClassificationResult classify(File file) {
        long t = SystemClock.uptimeMillis();
        Bitmap bitmap = BitmapFactory.decodeFile(file.getAbsolutePath());
        tensorImage = preprocess(bitmap);
        Log.i(filename + " -> preprocess time", String.valueOf(SystemClock.uptimeMillis() - t));
        t = SystemClock.uptimeMillis();
        model.run(tensorImage.getBuffer(), probBuffer.getBuffer().rewind());
        Log.i(filename + " -> inference time", String.valueOf(SystemClock.uptimeMillis() - t));
        return getTopResult(probProcessor.process(probBuffer).getFloatArray());

    }

    private TensorImage preprocess(Bitmap bitmap) {
        tensorImage.load(bitmap);
        ImageProcessor.Builder builder = new ImageProcessor.Builder();
        for (TensorOperator preprocessOperation : getPreprocessOperations()) {
            builder.add(preprocessOperation);
        }
        return builder.build().process(tensorImage);
    }
}
