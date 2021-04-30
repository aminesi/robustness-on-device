package com.amin.aseproject.classifier;

import android.content.Context;

import com.amin.aseproject.util.ClassificationResult;
import com.amin.aseproject.util.ImageNetLabel;
import com.amin.aseproject.util.InputUtil;
import com.amin.aseproject.util.Label;
import com.amin.aseproject.util.LabelDeserializer;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class Classifier {

    protected String filename = "";
    protected List<Label> labels = new ArrayList<>();
    protected InputUtil inputUtil;

    public Classifier(Context context) throws IOException {
        inputUtil = new InputUtil(context);
        getLabels();
    }

    public InputUtil getInputUtil() {
        return inputUtil;
    }

    private void getLabels() throws IOException {
        LabelDeserializer deserializer = new LabelDeserializer();
        deserializer.addSubType(ImageNetLabel.class);
        Gson gson = new GsonBuilder()
                .registerTypeAdapter(Label.class, deserializer)
                .create();
        labels = gson.fromJson(inputUtil.readFromAsset("labels.json"),
                new TypeToken<List<Label>>() {
                }.getType());
        Collections.sort(labels);
    }

    public abstract ClassificationResult classify(File file);

    protected ClassificationResult getTopResult(float[] probs) {
        int maxi = -1;
        float max = -1;
        for (int i = 0; i < probs.length; i++) {
            if (probs[i] > max) {
                maxi = i;
                max = probs[i];
            }
        }
        return new ClassificationResult(labels.get(maxi), max);
    }
}
