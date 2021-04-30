package com.amin.aseproject;

import android.content.Context;
import android.os.AsyncTask;
import android.os.SystemClock;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;

import com.amin.aseproject.classifier.Classifier;
import com.amin.aseproject.classifier.TFClassifier;
import com.amin.aseproject.util.ImageNetLabel;
import com.amin.aseproject.util.InputUtil;
import com.amin.aseproject.util.Label;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class Benchmark extends AsyncTask<Object, Double, BenchmarkResult> {
    public interface Progress {

        void weMove(double progress);
    }

    public interface OnFinishListener {

        void finish(BenchmarkResult result);
    }

    private Progress progress;

    private OnFinishListener finishListener;

    public void setFinishListener(OnFinishListener finishListener) {
        this.finishListener = finishListener;
    }

    public void setProgress(Progress progress) {
        this.progress = progress;
    }


    @Override
    protected BenchmarkResult doInBackground(Object... objects) {
        BenchmarkResult result = new BenchmarkResult();
        Classifier classifier = (Classifier) objects[0];
        String x_path = (String) objects[1];
        String y_path = (String) objects[2];
        try {
            List<Integer> y = new Gson().fromJson(
                    classifier.getInputUtil().readFromStorage(y_path),
                    new TypeToken<List<Integer>>() {
                    }.getType());
            File[] images = classifier.getInputUtil().listStorageFile(x_path, ".*.png");
            Arrays.sort(images);
            int correct_count = 0;
            long t = SystemClock.uptimeMillis();
            for (int i = 0; i < images.length; i++) {
                File image = images[i];
                Label label = classifier.classify(image).getLabel();
                if (label instanceof ImageNetLabel) {
                    if (((ImageNetLabel) label).getImageNetIndex() == y.get(i)) {
                        result.addCorrectIndex(i);
                        correct_count += 1;
                    }
                }
                publishProgress((double) (i + 1));
            }
            result.setBenchmarkTime(SystemClock.uptimeMillis() - t);
            result.setAccuracy(((double) correct_count) / images.length);
            Log.i("benchmark time", String.valueOf(result.getBenchmarkTime()));
            return result;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    protected void onProgressUpdate(Double... values) {
        if (values.length > 0) {
            progress.weMove(values[0]);
        }
    }

    @Override
    protected void onPostExecute(BenchmarkResult result) {
        super.onPostExecute(result);
        Log.i("accuracy", "accuracy is: " + result.getAccuracy() * 100 + "%");
        finishListener.finish(result);
    }
}
