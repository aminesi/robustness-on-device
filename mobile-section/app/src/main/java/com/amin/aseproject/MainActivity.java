package com.amin.aseproject;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.PowerManager;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;

import com.amin.aseproject.util.Configuration;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.Semaphore;

@SuppressLint({"SetTextI18n", "DefaultLocale"})
public class MainActivity extends AppCompatActivity {
    private static final int STORAGE_REQUEST = 1;
    private ProgressBar progressBar;
    private TextView progress;
    private static final int INPUT_SIZE = 3000;
    private Semaphore semaphore = new Semaphore(1);
    private TextView backendView;
    private TextView modelView;
    private TextView attackView;
    private TextView accuracyView;
    private TextView successView;
    private TextView timeView;
    private PowerManager.WakeLock wakeLock;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        backendView = findViewById(R.id.backend);
        modelView = findViewById(R.id.model);
        attackView = findViewById(R.id.attack);
        accuracyView = findViewById(R.id.accuracy);
        successView = findViewById(R.id.success);
        timeView = findViewById(R.id.time);
        progressBar = findViewById(R.id.progressBar);
        progress = findViewById(R.id.progress);
        progressBar.setIndeterminate(false);
        progressBar.setProgress(0);
        PowerManager powerManager = (PowerManager) getSystemService(POWER_SERVICE);
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "ase:wl");
        wakeLock.acquire();
        if (checkPermissions()) {
            runBenchmarkThread();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (wakeLock.isHeld()) {
            wakeLock.release();
        }
        wakeLock = null;
    }

    private boolean checkPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                return true;
            } else {
                ActivityCompat.requestPermissions(
                        this,
                        new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE},
                        STORAGE_REQUEST
                );
            }
        }
        return false;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == STORAGE_REQUEST) {
            if (grantResults.length > 0) {
                for (int grantResult : grantResults) {
                    if (grantResult != PackageManager.PERMISSION_GRANTED) {
                        return;
                    }
                }
                runBenchmarkThread();
            }
        }
    }

    private void runBenchmarkThread() {
        new Thread(() -> {
//            for (Configuration.Backend backend : Configuration.Backend.values()) {
//                for (Configuration.Model model : Configuration.Model.values()) {
            for (Configuration.Attack attack : Configuration.Attack.values()) {
                for (boolean quant : new boolean[]{false, true}) {
                    try {
                        semaphore.acquire();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    Configuration configuration = new Configuration(
                            Configuration.Backend.TORCH,
                            Configuration.Model.INCEPTION_V3,
                            attack,
                            quant
                    );
                    runOnUiThread(() -> doBenchmark(configuration));
                }
            }
//                }
//            }
            try {
                semaphore.acquire();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            wakeLock.release();
        }).start();
    }

    private void doBenchmark(Configuration configuration) {
        reset();
        backendView.setText(configuration.getBackend().toString());
        modelView.setText(configuration.getName());
        attackView.setText(configuration.getAttack().toString());
        Benchmark benchmark = new Benchmark();
        benchmark.setProgress(progress -> {
            progressBar.setProgress((int) Math.round(progress / INPUT_SIZE * 100));
            this.progress.setText(String.format("%d/%d - %.4f", (int) progress, INPUT_SIZE, progress / INPUT_SIZE * 100) + " %");
        });
        benchmark.setFinishListener(result -> {
            accuracyView.setText(result.getAccuracy() * 100 + " %");
            successView.setText(result.getSuccessRate() * 100 + " %");
            timeView.setText(result.getBenchmarkTime() / 1000 + " s");
            new Handler().postDelayed(() -> saveResults(configuration, result), 500);
        });
        try {
            benchmark.execute(
                    configuration.getClassifier().newInstance(this, configuration.getModelPath()),
                    configuration.getDataPath(),
                    "ase/mobile/selected_y.json");
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private void reset() {
        accuracyView.setText("----");
        successView.setText("----");
        timeView.setText("----");
        progress.setText("----");
        progressBar.setProgress(0);
    }

    private void saveResults(Configuration configuration, BenchmarkResult result) {
        View root = getWindow().getDecorView().getRootView();
        Bitmap bitmap = Bitmap.createBitmap(root.getWidth(), root.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        root.draw(canvas);
        String path = Environment.getExternalStorageDirectory().getAbsolutePath() + "/ase/results/" + configuration.getAttack() + "-" + configuration.getName();
        try {
            result.save(path + ".json");
            FileOutputStream fileOutputStream = new FileOutputStream(path + ".jpg");
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream);
            fileOutputStream.flush();
            fileOutputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        semaphore.release();
    }
}