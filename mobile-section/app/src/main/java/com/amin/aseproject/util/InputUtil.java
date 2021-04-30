package com.amin.aseproject.util;

import android.content.Context;
import android.os.Environment;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;

public class InputUtil {
    private final Context context;

    public InputUtil(Context context) {
        this.context = context;
    }

    public String readFromAsset(String name) throws IOException {
        return readStream(context.getAssets().open(name));
    }

    public String getAssetPath(String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    public String readFromStorage(String path) throws IOException {
        String rootPath = Environment.getExternalStorageDirectory().getAbsolutePath();
        path = rootPath + "/" + path;
        return readStream(new FileInputStream(path));
    }

    public File[] listStorageFile(String path, String filter) throws IOException {
        String rootPath = Environment.getExternalStorageDirectory().getAbsolutePath();
        path = rootPath + "/" + path;
        File file = new File(path);
        if (file.isDirectory()){
            return file.listFiles((dir, name) -> name.matches(filter));
        }
        return new File[]{};
    }

    private String readStream(InputStream stream) throws IOException {
        InputStreamReader inputStreamReader = new InputStreamReader(stream);
        char[] buffer = new char[1024];
        int read;
        StringBuilder builder = new StringBuilder();
        while ((read = inputStreamReader.read(buffer)) > 0) {
            builder.append(buffer, 0, read);
        }
        inputStreamReader.close();
        return builder.toString();
    }
}
