package com.amin.aseproject;

import com.google.gson.GsonBuilder;

import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class BenchmarkResult {
    private double accuracy;
    private double successRate;
    private List<Integer> correctIndices= new ArrayList<>();
    private long benchmarkTime;

    public double getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(double accuracy) {
        this.accuracy = accuracy;
        successRate = 1-accuracy;
    }

    public List<Integer> getCorrectIndices() {
        return correctIndices;
    }

    public void setCorrectIndices(List<Integer> correctIndices) {
        this.correctIndices = correctIndices;
    }

    public long getBenchmarkTime() {
        return benchmarkTime;
    }

    public void setBenchmarkTime(long benchmarkTime) {
        this.benchmarkTime = benchmarkTime;
    }

    public void addCorrectIndex(int index){
        this.correctIndices.add(index);
    }

    public double getSuccessRate() {
        return successRate;
    }

    public void save(String path) throws IOException {
        String s = new GsonBuilder().setPrettyPrinting().create().toJson(this);
        FileWriter writer = new FileWriter(path);
        writer.write(s);
        writer.flush();
        writer.close();
    }
}
