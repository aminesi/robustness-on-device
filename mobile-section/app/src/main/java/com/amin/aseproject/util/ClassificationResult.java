package com.amin.aseproject.util;

import androidx.annotation.NonNull;

public class ClassificationResult {
    private Label label;
    private float probability;

    public Label getLabel() {
        return label;
    }

    public float getProbability() {
        return probability;
    }

    public ClassificationResult(Label label, float probability) {
        this.label = label;
        this.probability = probability;
    }

    @NonNull
    @Override
    public String toString() {
        return label.toString() + "\t" + probability;
    }
}
