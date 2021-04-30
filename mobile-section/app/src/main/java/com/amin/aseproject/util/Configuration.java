package com.amin.aseproject.util;

import android.content.Context;

import androidx.annotation.NonNull;

import com.amin.aseproject.classifier.CaffeClassifier;
import com.amin.aseproject.classifier.Classifier;
import com.amin.aseproject.classifier.TFClassifier;
import com.amin.aseproject.classifier.TorchClassifier;

import java.lang.reflect.Constructor;

public class Configuration {
    public enum Attack {
        FGSM, BIM, BOUNDARY;

        @NonNull
        @Override
        public String toString() {
            return this.name().toLowerCase();
        }
    }

    public enum Model {
        MOBILENET_V2, RESNET50, INCEPTION_V3;

        @NonNull
        @Override
        public String toString() {
            return this.name().toLowerCase();
        }
    }

    public enum Backend {
        TF, TORCH;

        @NonNull
        @Override
        public String toString() {
            return this.name().toLowerCase();
        }
    }

    private Backend backend;
    private Model model;
    private Attack attack;
    private boolean quant;

    public Configuration(Backend backend, Model model, Attack attack, boolean quant) {
        this.backend = backend;
        this.model = model;
        this.attack = attack;
        this.quant = quant;
    }


    public Backend getBackend() {
        return backend;
    }

    public Model getModel() {
        return model;
    }

    public Attack getAttack() {
        return attack;
    }

    public boolean isQuant() {
        return quant;
    }

    public String getName() {
        return model + (quant ? "-quant" : "");
    }

    public String getModelPath() {
        return getName() + (backend == Backend.TF ? ".tflite" : ".pt");
    }

    public String getDataPath() {
        return String.format("ase/%s/%s/%s", backend, model, attack);
    }

    public Constructor<? extends Classifier> getClassifier() throws NoSuchMethodException {
        Class<? extends Classifier> clazz;
        switch (backend) {
            case TF:
                if (model == Model.RESNET50) {
                    clazz = CaffeClassifier.class;
                } else {
                    clazz = TFClassifier.class;
                }
                break;
            case TORCH:
                clazz = TorchClassifier.class;
                break;
            default:
                throw new IllegalArgumentException("bad backend!");
        }
        return clazz.getConstructor(Context.class, String.class);
    }

}

