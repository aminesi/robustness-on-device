package com.amin.aseproject.util;

import com.amin.aseproject.classifier.TFClassifier;

import java.util.Objects;

public class ImageNetLabel extends Label {
    private int kerasIndex;
    private int imageNetIndex;
    private String id;
    private String text;

    public ImageNetLabel() {
    }

    public ImageNetLabel(int kerasIndex, int imageNetIndex, String id, String text) {
        this.kerasIndex = kerasIndex;
        this.imageNetIndex = imageNetIndex;
        this.id = id;
        this.text = text;
    }

    public int getKerasIndex() {
        return kerasIndex;
    }

    public int getImageNetIndex() {
        return imageNetIndex;
    }

    public String getId() {
        return id;
    }

    public String getText() {
        return text;
    }

    @Override
    public String description() {
        return "ImageNetLabel{" +
                "kerasIndex=" + kerasIndex +
                ", imageNetIndex=" + imageNetIndex +
                ", id='" + id + '\'' +
                ", text='" + text + '\'' +
                '}';
    }

    @Override
    public String toString() {
        return text;
    }

    @Override
    public int compareTo(Label o) {
        if (o instanceof ImageNetLabel) {
            ImageNetLabel other = (ImageNetLabel) o;
            return this.kerasIndex - other.kerasIndex;
        }
        return super.compareTo(o);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        ImageNetLabel that = (ImageNetLabel) o;
        return Objects.equals(id, that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
