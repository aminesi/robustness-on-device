package com.amin.aseproject.util;

public abstract class Label implements Comparable<Label> {
    public abstract String description();

    @Override
    public int compareTo(Label o) {
        return 0;
    }
}
