package com.amin.aseproject.util;

import com.google.gson.Gson;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;

import java.lang.reflect.Field;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class LabelDeserializer implements JsonDeserializer<Label> {
    private Gson gson = new Gson();
    private List<Class<? extends Label>> classes = new ArrayList<>();

    public void addSubType(Class<? extends Label> clazz) {
        classes.add(clazz);
    }

    @Override
    public Label deserialize(JsonElement json, Type typeOfT, JsonDeserializationContext context)
            throws JsonParseException {
        if (json instanceof JsonObject) {
            for (Class<? extends Label> clazz : classes) {
                Set<String> names = new HashSet<>();
                for (Field declaredField : clazz.getDeclaredFields()) {
                    if (!declaredField.isAnnotationPresent(IgnoreField.class)) {
                        names.add(declaredField.getName());
                    }
                }
                if (((JsonObject) json).keySet().equals(names)) {
                    return gson.fromJson(json, clazz);
                }
            }
        }
        throw new JsonParseException("no appropriate subtype found!");
    }
}