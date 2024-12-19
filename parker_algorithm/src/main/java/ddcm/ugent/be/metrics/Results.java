/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ddcm.ugent.be.metrics;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 *
 * @author rnasfi
 */
public class Results {

    public static JSONObject results = new JSONObject();

    public static void dump(String file, JSONObject item) throws FileNotFoundException, IOException {

        File outputFile = new File(file);

        try (
                 BufferedWriter writer = new BufferedWriter(
                        new OutputStreamWriter(
                                new FileOutputStream(outputFile), StandardCharsets.UTF_8))) {
            writer.write(item.toString());
            writer.write(results.toString());
            writer.flush();
        }
    }

    //dump the metrics into json file
    public static void dumpResults(String file) throws FileNotFoundException, IOException {

        File outputFile = new File(file);
        try (
                 BufferedWriter writer = new BufferedWriter(
                        new OutputStreamWriter(
                                new FileOutputStream(outputFile), StandardCharsets.UTF_8))) {
            writer.write(results.toString());
            writer.flush();
        }
    }

//    public static JSONObject newStringObject(JSONObject mainObject, String key, String value) {
//        results.put(key, value);
//        return results;
//    }
//    public static JSONObject newIntegerObject(JSONObject mainObject, String key, int value) {
//        mainObject.put(key, value);
//        return mainObject;
//    }    
    public static void newStringObject(String key, String value) {

        results.put(key, value);

    }

    public static void newIntegerObject(String key, int value) {

        results.put(key, value);

    }

    public static void newArrayObject(String key, JSONArray arrayOfValues) {

        results.put(key, arrayOfValues);

    }

    public static void newItemInArrayObject(String key, Map newObject) {
        //'key' refers to possible 'repairMethods ' 
        // 'repairMethods' is a JSON Object (refered by key) in results here
        //is a Map< String key, ArrayList<Map> otherMethods> 
        //--> every item in otherMethods is a Map<String keyItem, Object valueItem>
        //we add a new repair method in the which refers to 

        if (results.has(key)) {
            JSONArray tempo = results.getJSONArray(key);
            tempo.put(tempo.length(), newObject);
            results.put(key, tempo);
        } else {
            List<Map> tempo = new ArrayList();
            tempo.add(newObject);
            results.put(key, tempo);
        }

    }

}
