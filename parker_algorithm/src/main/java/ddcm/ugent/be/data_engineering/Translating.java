/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ddcm.ugent.be.data_engineering;

import be.ugent.ledc.core.binding.DataReadException;
import be.ugent.ledc.core.binding.DataWriteException;
import be.ugent.ledc.core.dataset.DataObject;
import be.ugent.ledc.core.dataset.Dataset;
import java.io.FileReader;

import java.io.IOException;
import java.util.Map;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import com.fasterxml.jackson.databind.ObjectMapper;
import ddcm.ugent.be.binding.Binding;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 *
 * @author rnasfi
 */
public class Translating {

    public static void translate() throws IOException, DataReadException, DataWriteException {
        String path0 = "/home/rnasfi/Documents/data_repair/project_allergens/project_allergens";
        String path1 = "/home/rnasfi/Documents/data_repair/project/result/allergens";
        String path_dataset = path0 + "/csv/allergen_eng.csv"; //allergen_train.csv"; //allergen_eng.csv

        String lang = "all";
        String path_dict = path0 + "/result/dict/all_dict.json";
        if (lang.equals("all")) {
            path_dict = path0 + "/result/dict/all_dict.json";
        }
        if (lang.equals("fr")) {
            path_dict = path0 + "/result/dict/fr_dict.json";
        }
        if (lang.equals("de")) {
            path_dict = path0 + "/result/dict/de_dict.json";
        }
        if (lang.equals("it")) {
            path_dict = path0 + "/result/dict/it_dict.json";
        }
        if (lang.equals("sp")) {
            path_dict = path0 + "/result/dict/sp_dict.json";
        }
        if (lang.equals("bg")) {
            path_dict = path0 + "/result/dict/bg_dict.json";
        }

        try {
            Dataset allergenData = Binding.readData(path_dataset, ",", '"');

            boolean change = false;

            Map<String, Object> mapDeEng = jsonToMap(path_dict);

            int i = 0;

            for (DataObject o : allergenData.getDataObjects()) {
//                if (i == 23) {
//                    System.out.print(o);
//                }
                change = false;

                String ingred = o.getString("ingredients"); //new_lang ingredients

                if (ingred != null) {
                    for (String w : ingred.split("[|_,;\\s*.:`\\'\\(\\)]+")) {
                        String lw = w.toLowerCase().replace("_", "").replace("\"", "").replace("(", "");
//                        if (containsKeyWithRegex(mapDeEng, lw)) {
                        if (mapDeEng.containsKey(lw)) {   //keep only charachters                     
                            String trans = mapDeEng.get(lw).toString().toLowerCase();
                            String w_regex = "\\b" + w + "\\b";
                            ingred = ingred.replaceAll(w_regex, trans);
                            change = true;
                        }
                    }

                    if (change) {
                        o.setString("new_lang", ingred);
                    }
                }
                i += 1;
                if (!change) {
                    System.out.println(ingred);
                    o.setString("new_lang", ingred);
                }
            }

            Binding.writeDataCSV(path_dataset, allergenData, ",", '"');

        } catch (ParseException ex) {
            Logger.getLogger(Translating.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    public static Map<String, Object> jsonToMap(String filePath) throws IOException, ParseException {

        Object o = new JSONParser().parse(new FileReader(filePath));

//        System.out.print(o.toString());
        ObjectMapper objectMapper = new ObjectMapper();

        // Read JSON string into a Map
        Map<String, Object> jsonMap = objectMapper.readValue(o.toString(), Map.class);

        // Print the Map
//        for (Map.Entry<String, Object> entry : jsonMap.entrySet()) {
//            System.out.println(entry.getKey() + ": " + entry.getValue());
//        }
        return jsonMap;

    }

    private static boolean containsKeyWithRegex(Map<String, Object> map, String regex) {
        Pattern pattern = Pattern.compile(regex);

        for (String key : map.keySet()) {
            if (key.equals(regex)) {
                return true;
            }

            Matcher matcher = pattern.matcher(key);
            if (matcher.find()) {
                return true; // Found a key that matches the regular expression
            }
        }

        return false; // No key matched the regular expression
    }

    public static void detectTraces() {
        String label = "nuts";

        String regex1 = "(\\b[^(trace|traces)]\\b)*.*\\b(allergen)+\\b.*" +
                label + "\\b.*";
        Pattern pattern1 = Pattern.compile(regex1);

        String regex2 = "(\\ballergen\\b)*.*" + label + 
                ".*(trace)*";
        Pattern pattern2 = Pattern.compile(regex2);

        String regex3 = "(traces?)*(?:(?!\\ballergen).)*" + label + 
                ".*(allergen)*";
        Pattern pattern3 = Pattern.compile(regex3);

        String txt = "nuts porridge 8% lounges sty trace oil short con t "
                + "umeuue pipol the dust; garlic the dust from biologically "
                + "farm from biologically farming : nuts";
        
        Matcher matcher1 = pattern1.matcher(txt);
        Matcher matcher2 = pattern2.matcher(txt);
        Matcher matcher3 = pattern3.matcher(txt);
        

        
        if(!matcher1.find() & matcher2.find() ){//& matcher2.find()
            System.out.println("found");
        }
        else{
            System.out.println("Nee!!!");
        }

    }
}
