/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ddcm.ugent.be.metrics;

import be.ugent.ledc.core.binding.DataReadException;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

/**
 *
 * @author rnasfi
 */
public class WebCrawling {

    public static String crawl(String url, String targetId) throws DataReadException {

        try {
            Document doc = Jsoup.connect(url).get();
            Element section = doc.getElementById(targetId);
//            Elements classes = section.getElementsByClass(""); 

            if (section != null) {
//                System.out.println("Extracted section:");
//                System.out.println(section);
                String elementContent = section.text();
//                try ( FileWriter writer = new FileWriter("/home/rnasfi/Documents/data_repair/csv/allergen/health info.txt")) {
//                    writer.write(elementContent);
//                }
                return elementContent;
            } 
//            else {
//                System.out.println("Section with ID '" + targetId + "' not found.");
//                System.out.println("URL: " + url);
//                
//            }

        } catch (IOException e) {
            System.out.println("Error fetching the webpage: " + e.getMessage());
            
        }
        return "";
    }

    public static String extractBoundedSubstring(String input, String startPattern, String endPattern, String regexPersonalised) {
        // Pattern.quote(String pattern1): allow to ignnore meta charachters in pattern1 
        String regex = "(.*?)";

        if (regexPersonalised != null) {
            regex = regexPersonalised;
        }

        if (startPattern != null) {
            regex = Pattern.quote(startPattern) + regex;
        }

        if (endPattern != null) {
            regex = regex + Pattern.quote(endPattern);
        }

//        System.out.println(regex);
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(input);

        if (matcher.find()) {
            String patternToRemove = "\\bE.\\d\\b";
            // Removing the pattern using regular expression
            String matching = matcher.group(1).replaceAll(patternToRemove, "");
                        
            return matching;
        } else {
            // Return null or an empty string if the pattern is not found
            return null;
        }
    }

}
