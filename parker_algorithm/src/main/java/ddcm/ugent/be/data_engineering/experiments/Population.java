/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ddcm.ugent.be.data_engineering.experiments;

import be.ugent.ledc.core.binding.BindingException;
import be.ugent.ledc.core.binding.DataReadException;
import be.ugent.ledc.core.binding.DataWriteException;
import be.ugent.ledc.core.binding.jdbc.schema.TableSchema;
import be.ugent.ledc.core.dataset.ContractedDataset;
import be.ugent.ledc.core.dataset.DataObject;
import be.ugent.ledc.core.dataset.Dataset;
import be.ugent.ledc.core.dataset.SimpleDataset;
import be.ugent.ledc.sigma.datastructures.contracts.SigmaContractorFactory;
import be.ugent.ledc.sigma.datastructures.fd.PartialKey;
import be.ugent.ledc.sigma.datastructures.rules.SigmaRuleset;
import be.ugent.ledc.sigma.datastructures.rules.SufficientSigmaRuleset;
import be.ugent.ledc.sigma.io.ConstraintIo;
import be.ugent.ledc.sigma.repair.RepairException;
import be.ugent.ledc.sigma.repair.cost.functions.CostFunction;
import be.ugent.ledc.sigma.repair.cost.functions.SimpleIterableCostFunction;
import be.ugent.ledc.sigma.repair.cost.models.DefaultObjectCost;
import be.ugent.ledc.sigma.repair.cost.models.ParkerModel;
import be.ugent.ledc.sigma.sufficientsetgeneration.FCFGenerator;
import ddcm.ugent.be.binding.Binding;
import ddcm.ugent.be.data_engineering.Data_engineering;
import ddcm.ugent.be.data_engineering.Repairing;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author rnasfi
 */
public class Population {
    private static boolean done = true;

    public static String[] ageAttribute = {"elderly", "adults", "adolescents", "children"};

    public static String[] genderAttribute = {"female", "male", "others", "gender"};

    public static String[] healthAttribute = {"healthy_volunteers", "medical_condition"};

    public static String[] allAttributes = {"female", "male",
        "elderly", "adults", "adolescents", "children","healthy_volunteers",};//"healthy_volunteers" "eudract_number", "others"

    public static String[] allAttributes1 = {"female", "male", //"others",
        "elderly", "adults", "adolescents", "children",
        "healthy_volunteers", "medical_condition", "gender", "population_age"};

    public static final String path = Data_engineering.path + "/data/trials_population";

    public static Map<String, String[]> corpus = new HashMap<>();

    public static void filling_corpus() {
        String[] elderly = {">= 65", ">= 75", "(6[5-9]|[7-9][0-9])", "elderly", "senior citizens", "retirees", "older adults", "geriatric population", "aging population"};
        String[] adults = {">= ?18", "> ?17", "(19|[2-6][0-9])", "40 to 60", "18 to 25", "adults", "grown-ups", "mature individuals", "working population", "middle-aged", "responsible individuals", "parents"};
        String[] adolescents = {"< 18", "<= 17", "[^>=]{1,2}\\s?1[0-8]", "adolescents", "teenagers", "youths", "pubescents", "young adults", "high school students"};
        String[] subjects_under_18 = {"< ?18", "<= 17", "[^>=]{1,2}[1[0-8]|[0-9]]", "subjects_under_18", "minors", "juveniles", "underage individuals", "children and teens", "legal minors", "dependents"};
        String[] children = {"< ?18", "<= ?17", "(?:[^>])+([10|[0-9]])", "children", "kid", "kids", "minors", "infants", "toddlers", "youngsters", "offspring"};

        String[] female = {"women", "woman", "girls", "ladies", "females", "female",
            "womenfolk", "sisters"};
        String[] male = {"men", "man", "boys", "gentlemen", "male", "males", "brothers", "sons"};
        String[] others = {"non-binary individuals", "gender non-conforming",
            "diverse individuals", "those outside the binary", "genderqueer individuals", "people of all genders"};

        // Add corpus words to the HashSet
        corpus.put("elderly", elderly);
        corpus.put("adults", adults);
        corpus.put("adolescents", adolescents);
        corpus.put("subjects_under_18", subjects_under_18);
        corpus.put("children", children);
        corpus.put("female", female);
        corpus.put("male", male);
        corpus.put("others", others);
    }

    public static Dataset repair(Dataset data, String[] partK, String train) throws DataReadException, RepairException, IOException, DataWriteException {

        //Dataset data3 = Binding.readData(file, ";");
        SigmaRuleset trialsPopulationRules = ConstraintIo
                .readSigmaRuleSet(new File("rules/trials_population.rules"))
                .project(partK);

        //Build a sufficient set with the FCF generator
        SufficientSigmaRuleset sufficientPopulationRules = SufficientSigmaRuleset
                .create(trialsPopulationRules, new FCFGenerator());


        Dataset parkerRepaired = Repairing.parkerRepair(
                data,
                buildCostModel(partK, sufficientPopulationRules));

        String parkerFile = Population.path + "/trials_population_parker" + train + ".csv";
        System.out.print(parkerFile + "\n");
        
         if(done){
//            Binding.writeDataCSV(parkerFile, parkerRepaired, ",", '"');
            done = false;
        }        
        return parkerRepaired;
    }

    private static ParkerModel buildCostModel(String[] partK, SufficientSigmaRuleset sufficientRules) throws RepairException {
        Map<String, CostFunction<?>> costFunctions = new HashMap<>();

        Map<Integer, Map<Integer, Integer>> costMap = new HashMap<>();

        //Change 0 into...
        costMap.put(0, new HashMap<>());
        costMap.get(0).put(1, 1);

        //Change 1 into...
        costMap.put(1, new HashMap<>());
        costMap.get(1).put(0, 2);

        //Build cost functions.
        for (String a : allAttributes) {
            costFunctions.put(a, new SimpleIterableCostFunction<>(costMap));
        }

        return new ParkerModel(
                getPartialKey(partK),
                costFunctions,
                new DefaultObjectCost(),
                sufficientRules
        );

    }

    public static boolean checkAge(String label, String text) {

        if (corpus.containsKey(label)) {
            for (String v : corpus.get(label)) {

                String w = "\\b(" + v + ")\\s(?:age|years| of years)+\\b";

                Pattern pattern = Pattern.compile(w, Pattern.CASE_INSENSITIVE);
                Matcher matcher = pattern.matcher(text.toLowerCase());

                if (matcher.find()) {
                    System.out.println("label: " + label + " -- keyword: " + w);
                    System.out.println("matcher: " + matcher);
                    return true;
                }

            }
        }
        return false;
    }

    public static boolean isLabelTrue(String text, String label) {

        // Check if any word in the text is in the corpus
        if (corpus.containsKey(label)) {
            for (String v : corpus.get(label)) {
                String w = "\\b" + v + "\\b";

                Pattern pattern = Pattern.compile(w, Pattern.CASE_INSENSITIVE);
                Matcher matcher = pattern.matcher(text.toLowerCase());

                if (matcher.find()) {
                    return true;
                }
            }
        }
        return false;
    }

    public static Dataset validating(Dataset data, String... labels) {

        Dataset data1 = new SimpleDataset();
        for (DataObject o : data) {
            boolean valid = true;
            String included = o.getString("inclusion");
            String excluded = o.getString("exclusion");

            DataObject o1 = new DataObject();

            o1.set("eudract_number", o.getString("eudract_number"));
            for (String l : labels) {
                if (o.getAttributes().contains(l)) {

                    int v1 = (int) o.get(l);

                    // check if the value v1 is included
                    if (included != null & v1 == 1) {
                        valid &= (isLabelTrue(o.getString("inclusion"), l) | !(isLabelTrue(excluded, l)));
                        if (Arrays.toString(ageAttribute).contains(l)) {

                            valid &= checkAge(l, included);
                        }
                    }

                    if (valid) {
                        o1.set(l, o.get(l));
                    } else {
                        System.out.println(o.getString("eudract_number"));
                        System.out.println(l + ": " + o.get(l));
                        if (o.getAttributes().contains(l + "_ctgov")) {
                            System.out.println(" <> ctgov: " + o.get(l + "_ctgov"));
                        }
                        System.out.println("its text: " + included);
                        break;
                    }
                }
            }
            if (valid) {
                data1.addDataObject(o1);
            }
        }
        System.out.println("nb rows " + data1.getSize());
        return data1;
    }

    public static void validate_population_labeling() throws DataReadException, DataWriteException, IOException {

        filling_corpus();

//        Dataset data = Binding.readData(path + "/csv/trials_population_train.csv", ",", '"');
//        Dataset data1 = validating(data, allAttributes);       
        SigmaRuleset rules = ConstraintIo
                .readSigmaRuleSet(new File("rules/trials_population.rules"))
                .project(Population.allAttributes1);
        Dataset rawGoldStandard = Binding.getRawGoldStd("trials_population", ",");
        Dataset unclean = rawGoldStandard.select(o -> !(rules.isSatisfied(o)));
        System.out.println("nb of violating rows in G.Std:" + unclean.getSize());

    }

    public static void merge_gs_description() throws DataReadException, DataWriteException {
        // read rows from the trials population
        Dataset gs = Binding.readData(path + "/csv/golden_standard/gs_v0.csv", ",", '"');
        Dataset merg = new SimpleDataset();

        Dataset desc = new SimpleDataset();
        for (int j = 1000; j < 39000; j = j + 1000) {
            System.out.print(j + "\t");

            String desc_path = path + "/raw/population_exclu_inclu_sion_" + j + ".csv";
            System.out.println(desc_path);
            desc = Binding.readData(desc_path, ",", '"');
            int i = 0;
            for (DataObject d : desc) {
                for (DataObject g : gs) {
                    if (d.getString("eudract_number").equals(g.getString("eudract_number"))) {
                        DataObject o = d;
                        for (String a : allAttributes) {
                            o.set(a, g.get(a));
                        }
                        merg.addDataObject(o);
                        i += 1;
                    }

                }
            }
        }

        System.out.println(" done with merging!! ");
        Binding.writeDataCSV(path + "/csv/population_not_full_data.csv", merg, ",", '"');

        Map<String, List<DataObject>> eudractNumberMap = merg
                .project("eudract_number", "male")
                .stream()
                .collect(Collectors.groupingBy(o -> o.getString("eudract_number")));
        System.out.println("nb eudract " + eudractNumberMap.size());

    }

    public static void insert_into_database() throws DataReadException, DataWriteException, BindingException {

        Dataset dataToInsert = new SimpleDataset();
////insert the exclusion & nclusion description
//        Dataset merg = new SimpleDataset();
//        for (int j = 1000; j < 39000; j = j + 1000) {
//            System.out.print(j + "\t");
//
//            String desc_path = path + "/csv/description/population_exclu_inclu_sion_" + j + ".csv";
//            System.out.println(desc_path);
//            Dataset desc = Binding.readData(desc_path, ",", '"');
//            for (DataObject d : desc) {
//                merg.addDataObject(d);
//            }
//        }
//
//        Map<DataObject, List<DataObject>> eudractNumberMap = merg
//                .getDataObjects()//project("eudract_number", "protocol_country_code")
//                .stream()
//                .collect(Collectors.groupingBy(o -> o.project("eudract_number", "protocol_country_code")));
//
//        for (DataObject o : eudractNumberMap.keySet()) {
//            merg1.addDataObject(eudractNumberMap.get(o).get(0));
//        }
//
//        System.out.println("nb eudract " + merg1.getSize());
//        System.out.println("attributes " + merg1.getDataObjects().get(0).getAttributes());
//        Binding.writeJDBC("eudract", merg1, "remote", "population_description");

//        Dataset gs = Binding.readData(path + "/csv/golden_standard/gs_v0.csv", ",", '"');
//
//        Map<String, List<DataObject>> gsNumberMap = gs
//                .getDataObjects()
//                .stream()
//                .collect(Collectors.groupingBy(o -> o.getString("eudract_number")));
//
//        for (String e : gsNumberMap.keySet()) {
//            dataToInsert.addDataObject(gsNumberMap.get(e).get(0));
//        }

        String tableName = "trial_population_golden_standard";

        dataToInsert = Binding.getRawGoldStd("trials_population", ",").project(allAttributes);

        DataObject o = dataToInsert.getDataObjects().get(0);

        TableSchema table = creatTrialPopulationSchema(tableName, o.getAttributes());

        Binding.writeJDBC("eudract", dataToInsert, "remote", table);

    }

    // Define the table where the data will be inserted
    // The table schema should be the same as defined in the database    
    public static TableSchema creatTrialPopulationSchema(String tableName, Set<String> attributes) {
        TableSchema table = new TableSchema();
        table.setName(tableName);
        for (String a : attributes) {
            if (a.equals("eudract_number")) {
                table.setAttributeProperty(a, "text", null);
            } else {
                table.setAttributeProperty(a, "integer", null);
            }
        }
        return table;
    }

    private static PartialKey getPartialKey(String[] partK) {

        PartialKey partialkey = new PartialKey(//Set<String> key, Set<String> determined
                Stream.of("eudract_number").collect(Collectors.toSet()),
                Arrays.asList(partK).stream().collect(Collectors.toSet())
        );

        System.out.println("Number of repair attributes: "
                + partialkey.involvedAttributes().stream().distinct().count());
        return partialkey;
    }

    public static ContractedDataset fix_attributes(ContractedDataset data) {
        for (String s : allAttributes) {
            // the attributes to be repaires are supposdly integer
            if (data.getDataObjects().get(0).getAttributes().contains(s)) {
                data = data.asInteger(s, SigmaContractorFactory.INTEGER);
            }
        }
        return data;
    }
}
