/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ddcm.ugent.be.data_engineering.experiments;

import be.ugent.ledc.core.binding.BindingException;
import be.ugent.ledc.core.binding.DataReadException;
import be.ugent.ledc.core.binding.DataWriteException;
import be.ugent.ledc.core.binding.csv.CSVBinder;
import be.ugent.ledc.core.binding.csv.CSVDataReader;
import be.ugent.ledc.core.binding.csv.CSVProperties;
import be.ugent.ledc.core.binding.jdbc.JDBCBinder;
import be.ugent.ledc.core.binding.jdbc.schema.TableSchema;
import be.ugent.ledc.core.dataset.ContractedDataset;
import be.ugent.ledc.core.dataset.DataObject;
import be.ugent.ledc.core.dataset.Dataset;
import be.ugent.ledc.core.dataset.SimpleDataset;
import be.ugent.ledc.sigma.datastructures.contracts.SigmaContractorFactory;
import be.ugent.ledc.sigma.datastructures.fd.PartialKey;
import be.ugent.ledc.sigma.datastructures.rules.SigmaRuleset;
import be.ugent.ledc.sigma.datastructures.rules.SufficientSigmaRuleset;
import be.ugent.ledc.sigma.repair.ParkerRepair;
import be.ugent.ledc.sigma.repair.RepairException;
import be.ugent.ledc.sigma.repair.cost.functions.CostFunction;
import be.ugent.ledc.sigma.repair.cost.functions.SimpleIterableCostFunction;
import be.ugent.ledc.sigma.repair.cost.models.DefaultObjectCost;
import be.ugent.ledc.sigma.repair.cost.models.ParkerModel;
import be.ugent.ledc.sigma.repair.selection.RandomRepairSelection;
import be.ugent.ledc.sigma.sufficientsetgeneration.FCFGenerator;
import ddcm.ugent.be.binding.Binding;
import ddcm.ugent.be.data_engineering.Data_engineering;
import ddcm.ugent.be.data_engineering.Repairing;

import ddcm.ugent.be.metrics.WebCrawling;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 *
 * @author rnasfi
 */
public class Allergen {
    private static boolean done = true;

    public static String path = Data_engineering.path + "/data/allergens";
    public static String dataName = "allergens";
    
        public static Set<String> allergenAttributes = Stream.of(
            "nuts", "peanut", "milk", "soy", "gluten", "eggs"
//                , "sesame", "celery"
//                , "almondnuts",
//            "brazil_nuts", "macadamia_nuts", "hazelnut"//,
//            "pistachio", "walnut", "cashew",
//            "crustaceans", "fish",
//            "lupin", "molluscs", "mustard",
//            "sulfite"
    ).collect(Collectors.toSet());

        //for contractd dataset
    public static final String[] attrib_int = {"almondnuts", "brazil_nuts",
        "buckwheat", "cashew", "celery", "crustaceans", "eggs", "fish",
        "gluten", "hazelnut", "lupin", "macadamia_nuts", "milk",
        "molluscs", "mustard", "nuts", "oat", "peals", "peanut",
        "pistachio", "rice", "sesame", "soy", "sulfite", "walnut"};

    public static Set<String> toBeRepairedAttributes(Dataset data) {
        Set<String> toBeRepairedAttribute = allergenAttributes;
        toBeRepairedAttribute.retainAll(data.getDataObjects().get(0).getAttributes());
        return toBeRepairedAttribute;
    }


    public static Dataset getAllergensClean() throws DataReadException, IOException, DataWriteException {

        String file = path + "csv/allergen_eng.csv";
        String rulesFile = path + "rules/allergens.rules";

        Dataset uncleanData = Binding.readContractedData(file, "allergens", ",", '"');

        //uncleanData = Binding.imputeMissingValues(uncleanData, attrib_int);
        Dataset cleanData = Binding.getCleanData(uncleanData, rulesFile);
        System.out.print("number of rows satisfying rules:" + cleanData.getSize());

        Binding.writeDataCSV(path + "csv/allergen_parker_v1.1.csv", cleanData, ",", '"');

        return cleanData;
    }

    // crawl FoodFacts websites and get ingredients descriptions
    public static void getDescriptions() throws DataReadException, DataWriteException {

        //1. Get dataset from postgres (or csv if exists)
        String query = "SELECT distinct code FROM openfoodfacts_full.allergen_dataset";
        String schema = "";
        String local = "remote";

        Dataset dataset = Binding.readData(query, schema, local);

        String url = "https://be.openfoodfacts.org/product/"; // URL of the website to crawl
        String targetId = "panel_ingredients_content";

        Dataset descriptionDataset = new SimpleDataset();

        //2. Loop over each urls
        for (DataObject d : dataset) {
            String code = d.getString("code");
            System.out.println("URL: " + url + code);
            String ingredients = WebCrawling.crawl(url + code, targetId).replace(";", "");

            DataObject newRow = new DataObject().set("code", code);
            newRow.set("ingredients", ingredients);

            descriptionDataset.addDataObject(newRow);
        }

        Binding.writeDataCSV(path + "ingredients_description.csv", descriptionDataset, ";");
    }

    public static ContractedDataset fix_attributes(ContractedDataset data) {
        // new_lang is an extra column added as an english translation of the ingredients
        String ingredients = "new_lang";  // new_lang, ingredients        
        for (String s : attrib_int) {
            // the attributes to be repaires are supposdly integer
            if (data.getDataObjects().get(0).getAttributes().contains(s)) { 
//data.getDataObjects().get(0).getAttributes().contains(ingredients) | 
                data = data.asInteger(s, SigmaContractorFactory.INTEGER);
            }

        }
        return data;
    }
    
    public static void insert_into_database() throws DataReadException, DataWriteException, BindingException{
        Dataset dataToInsert = new SimpleDataset();
        Dataset data = new SimpleDataset();
        
        String tableName = "allergen_dataset";
        String fileName = path + "csv/extras/migipedia.csv";

        data = Binding.readContractedData(fileName, "allergens", ",", '"');
        DataObject o = data.getDataObjects().get(0);//migipedia have all the attributes
        
//        fileName = path + "csv/extras/piccantino-struct.csv";
//        fileName = path + "csv/extras/das-ist-drin.csv";
//        fileName = path + "csv/extras/alleregens_dataset_v0.csv";
        fileName = path + "csv/allergens_dataset.csv";
        dataToInsert = Binding.readContractedData(fileName, "allergens", ",");
        o = dataToInsert.getDataObjects().get(0);
        
//        for(DataObject t : dataToInsert){// alleregens_dataset_v0.csv have already the columns source
//            t.setString("source", "dasistdrin");
//        }

        TableSchema table = createAllergensSchema(tableName, o.getAttributes());

//        Binding.writeJDBC("allergens", dataToInsert, "localhost", table);
        Binding.writeJDBC("openfoodfacts_full", dataToInsert, "remote", table);        
    }
    
        // Define the table where the data will be inserted
    // The table schema should be the same as defined in the database    
    public static TableSchema createAllergensSchema(String tableName, Set<String> attributes) {
        TableSchema table = new TableSchema();
        table.setName(tableName);
        
        for (String a : attributes) {
//            if(!a.equals("traces")) // new tables schema do not include the column traces
          if(Arrays.asList(attrib_int).contains(a))
              table.setAttributeProperty(a, "integer", null);
          else 
              table.setAttributeProperty(a, "text", null);                         
        }
        
        // some datasets are without source column
//        table.setAttributeProperty("source", "text", null);
        return table;
    }
//    public static ContractedDataset dataForParker(ContractedDataset testData, String allergensFolder) throws DataReadException {//funny name
//        // Feed Parker with much possible rows to get more combination of values
//        String allAllergenFile = allergensFolder + "/csv/allergen_updated.csv"; 
//        // Include all rows of allergens dataset
//        Dataset allAllergens = Binding.readContractedData(allAllergenFile, dataName, ",", '"');
//        for(DataObject al : allAllergens){
//            String source = al.getString("source");
//            Long code = al.getLong("code");
//            for(DataObject alt : testData){
//                
//            }
//
//        }
//
//        
//        return testData;
//    }


    //goldStandard is transformed dataset
    // that will include a composite candidate key (code, source)    
    public static Dataset getGoldStd(Dataset rawGoldStd, Set<Long> codeInGoldStandard, Map<Long, List<DataObject>> codeMap) throws DataReadException {

        //gs = gold std dataset
        Dataset goldStandard = new SimpleDataset();

        for (Long code : codeMap.keySet()) {
            if (codeInGoldStandard.contains(code)) {

                Dataset selected = rawGoldStd
                        .select(g -> g.getLong("code").equals(code));

                DataObject gso = rawGoldStd
                        .select(g -> g.getLong("code").equals(code))
                        .getDataObjects().get(0);

                codeMap.get(code)
                        .stream()
                        .map(d -> d.getString("source"))
                        .map(cc -> new DataObject().concat(gso)
                        .setString("source", cc))
                        //add this tuple into the gold std including the source attrb
                        .forEach(g -> goldStandard.addDataObject(g));
            }
        }
        return goldStandard;
    }

    public static PartialKey getPartialKey(String[] allergens) {//wrong 
        // allergenAttributes : preset all the present attributes representing an allergen
        // toBeRepairedAttributes : preset the involved attributes in the repair process
        PartialKey partialkey = new PartialKey(Stream.of("code")
                .collect(Collectors.toSet()),
                Stream.of(allergens).collect(Collectors.toSet())
        );

        System.out.println("Number of FDs: "
                + partialkey.getKey().stream().count());

        System.out.println("Number of repair attributes: "
                + partialkey.involvedAttributes().stream().distinct().count());

        return partialkey;
    }

    private static ParkerModel getParkerCostModel(SufficientSigmaRuleset sufficientRules, String[] allergens) throws RepairException {
        Map<String, CostFunction<?>> costFunctions = new HashMap<>();

        Map<Integer, Map<Integer, Integer>> costMap = new HashMap<>();

        //if one sources say that it contains nuts (=1 /2)
        //and another say it doesn't contain nuts (=0)
        //Change 0 into 1 or is of low risk (=1)
        costMap.put(0, new HashMap<>());
        costMap.get(0).put(1, 1);
        costMap.get(0).put(2, 1);

        //Change 1 into 0 is of high risk (=3)
        // 1 into 2 is of low risk (=1)
        costMap.put(1, new HashMap<>());
        costMap.get(1).put(0, 3);
        costMap.get(1).put(2, 1);

        //Change 2 into...
        costMap.put(2, new HashMap<>());
        costMap.get(2).put(0, 3);
        costMap.get(2).put(1, 3);
        
        //Build cost functions.
        for (String a : allergens) {
            costFunctions.put(a, new SimpleIterableCostFunction<>(costMap));
        }

        return new ParkerModel(
                getPartialKey(allergens),
                costFunctions,
                new DefaultObjectCost(),
                sufficientRules
        );
    }

    public static Dataset repair(Dataset data, String dataName, SigmaRuleset allergensRules, String[] allergens, String train) throws DataReadException, RepairException, IOException, DataWriteException {
        SigmaRuleset rules = allergensRules.project(allergens); //Full key constraints-PFrequently based

        SufficientSigmaRuleset sufficientRules = SufficientSigmaRuleset.create(rules, new FCFGenerator());
        System.out.println("rules:\n" + sufficientRules);

//        System.out.print("\n" + dataName + "\n");
//        Dataset violat = data.select(o -> {
//            return o.get("code").toString().equals("4104420077621");
//        });
//        System.out.println("Is the tuple ");
//        System.out.print(violat.getDataObjects().get(0) + " \n violating? ");
//        System.out.print(rules.isSatisfied(violat.getDataObjects().get(0)) + "\n");

        Dataset clean = data.select(o -> (rules.isSatisfied(o)));
        Dataset unclean = data.select(o -> !(rules.isSatisfied(o)));
        System.out.print("Number of edit rules violating tuples: "
                + unclean.getSize()
                + " out of " + data.getSize()
                + "\n");

        Dataset parkerRepaired = Repairing.parkerRepair(
                data, getParkerCostModel(sufficientRules, allergens), sufficientRules, clean);
        
    
        if(done){            
//            String parkerFile = path + "/allergens_parker"+ train + ".csv";
//            System.out.print(parkerFile + "\n");
//            Binding.writeDataCSV(parkerFile, parkerRepaired, ",", '"');
            done = false;
        }      

        return parkerRepaired;
    }

    public static Dataset repair_ledc(SufficientSigmaRuleset sufficientRules) throws DataReadException, RepairException {

        CSVBinder csvBinder = new CSVBinder(
                new CSVProperties(true, ",", '"'),
                new File(path
                        + "csv/allergen.csv")
        );
        Dataset dataset = new CSVDataReader(csvBinder).readDataWithTypeInference(100);
        String[] attributes = toBeRepairedAttributes(dataset).toArray(new String[0]);
        ParkerRepair repairEngine = new ParkerRepair(
                getParkerCostModel(sufficientRules, attributes),
                new RandomRepairSelection()
        );
        return repairEngine.repair(dataset);
    }

    public static void read_openfoodfacts(String file) throws DataReadException{
        System.out.println("\n Reading new dataset from OpenFoodFacts website");
        Dataset data1 = Binding.readData(file, "\t", '"');
        //connect to the database openfoodfacts in tundra
        JDBCBinder jdbc = Binding.binderJDBC("openfoodfacts_full", "remote");
    }
}
