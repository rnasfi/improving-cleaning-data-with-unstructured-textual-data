/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Project/Maven2/JavaApp/src/main/java/${packagePath}/${mainClassName}.java to edit this template
 */
package ddcm.ugent.be.data_engineering;

import be.ugent.ledc.core.binding.BindingException;
import be.ugent.ledc.core.binding.DataReadException;
import be.ugent.ledc.core.binding.DataWriteException;
import be.ugent.ledc.core.dataset.Dataset;
import be.ugent.ledc.sigma.datastructures.rules.SigmaRuleset;
import be.ugent.ledc.sigma.io.ConstraintIo;
import be.ugent.ledc.sigma.repair.RepairException;
import ddcm.ugent.be.binding.Binding;
import ddcm.ugent.be.data_engineering.experiments.Allergen;
import ddcm.ugent.be.data_engineering.experiments.Eudract;
import ddcm.ugent.be.data_engineering.experiments.Population;
import ddcm.ugent.be.metrics.Evaluation;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author rnasfi
 */
public class Data_engineering {

    public static String path = "/home/rnasfi/Documents/data_repair/improving-data-cleaning-with-unstructured-data";

    public static String[] datasets = {"trials_design", "trials_population", "allergens", "flight", "adult", "cardio", "academic", "who", "tobacco"};

    public static String[] repairing = {"ML classification", "Parker Algorithm", "ML classification and then Parker Algorithm", "Holoclean", "BCLean"};

    public static String[] methodsML = {
        "count-vect-logistic_regression", "tf-idf-logistic_regression",
        "tf-idf-xgboost",
        "count-vect-xgboost",
        "tf-idf-mnb",
        "count-vect-mnb",
        "bert-ann",};
    public static boolean per_attribute = false;
    

    public static void main(String[] args) throws DataReadException, IOException, RepairException, DataWriteException, FileNotFoundException, InterruptedException, BindingException {
        boolean parker = true;
        
        int data_index = 2;
        
        String train = "_test"; //"_test"; // _train
        String cst = "with_constraints"; //"without_constraints"
//        repairwithParker(datasets[data_index], train);
        exposingArtifacts(datasets[data_index], parker, per_attribute, train, cst);
    }

    public static void trialsDesign(String dataName, String[] listMLs, Boolean parker, Boolean per_attribute, String train, String cst) throws DataReadException, IOException, RepairException, DataWriteException {
        System.out.println("Trials design experiment:" + dataName + " attributes");

        String eudractFile = Eudract.path + "/repaired/trials_design";
        String originalDataFile = Eudract.path + "/trials_design" + train + ".csv";
        Dataset clinicalTrialsData = Binding.readContractedData(originalDataFile, dataName, ",", '"');

        List<Double> thresholds = new ArrayList<>(Arrays.asList(0.5, 0.75, 0.95));
        String[] attributesToRepair = getAttributesToRepair(dataName);// {"parallel_group", "randomised", "crossover", "controlled", "arms", "open", "single_blind", "double_blind"}

        if (parker) {
            String ml = "tf-idf-xgboost";
            Evaluation.evaluateRepairs(dataName, clinicalTrialsData, eudractFile, attributesToRepair, cst, ml);
        } else {
            Evaluation.evaluateMLPredictions(clinicalTrialsData, dataName, eudractFile, listMLs, attributesToRepair, per_attribute);
        }
    }

    public static void allergens(String dataName, String[] listMLs, Boolean parker, Boolean per_attribute, String train, String cst) throws DataReadException, IOException, RepairException, DataWriteException {

        String allergenFile = Allergen.path + "/allergens_test.csv";
        Dataset allergenData = Binding.readContractedData(allergenFile, dataName, ",", '"');// to convert code into Long type        

        String[] attributesToRepair = getAttributesToRepair(dataName);

        List<Double> thresholds = new ArrayList<>(Arrays.asList(0.5, 0.75, 0.95));
        String ml = "count-vect-xgboost";

        allergenFile = Allergen.path + "/repaired/allergens";

        if (parker) {
            Evaluation.evaluateRepairs(dataName, allergenData, allergenFile, attributesToRepair, cst, ml);//, thresholds
        } else {
            Evaluation.evaluateMLPredictions(allergenData, dataName, allergenFile, listMLs, attributesToRepair, per_attribute);
        }
    }

    public static void trialsPopulation(String dataName, String[] listMLs, Boolean parker, Boolean per_attribute, String train, String cst) throws DataReadException, IOException, RepairException, DataWriteException {

        String populationFile = Population.path + "/" + dataName + train + ".csv";
        Dataset populationData = Binding.readContractedData(populationFile, dataName, ",", '"');

        String ml = "tf-idf-xgboost";
        String[] attributesToRepair = getAttributesToRepair(dataName);

        List<Double> thresholds = new ArrayList<>(Arrays.asList(0.5, 0.75, 0.95));
        populationFile = Population.path + "/repaired/trials_population";

        if (parker) {
            Evaluation.evaluateRepairs(dataName, populationData, populationFile, attributesToRepair, cst, ml);
        } else {
            Evaluation.evaluateMLPredictions(populationData, dataName, populationFile, listMLs, attributesToRepair, per_attribute);
        }
    }

    static private void exposingArtifacts(String dataName, Boolean parker, Boolean per_attribute, String train, String cst) throws DataReadException, RepairException, DataWriteException, FileNotFoundException, IOException, InterruptedException, BindingException {
        String[] listMLs = methodsML;

        switch (dataName) {
            case "trials_population":
                System.out.println("Trials population experiment");

                trialsPopulation(dataName, listMLs, parker, per_attribute, train, cst);
                break;

            case "trials_design":
                //for clinical trials dataset, define the design attributes               
                trialsDesign(dataName, listMLs, parker, per_attribute, train, cst);
                break;

            case "allergens":
                System.out.println("Allergens experiment");
                allergens(dataName, listMLs, parker, per_attribute, train, cst);
                break;

            default:
                break;
        }

    }

    public static void execute(int i, Boolean parker, Boolean per_attribute, String train, String cst) throws DataReadException, IOException, RepairException, DataWriteException, FileNotFoundException, InterruptedException, BindingException {
        //        Translating.detectTraces();
        exposingArtifacts(datasets[i], parker, per_attribute, train, cst);
    }

    public static void repairwithParker(String dataName, String train) throws DataWriteException, DataReadException, IOException, RepairException {
        String repairFile = getDataFolder(dataName) + "/" + dataName + train + ".csv";
        Dataset data = getDataset(dataName, repairFile);
        
        if (dataName.equals("allergens"))data = getDataset(dataName);
        
        Dataset repairedParker = getParkerRepairData(dataName, data, getAttributesToRepair(dataName), train);

//        Binding.writeDataCSV(repairFile, repairedParker, ",", '"');
    }

    //get a repaired dataset by using the Parker Engine
    public static Dataset getParkerRepairData(String dataName, Dataset data, String[] attributesToRepair, String train) throws IOException, DataReadException, RepairException, DataWriteException {
        switch (dataName) {
            case "trials_design":
                SigmaRuleset eudractRules = ConstraintIo.readSigmaRuleSet(new File("rules/trials_design.rules"));
                return Eudract.repair(data, dataName, eudractRules, attributesToRepair, train);

            case "allergens":
                SigmaRuleset allergensRules = ConstraintIo.readSigmaRuleSet(new File("rules/allergens.rules"));
                return Allergen.repair(data, dataName, allergensRules, attributesToRepair, train);

            case "trials_population":
                return Population.repair(data, attributesToRepair, train);
        }
        return null;
    }

    public static String[] getAttributesToRepair(String dataName) {
        switch (dataName) {
            case "trials_design":
                System.out.println("partk" + Arrays.toString(Eudract.designAttribute));
                return Eudract.designAttribute;

            case "allergens":
                return Allergen.allergenAttributes.toArray(new String[0]);

            case "trials_population":
                return Population.allAttributes;
        }
        return null;
    }

    public static Dataset getDataset(String dataName, String fileName) throws DataReadException {
        return Binding.readContractedData(fileName, dataName, ",", '"');
    }

    public static Dataset getDataset(String dataName) throws DataReadException {
        switch (dataName) {
            case "trials_design":
                String eudractFile = getDataFolder(dataName) + "/trials_design.csv";
                return Binding.readContractedData(eudractFile, dataName, ",", '"');

            case "allergens":
                String allergenFile = getDataFolder(dataName) + "/allergens.csv";
                return Binding.readContractedData(allergenFile, dataName, ",", '"');// to convert code into Long type        

            case "trials_population":
                String populationFile = getDataFolder(dataName) + "/" + dataName + ".csv";
                return Binding.readContractedData(populationFile, dataName, ",", '"');
        }
        return null;
    }

    public static String getDataFolder(String dataName) {
        switch (dataName) {
            case "trials_design":
                return Eudract.path;

            case "allergens":
                return Allergen.path;

            case "trials_population":
                return Population.path;
        }
        return null;

    }

}
