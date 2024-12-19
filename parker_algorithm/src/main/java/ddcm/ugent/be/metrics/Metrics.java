package ddcm.ugent.be.metrics;

import be.ugent.ledc.core.binding.DataReadException;
import be.ugent.ledc.core.dataset.ContractedDataset;
import be.ugent.ledc.core.dataset.DataObject;
import be.ugent.ledc.core.dataset.Dataset;
import be.ugent.ledc.core.dataset.SimpleDataset;

import be.ugent.ledc.core.util.SetOperations;
import be.ugent.ledc.sigma.datastructures.rules.SigmaRuleset;
import be.ugent.ledc.sigma.io.ConstraintIo;

import ddcm.ugent.be.binding.Binding;
import ddcm.ugent.be.data_engineering.experiments.Allergen;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class Metrics {

    private static int parker_done = 0;

    public static void compute(Dataset originalDataset, Dataset repairedDataset, String dataName, String key, String[] selectedAttributesToRepair, String method) throws DataReadException, IOException {

        RepairMetrics cellStatistics = null;
        switch (dataName) {
            case "trials_design":
                cellStatistics = eudract_metrics(originalDataset, repairedDataset, selectedAttributesToRepair);// It depends on which attributes we are correcting
                break;

            case "allergens":
                cellStatistics = allergens_metrics(originalDataset, repairedDataset, selectedAttributesToRepair);
                break;

            case "trials_population":
                cellStatistics = trials_population_metrics(originalDataset, repairedDataset, selectedAttributesToRepair);
                break;
                
            default:
                break;
        }

        System.out.println("repairs: " + cellStatistics.getPositives());
        System.out.println("Correct repairs: " + cellStatistics.getTruePositives());
        System.out.println("Number of errors: " + cellStatistics.getErrors());

        double p1 = 0;
        double r1 = 0;
        double f1 = 0;
        if (cellStatistics.getF1() > 0) {
            p1 = cellStatistics.getPrecision();
        }
        // What proportion of positive identifications was actually correct?
        System.out.println("Precision: " + p1);
        if (cellStatistics.getRecall() > 0) {
            r1 = cellStatistics.getRecall();
        }
        // What proportion of actual positives was identified correctly?
        System.out.println("Recall: " + r1);// truePositives / errors

        if (cellStatistics.getF1() > 0) {
            f1 = cellStatistics.getF1();
        }
        System.out.println("F1: " + f1);

        // Put it in HashMap to be saved as Json file
        if (!(parker_done != 0 & key.equals("Parker Algorithm"))) {

            Map metricsMap = new HashMap();
            metricsMap.put("key", key);

            // add the nlp method and the classifier  if the repair used machine learnig
            if (key.equals("Parker")) {
                metricsMap.put("repair method", key);
            }else{
                metricsMap.put("repair method", method);
            }

            metricsMap.put("Number of errors", cellStatistics.getErrors());
            metricsMap.put("Number of repairs", cellStatistics.getPositives());
            metricsMap.put("Number of correct repairs", cellStatistics.getTruePositives());
            metricsMap.put("Recall", r1);
            metricsMap.put("Precision", p1);
            metricsMap.put("F1", f1);
            //ds.add(repairedDataset);

            Results.newItemInArrayObject("repair methods", metricsMap);
            if (parker_done == 0 & key.equals("Parker Algorithm")) {
                parker_done = 1;
            }
        }

    }

    public static RepairMetrics trials_population_metrics(Dataset dirty, Dataset repair, String[] selectedAttributes) throws DataReadException {
        System.out.println("Reading gold standard");
        // rawGoldStandard of Allergen dataset    
        Dataset rawGoldStandard = Binding.getRawGoldStd("trials_population", ",");

        //goldStandard is transformed dataset        
        Dataset goldStandard = new SimpleDataset();

        // get all available eudract numbers in the gold std file 
        //and collect them into a Set of Strings
        Set<String> eudractInGoldStandard = rawGoldStandard
                .stream()
                .map(o -> o.getString("eudract_number"))
                .collect(Collectors.toSet());

        // eudractNumberMap: eudract --> {eudract, protocol_country_code}
        Map<String, List<DataObject>> eudractNumberMap = repair
                .project("eudract_number", "protocol_country_code")                
                .stream()
                .collect(Collectors.groupingBy(o -> o.getString("eudract_number")));
        // convert gs from type ContractedDataset to type Dataset       

        System.out.println("Gold standard compilation");
        //
        //add protocol_country_code to each clinical trials in reaw gold std
        //multiple protocol_country_code can be related to the same clinical trial
        for (String eudract : eudractNumberMap.keySet()) {
            if (eudractInGoldStandard.contains(eudract)) {

                DataObject gso = rawGoldStandard
                        .select(g -> g.getString("eudract_number").equals(eudract))
                        .getDataObjects().get(0);

                eudractNumberMap.get(eudract)
                        .stream()
                        .map(d -> d.getString("protocol_country_code"))
                        .map(cc -> new DataObject().concat(gso)
                        .setString("protocol_country_code", cc))
                        .forEach(g -> goldStandard.addDataObject(g));
            }
        }

        System.out.println("Computing statistics");

        Set<String> attributesToEvaluate = getEvaluatedAttributes("trials_population", selectedAttributes);
        System.out.println("Attributes to evaluate:" + attributesToEvaluate);

        // get the overlaped clinical trials between the dirty dataset and the gold standard
        Dataset overlap = dirty
                .inverseProject("inclusion", "exclusion", "title", 
                        "population_age", "gender", "medical_condition")
                .select(d
                -> eudractInGoldStandard.contains(d.getString("eudract_number")));
        Dataset overlapInRepair = repair
                .inverseProject("inclusion", "exclusion", "title", 
                        "population_age", "gender", "medical_condition")
                .select(r
                -> eudractInGoldStandard.contains(r.getString("eudract_number")));
        System.out.println("Number of rows in the repaired dataset with ground truth dataset: "
                + overlapInRepair.getSize());
        
        System.out.println("overlap size:" + overlap.getSize());
        System.out.println("overlapInRepair size:" + overlapInRepair.getSize());

        if (overlap.getSize() > 0) {
            RepairMetrics cellStatistics = Statistics.getCellImputationStatistics(
                    overlap,
                    overlapInRepair,
                    goldStandard,
                    SetOperations.set("eudract_number", "protocol_country_code"),
                    SetOperations.set("eudract_number", "protocol_country_code"),
                    attributesToEvaluate,
                    true);

            System.out.println("Number of rows in original dataset: " + dirty.getSize());
            System.out.println("Number of rows in groundtruth (overlapped): " + overlap.getSize());

            Results.newIntegerObject("Number of rows in groundtruth (overlapped): ", overlap.getSize());
            return cellStatistics;
        }

        return null;
    }

    public static RepairMetrics allergens_metrics(Dataset dirty, Dataset repair, String[] selectedAttributes) throws DataReadException, IOException {

        System.out.println("Reading gold standard");
        // rawGoldStandard of Allergen dataset    
        Dataset rawGoldStandard = Binding.getRawGoldStd("allergens", ",");

        // get all available code in the gold std file 
        //and collect them into a Set of Strings
        Set<Long> codeInGoldStandard = rawGoldStandard
                .stream()
                .map(o -> o.getLong("code"))
                .collect(Collectors.toSet());

        System.out.println("--> Gold standard compilation");

        //multiple sources can be related to the same code
        // codeMap: code --> {code, source} :: Functional Dependancy
        Map<Long, List<DataObject>> codeMap = repair
                .project("code", "source")
                .stream()
                .collect(Collectors.groupingBy(o -> o.getLong("code")));//.longValue()

        Dataset goldStandard = Allergen.getGoldStd(
                rawGoldStandard,
                codeInGoldStandard,
                codeMap);

        System.out.println("--> Computing statistics");

        Set<String> attributesToEvaluate = getEvaluatedAttributes("allergens", selectedAttributes);
//        attributesToEvaluate.retainAll(repair.getDataObjects().get(0).getAttributes());
//        attributesToEvaluate.retainAll(goldStandard.getDataObjects().get(0).getAttributes());
        System.out.println("Attributes to evaluate:" + attributesToEvaluate);

        // get the overlaped clinical trials between the dirty dataset and the gold standard
        Dataset overlap = dirty.select(d -> codeInGoldStandard.contains(d.getLong("code")));
        Dataset overlapInRepair = repair.select(r -> codeInGoldStandard.contains(r.getLong("code")));
        System.out.println("Number of rows in the repaired dataset with ground truth dataset: "
                + overlapInRepair.getSize());

        if (overlap.getSize() > 0) {
            // Calculate imputation statistics on cell-level between Dataset originalDataset and Dataset repair based on given ground truth.
            // getCellImputationStatistics( originalDataset, repairedDatset,  groundTruth,  repairKey,  groundTruthKey,  diffAttributes, ignoreNoMatches)
            // = new RepairMetrics(errors, repairs, correctRepairs)
            RepairMetrics cellStatistics = Statistics.getCellImputationStatistics(
                    overlap,//originalDataset
                    //select rows that were repaired and exist in gold std dataset 
                    overlapInRepair,//repairedDatset
                    goldStandard,//groundTruth
                    //SetOperations.set(s) = Stream.of(s).collect(Collectors.toSet());
                    SetOperations.set("code", "source"),//repair Keys (i.e. attributes)
                    SetOperations.set("code", "source"),//groundTruth Keys
                    attributesToEvaluate,
                    true);

            System.out.println("Number of rows in original dataset: " + dirty.getSize());
            System.out.println("Number of rows in groundtruth (overlapped): " + overlap.getSize());

            Results.newIntegerObject("Number of rows in groundtruth (overlapped): ", overlap.getSize());

            return cellStatistics;

        }

        return null;
    }
    public static RepairMetrics eudract_metrics(Dataset dirty, Dataset repair, String[] selectedAttributes) throws DataReadException, IOException {

        // eudractNumberMap: eudract --> {eudract, protocol_country_code}
        Map<String, List<DataObject>> eudractNumberMap = repair
                .project("eudract_number", "protocol_country_code")
                .stream()
                .collect(Collectors.groupingBy(o -> o.getString("eudract_number")));

        System.out.println("Reading gold standard");

        //gs = gold std dataset
        ContractedDataset gs = Binding.readContractedData("data/golden_standards/trials_design_golden_standard.csv",
                "trials_design", ",", '"');
        //        
        // convert gs from type ContractedDataset to type Dataset       
        Dataset rawGoldStandard = gs.getAsSimpleDataset();

        Set<String> eudractInGoldStandard = rawGoldStandard
                .stream()
                .map(o -> o.getString("eudract_number"))
                .collect(Collectors.toSet());

        //goldStandard is transformed dataset        
        Dataset goldStandard = new SimpleDataset();

        System.out.println("Gold standard compilation");
        //
        //add protocol_country_code to each clinical trials in reaw gold std
        //multiple protocol_country_code can be related to the same clinical trial
        for (String eudract : eudractNumberMap.keySet()) {
            if (eudractInGoldStandard.contains(eudract)) {

                DataObject gso = rawGoldStandard
                        .select(g -> g.getString("eudract_number").equals(eudract))
                        .getDataObjects().get(0);

                eudractNumberMap.get(eudract)
                        .stream()
                        .map(d -> d.getString("protocol_country_code"))
                        .map(cc -> new DataObject().concat(gso)
                        .setString("protocol_country_code", cc))
                        .forEach(g -> goldStandard.addDataObject(g));
            }
        }

        System.out.println("Computing statistics");

        Set<String> attributesToEvaluate = getEvaluatedAttributes("trials_design", selectedAttributes);

        attributesToEvaluate.retainAll(goldStandard.getDataObjects().get(0).getAttributes());

        // get the overlaped clinical trials between the dirty dataset and the gold standard
        Dataset overlap = dirty.select(d -> eudractInGoldStandard.contains(d.getString("eudract_number")));

        // Calculate imputation statistics on cell-level between Dataset originalDataset and Dataset repair based on given ground truth.
        // getCellImputationStatistics( originalDataset, repairedDatset,  groundTruth,  repairKey,  groundTruthKey,  diffAttributes, ignoreNoMatches)
        // = new RepairMetrics(errors, repairs, correctRepairs)
        RepairMetrics cellStatistics = Statistics.getCellImputationStatistics(
                overlap,//originalDataset
                //select rows that were repaired and exist in gold std dataset 
                repair.select(r -> eudractInGoldStandard.contains(r.getString("eudract_number"))),//repairedDatset
                goldStandard,//groundTruth
                //SetOperations.set(s) = Stream.of(s).collect(Collectors.toSet());
                SetOperations.set("eudract_number", "protocol_country_code"),//repairKey
                SetOperations.set("eudract_number", "protocol_country_code"),//groundTruthKey
                attributesToEvaluate,
                true);
        SigmaRuleset eudractRules = ConstraintIo.readSigmaRuleSet(new File("rules/trials_design.rules"));

//        SufficientSigmaRuleset ruleset = SufficientSigmaRuleset.create(eudractRules, new FCFGenerator());

        System.out.println("Number of rows: " + dirty.getSize());
        System.out.println("Number of rows in groundtruth (overlapped): " + overlap.getSize());
        System.out.println("Number of FDs: " + "1");
        System.out.println("Number of attributes (" + attributesToEvaluate + "): " + attributesToEvaluate.size());

        return cellStatistics;
    }

    private static Set<String> getEvaluatedAttributes(String dataName, String[] attributes) {
        switch (dataName) {
            case "allergens":
//                return Allergen.allergenAttributes;
                return Arrays.asList(attributes).stream().collect(Collectors.toSet());

            case "trials_design":
                return Arrays.asList(attributes).stream().collect(Collectors.toSet());//Eudract.getDesignAttributes(dataName)
                
            case "trials_population":
                return Arrays.asList(attributes).stream().collect(Collectors.toSet());        

            default:
                return new HashSet<>(Arrays.asList(attributes));
        }
    }

}
