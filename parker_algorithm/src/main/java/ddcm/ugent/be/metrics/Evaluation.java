/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ddcm.ugent.be.metrics;

import be.ugent.ledc.core.binding.DataReadException;
import be.ugent.ledc.core.binding.DataWriteException;
import be.ugent.ledc.core.dataset.Dataset;
import be.ugent.ledc.sigma.repair.RepairException;
import ddcm.ugent.be.binding.Binding;
import ddcm.ugent.be.data_engineering.Data_engineering;
import static ddcm.ugent.be.data_engineering.Data_engineering.getDataFolder;
import static ddcm.ugent.be.data_engineering.Data_engineering.getParkerRepairData;
import java.io.IOException;
import java.util.List;

/**
 *
 * @author rnasfi
 */
public class Evaluation {

    public static void run(Dataset data, Dataset repairedTestData, String dataName, String[] lists, String ml) throws DataReadException, IOException {

        for (String l : lists) {
            String strategy = "ML classification";
            System.out.print("\n" + strategy + ":" + l + "\n");

            String[] attributeToRepair = new String[1];
            attributeToRepair[attributeToRepair.length - 1] = l;

            Metrics.compute(data, repairedTestData, dataName, l, attributeToRepair, ml);
        }
    }

    public static void evaluationProcess(String dataName, String fileRepairedML, String repairedFileParkerML, Dataset originalData, String[] attributesToRepair, String ml, String th) throws DataReadException, IOException, RepairException, DataWriteException {
        String strategy;

        //upload a repaired datset with ML classification            
        strategy = "ML classification";
        System.out.println(fileRepairedML);
        Dataset repairedML = Binding.readContractedData(fileRepairedML, dataName, ",", '"');
        System.out.print("\n" + strategy + ":" + ml + "\n");
        if (Data_engineering.per_attribute) {
            run(originalData, repairedML, dataName, attributesToRepair, strategy);
        } else {
            Metrics.compute(originalData, repairedML, dataName, strategy, attributesToRepair, th);
        }

        // evaluate the use of Parker
        strategy = "parker";
        System.out.print("\n" + strategy + "\n");
        String repairedFile = getDataFolder(dataName) + "/" + dataName + "_" + strategy + "_test.csv";
        System.out.println("file name" + repairedFile);
//        Dataset repairedParker = Binding.readContractedData(repairedFile, dataName, ",", '"'); 
        Dataset repairedParker = getParkerRepairData(dataName, originalData, attributesToRepair, "_test");
        System.out.println("file size: " + repairedParker.getSize());
        if (Data_engineering.per_attribute) {
            run(originalData, repairedParker, dataName, attributesToRepair, strategy);
        } else {
            Metrics.compute(originalData, repairedParker, dataName, strategy, attributesToRepair, th);
        }

        strategy = "ML classification and then Parker Algorithm";
        System.out.print("\n" + strategy + ":" + ml + "\n");
        Dataset repairedMLParker = getParkerRepairData(dataName, repairedML, attributesToRepair, "_test");
        System.out.println("nb rows: " + repairedMLParker.getSize());
        if (Data_engineering.per_attribute) {
            run(originalData, repairedMLParker, dataName, attributesToRepair, strategy);
        } else {
            Metrics.compute(originalData, repairedMLParker, dataName, strategy, attributesToRepair, th);
        }

        strategy = "Parker Algorithm then ML classification";
        System.out.print("\n" + strategy + ":" + ml + "\n");
        System.out.println("file name" + repairedFileParkerML);
        Dataset repairedParkerML = Binding.readContractedData(repairedFileParkerML, dataName, ",", '"');
        if (Data_engineering.per_attribute) {
            run(originalData, repairedParkerML, dataName, attributesToRepair, strategy);
        } else {
            Metrics.compute(originalData, repairedParkerML, dataName, strategy, attributesToRepair, th);
        }
       
//        strategy = "holoclean";
//        System.out.print("\n" + strategy + "\n");
//        repairedFile = getDataFolder(dataName) + "/" + dataName + "_" + strategy + ".csv";
//        System.out.print("\n repairedFile: " + repairedFile + "\n");
//        Dataset repairedHoloClean = Binding.readContractedData(repairedFile, dataName, ",", '"');
//        Metrics.compute(originalData, repairedHoloClean, dataName, strategy, attributesToRepair, th); 
        strategy = "bclean";
        System.out.print("\n" + strategy + "\n");
        repairedFile = getDataFolder(dataName) + "/" + dataName + "_" + strategy + ".csv";
        System.out.print("\n repairedFile: " + repairedFile + "\n");
        Dataset repairedBClean = Binding.readContractedData(repairedFile, dataName, ",", '"');
        if (Data_engineering.per_attribute) {
            run(originalData, repairedBClean, dataName, attributesToRepair, strategy);
        } else {
            Metrics.compute(originalData, repairedBClean, dataName, strategy, attributesToRepair, th);
        }//        
        strategy = "raha";
        System.out.print("\n" + strategy + "\n");
        repairedFile = getDataFolder(dataName) + "/" + dataName + "_" + strategy + ".csv";
        System.out.print("\n repairedFile: " + repairedFile + "\n");
        Dataset repairedRaha = Binding.readContractedData(repairedFile, dataName, ",", '"');
        if (Data_engineering.per_attribute) {
            run(originalData, repairedRaha, dataName, attributesToRepair, strategy);
        } else {
            Metrics.compute(originalData, repairedRaha, dataName, strategy, attributesToRepair, th);
        }

    }

    public static void evaluateRepairs(String dataName, Dataset originalData, String dataFolder, String[] attributesToRepair, String cst, String ml) throws IOException, DataReadException, RepairException, DataWriteException {

        //evaluate the performance of each repair method
        String suffix = "_threshold";
        String th = "0";
        String fileRepairedML = dataFolder + "_" + ml + "_ML_repair_" + cst + "_threshold.csv";
        String repairedFileParkerML = dataFolder + "_" + ml + "_ML_repair_with_parker_threshold.csv";

        evaluationProcess(dataName, fileRepairedML, repairedFileParkerML, originalData, attributesToRepair, ml, "0");

        if (!Data_engineering.per_attribute) {
            // and save them for each repair method (including a combination)
            Results.dumpResults(Data_engineering.path + "/results/" + dataName + "/" + dataName + "-metrics_" + cst + ".json");
        } else {
            Results.dumpResults(Data_engineering.path + "/results/" + dataName + "/" + dataName + "_per_label_metrics.json");
        }

    }

    /*
        Evaluation over a list of thresholds to observe the evolution of the repair metrics.
     */
    public static void evaluateRepairs(String dataName, Dataset originalData, String dataFolder, String[] attributesToRepair, String cst, String ml, List<Double> thresholds) throws IOException, DataReadException, RepairException, DataWriteException {
        for (int i = 1; i < thresholds.size() + 1; i++) {
            System.out.print("\n threshold:" + thresholds.get(i - 1) + "\n");

            //evaluate the performance of each repair method
            String fileRepairedML = dataFolder + "_" + ml + "_ML_repair_" + cst + i + ".csv";
            String repairedFileParkerML = dataFolder + "_" + ml + "_ML_repair_with_parker" + i + ".csv";
            evaluationProcess(dataName, fileRepairedML, repairedFileParkerML, originalData, attributesToRepair, ml, Double.toString(thresholds.get(i - 1)));
        }

        if (!Data_engineering.per_attribute) {
            // and save them for each repair method (including a combination)
            Results.dumpResults(Data_engineering.path + "/results/" + dataName + "/" + dataName + "-metrics_" + cst + ".json");
        } else {
            Results.dumpResults(Data_engineering.path + "/results/" + dataName + "/" + dataName + "_per_label_metrics.json");
        }

    }

    public static void evaluateMLPredictions(Dataset originalData, String dataName, String dataFolder, String[] listMLs, String[] attributesToRepair, Boolean per_attribute) throws IOException, DataReadException {
        String[] training_with_constraints = {"without_constraints", "with_constraints"};
        for (String m : listMLs) {
            for (String cst : training_with_constraints) {

                String model = m + "_" + cst;
                String repairedFileML = dataFolder + "_" + m + "_ML_repair" + "_" + cst + ".csv";
                System.out.println("data:" + repairedFileML + "\nML:" + model);

                Dataset repairedTestData = Binding.readContractedData(repairedFileML, dataName, ",", '"');
                if (!per_attribute) {
                    Metrics.compute(originalData, repairedTestData, dataName, "ML classification", attributesToRepair, model);
                } else {
                    run(originalData, repairedTestData, dataName, attributesToRepair, model);
                }
            }
        }
        if (!per_attribute) {
            Results.dumpResults(Data_engineering.path + "/results/" + dataName + "/" + dataName + "_metrics.json");
        } else {
            Results.dumpResults(Data_engineering.path + "/results/" + dataName + "/per_label/" + dataName + "_per_label_metrics.json");
        }
    }

}
