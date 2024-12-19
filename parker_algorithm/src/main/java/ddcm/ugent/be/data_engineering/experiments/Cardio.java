/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ddcm.ugent.be.data_engineering.experiments;

import be.ugent.ledc.core.binding.DataReadException;
import be.ugent.ledc.core.binding.DataWriteException;
import be.ugent.ledc.core.dataset.ContractedDataset;
import be.ugent.ledc.core.dataset.Dataset;
import be.ugent.ledc.core.dataset.FixedTypeDataset;
import be.ugent.ledc.sigma.datastructures.contracts.SigmaContractorFactory;
import be.ugent.ledc.sigma.datastructures.rules.SigmaRuleset;
import be.ugent.ledc.sigma.datastructures.rules.SufficientSigmaRuleset;
import be.ugent.ledc.sigma.io.ConstraintIo;
import be.ugent.ledc.sigma.repair.RepairException;
import be.ugent.ledc.sigma.repair.cost.models.ConstantCostModel;
import be.ugent.ledc.sigma.sufficientsetgeneration.FCFGenerator;
import ddcm.ugent.be.binding.Binding;
import ddcm.ugent.be.data_engineering.Costing;
import ddcm.ugent.be.data_engineering.Mining;
import ddcm.ugent.be.data_engineering.Repairing;
import java.io.File;
import java.io.FileNotFoundException;

/**
 *
 * @author rnasfi
 */
public class Cardio {

    public static void test(String file, String dataName) throws DataReadException, FileNotFoundException, DataWriteException, RepairException {
        String[] cardioAttrOrd //, "ap_hi", "ap_lo"
                = {"age", "gender", "height", "cholesterol", "gluc", "smoke", "alco", "active", "cardio", "weight"};

        ContractedDataset dataCardio1 = Binding.readContractedData(file, dataName, ",", null);
        Dataset dataCardio2 = Binding.readData(file, ",");
        FixedTypeDataset<String> dataCardio3 = Binding.readFixedTypeData(file, ",", null);
        System.out.println("#Cardio set objects: " + dataCardio2.getSize());

        SigmaRuleset rulesCardio = Mining.monotone(dataCardio1, cardioAttrOrd);

        SufficientSigmaRuleset sufficientRules = SufficientSigmaRuleset.create(rulesCardio, new FCFGenerator());
        sufficientRules.stream().filter(rule -> rule.getInvolvedAttributes().size() > 1).
                forEach(System.out::println);

        ConstraintIo.writeSigmaRuleSet(sufficientRules, new File("rules/cardio.rules"));

        ConstantCostModel costModel = new ConstantCostModel(Costing.assignConstantCost(cardioAttrOrd));

        Dataset cleanData = dataCardio1.select(o -> sufficientRules.isSatisfied(o));
        Dataset repairedDataset = Repairing.sequentialRepair(dataCardio1, sufficientRules, costModel, rulesCardio);
        Binding.writeData(file, "_repaired", repairedDataset, ",");

//        Dataset jointRepairedDataset = Repairing.jointRepair(dataCardio1, sufficientRules, cleanData);
//        Binding.writeData(file, "_jointRepaired", jointRepairedDataset, ",");
//        Map<DataObject, Double> outliersCardio = Mining.isolationForest(dataCardio2, cardioAttrOrd);//.zscore(data3, ordAttr);
//        System.out.println("#outliers in cardio dataset: " + outliersCardio.size());
    }

    public static ContractedDataset fix_attributes(ContractedDataset data) {
        return data
                .asInteger("age", SigmaContractorFactory.INTEGER)
                .asInteger("gender", SigmaContractorFactory.INTEGER)
                .asInteger("height", SigmaContractorFactory.INTEGER)
                .asBigDecimal("weight", SigmaContractorFactory.BIGDECIMAL_ONE_DECIMAL)
                .asInteger("ap_hi", SigmaContractorFactory.INTEGER)
                .asInteger("ap_lo", SigmaContractorFactory.INTEGER)
                .asInteger("cholesterol", SigmaContractorFactory.INTEGER)
                .asInteger("gluc", SigmaContractorFactory.INTEGER)
                .asInteger("smoke", SigmaContractorFactory.INTEGER)
                .asInteger("alco", SigmaContractorFactory.INTEGER)
                .asInteger("active", SigmaContractorFactory.INTEGER)
                .asInteger("cardio", SigmaContractorFactory.INTEGER);
    }
}
