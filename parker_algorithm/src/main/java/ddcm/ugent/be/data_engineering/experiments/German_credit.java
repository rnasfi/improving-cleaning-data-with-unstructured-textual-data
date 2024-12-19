/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ddcm.ugent.be.data_engineering.experiments;

import be.ugent.ledc.core.binding.DataReadException;
import be.ugent.ledc.core.binding.DataWriteException;
import be.ugent.ledc.core.dataset.ContractedDataset;
import be.ugent.ledc.core.dataset.DataObject;
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
import java.util.Map;

/**
 *
 * @author rnasfi
 */
public class German_credit {

    public static String[] attributes = {"checking_acco", "credit_history", "purpose", "saving_accounts", "employed_since",
        "status", "debtors", "property", "plans", "housing", "job", "phone", "foreign", "duration", "credit", "rate_income", "residence_since",
        "age", "credit_nb", "dependable", "score"};
    public static String[] nomAttr = {"checking_acco", "credit_history", "purpose", "saving_accounts", "employed_since",
        "status", "debtors", "property", "plans", "housing", "job", "phone", "foreign"};
    public static String[] ordAttr = {"duration", "credit", "rate_income", "residence_since",
        "age", "credit_nb", "dependable", "score"};

    public static void test(String file, String dataName) throws DataReadException, RepairException, DataWriteException, FileNotFoundException {

        FixedTypeDataset<String> data1 = Binding.readFixedTypeData(file, ",", null);
        ContractedDataset data2 = Binding.readContractedData(file, dataName, ",", null);
        Dataset data3 = Binding.readData(file, ",");

        SigmaRuleset rules1 = Mining.associationBased(data1, nomAttr);
        SufficientSigmaRuleset sufficientAssociationRules = SufficientSigmaRuleset.create(rules1, new FCFGenerator());

        SigmaRuleset rules2 = Mining.liftBased(data1, nomAttr);

        SigmaRuleset rules3 = Mining.monotone(data2, ordAttr);
        SufficientSigmaRuleset sufficientMonotoneRules = SufficientSigmaRuleset.create(rules3, new FCFGenerator());

        SigmaRuleset rules = rules1.merge(rules2).merge(rules3);
//        rules.stream().filter(rule -> rule.getInvolvedAttributes().size() > 1).
//                forEach(System.out::println);

        System.out.println("****sufficientRules: ");
        SufficientSigmaRuleset sufficientRules = SufficientSigmaRuleset.create(rules, new FCFGenerator());

//        SufficientSigmaRuleset sufficientRuleset1 = sufficientMonotoneRules.merge(sufficientAssociationRules);
        sufficientRules.stream().filter(rule -> rule.getInvolvedAttributes().size() > 1).
                forEach(System.out::println);

        ConstraintIo.writeSigmaRuleSet(rules, new File("rules/german_credit.rules"));

        ConstantCostModel costModel = new ConstantCostModel(Costing.assignConstantCost(attributes));

        Dataset repairedDataset = Repairing.sequentialRepair(data2, sufficientRules, costModel, rules);
        Binding.writeData(file, "_repaired", repairedDataset, ",");

        Map<DataObject, Double> outliers = Mining.zscore(data3, ordAttr);//.isolationForest(data3, ordAttr);//.zscore(data3, ordAttr);
        System.out.println("#outliers in german credits dataset: " + outliers.size());
    }

    public static ContractedDataset fix_attributes(ContractedDataset data) {
        return data
                .asInteger("age", SigmaContractorFactory.INTEGER)
                .asInteger("job", SigmaContractorFactory.INTEGER)
                .asInteger("credit_amount", SigmaContractorFactory.INTEGER)
                .asInteger("duration", SigmaContractorFactory.INTEGER);
    }

    public static ContractedDataset fix_attributes_v1(ContractedDataset data) {
        for (String s : ordAttr) {
            data = data.asInteger(s, SigmaContractorFactory.INTEGER);
        }
        return data;
    }
}
