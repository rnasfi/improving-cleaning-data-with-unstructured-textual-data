/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ddcm.ugent.be.data_engineering;

import be.ugent.ledc.sigma.datastructures.fd.PartialKey;
import be.ugent.ledc.sigma.repair.cost.functions.ConstantCostFunction;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import be.ugent.ledc.sigma.datastructures.rules.SufficientSigmaRuleset;
import be.ugent.ledc.sigma.repair.RepairException;
import be.ugent.ledc.sigma.repair.cost.models.ParkerModel;
import be.ugent.ledc.sigma.datastructures.fd.PartialKey;

/**
 *
 * @author rnasfi
 */
public class Costing {

    public static Map<String, ConstantCostFunction> assignConstantCost(String... attributes) {
        //The cost model assumes constant cost for each attribute
        Map<String, ConstantCostFunction> costMap = new HashMap<>();

        for (String a : attributes) {
            costMap.put(a, new ConstantCostFunction(1));
        }

        return costMap;
    }

    public static Map<String, ConstantCostFunction> adultCosts() {
        //The cost model assumes constant cost for each attribute
        Map<String, ConstantCostFunction> costMap = new HashMap<>();

        costMap.put("income", new ConstantCostFunction(1));
        costMap.put("education", new ConstantCostFunction(1));
        costMap.put("education_num", new ConstantCostFunction(1));
        costMap.put("occupation", new ConstantCostFunction(1));
        costMap.put("race", new ConstantCostFunction(1));
        costMap.put("sex", new ConstantCostFunction(1));
        costMap.put("hours_per_week", new ConstantCostFunction(1));
        costMap.put("fnlwgt", new ConstantCostFunction(1));
        costMap.put("capital_loss", new ConstantCostFunction(1));
        costMap.put("capital_gain", new ConstantCostFunction(1));
        costMap.put("native_country", new ConstantCostFunction(1));
        costMap.put("marital_status", new ConstantCostFunction(1));
        costMap.put("workclass", new ConstantCostFunction(1));
        costMap.put("relationship", new ConstantCostFunction(1));
        costMap.put("age", new ConstantCostFunction(1));

        return costMap;
    }

    public static ParkerModel costParkerModel(SufficientSigmaRuleset sufficientRules, String[] pks, String[] attributes) throws RepairException {
        return new ParkerModel(new PartialKey(
                Stream.of(pks).collect(Collectors.toSet()),
                Stream.of(attributes).collect(Collectors.toSet()) ),
                sufficientRules);
    }

}
