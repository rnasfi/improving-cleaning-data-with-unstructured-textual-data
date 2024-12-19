/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ddcm.ugent.be.data_engineering;

import be.ugent.ledc.core.dataset.Dataset;
import be.ugent.ledc.sigma.datastructures.rules.SigmaRuleset;
import be.ugent.ledc.sigma.datastructures.rules.SufficientSigmaRuleset;
import be.ugent.ledc.sigma.repair.NullBehavior;
import be.ugent.ledc.sigma.repair.RepairEngine;
//import be.ugent.ledc.sigma.repair.RepairException;
import be.ugent.ledc.sigma.repair.SequentialRepair;
import be.ugent.ledc.sigma.repair.JointRepair;
import be.ugent.ledc.sigma.repair.ParkerRepair;
import be.ugent.ledc.sigma.repair.RepairException;
import be.ugent.ledc.sigma.repair.cost.models.ConstantCostModel;
import be.ugent.ledc.sigma.repair.cost.models.ParkerModel;
import be.ugent.ledc.sigma.repair.selection.FrequencyRepairSelection;
import be.ugent.ledc.sigma.repair.selection.RandomRepairSelection;

/**
 *
 * @author rnasfi
 */
public class Repairing {

    private static RepairEngine sequentialRepairEngine(SufficientSigmaRuleset sufficientRules, ConstantCostModel costModel, SigmaRuleset rules, Dataset cleanData) throws RepairException {
        return new SequentialRepair(
                sufficientRules,
                costModel,
                NullBehavior.CONSTRAINED_REPAIR,
                cleanData
        );
    }

    public static Dataset sequentialRepair(Dataset data, SufficientSigmaRuleset sufficientRules, ConstantCostModel costModel, SigmaRuleset rules) throws RepairException {
        Dataset repaired = sequentialRepairEngine(sufficientRules, costModel, rules, data.select(o -> rules.isSatisfied(o))).repair(data);
        return repaired;

    }

    public static Dataset jointRepair(Dataset data, SufficientSigmaRuleset rules, Dataset cleanData) throws RepairException {
        return (new JointRepair(rules, cleanData)).repair(data);
    }

    public static Dataset parkerRepair(Dataset data, ParkerModel costModel, SufficientSigmaRuleset rules, Dataset cleanData) throws RepairException {
        //Pass the cost model and the repair selector to make a parker engine
        return (new ParkerRepair(
                costModel,
                new FrequencyRepairSelection(cleanData, rules)
        )).repair(data);

    }

    public static Dataset parkerRepair(Dataset data, ParkerModel costModel) throws RepairException {
        return new ParkerRepair(
                costModel,
                new RandomRepairSelection()
        ).repair(data);

    }

}
