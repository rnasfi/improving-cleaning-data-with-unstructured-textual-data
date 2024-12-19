/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ddcm.ugent.be.data_engineering.experiments;

import be.ugent.ledc.core.binding.DataReadException;
import be.ugent.ledc.core.binding.DataWriteException;
import be.ugent.ledc.core.dataset.Dataset;
import be.ugent.ledc.sigma.datastructures.fd.PartialKey;
import be.ugent.ledc.sigma.datastructures.rules.SigmaRuleset;
import be.ugent.ledc.sigma.datastructures.rules.SufficientSigmaRuleset;
import be.ugent.ledc.sigma.repair.RepairException;
import be.ugent.ledc.sigma.repair.cost.models.ParkerModel;
import be.ugent.ledc.sigma.sufficientsetgeneration.FCFGenerator;
import ddcm.ugent.be.binding.Binding;
import ddcm.ugent.be.data_engineering.Data_engineering;
import ddcm.ugent.be.data_engineering.Repairing;
import java.io.IOException;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 *
 * @author rnasfi
 */
public class Eudract {
    private static boolean done = true;

    public static String[] designAttribute = {"parallel_group", "randomised", "crossover", "controlled", "arms", "open", "single_blind", "double_blind"};

    public static String[] maskingAttribute = {"open", "single_blind", "double_blind"};

    public static String[] assignAttribute = {"parallel_group", "randomised", "crossover"};

    public static String[] controlAttribute = {"controlled", "arms", "active_comparator", "placebo"}; //, "active_comparator", "placebo"

    public static String[] assigningAttributes = {"single_blind", "double_blind", "open", "randomised", "controlled", "crossover",
        "parallel_group", "arms", "active_comparator", "placebo"};

    public static final String path = Data_engineering.path + "/data/trials_design";

    public static Dataset repair(Dataset data_to_repair, String dataName, SigmaRuleset eudractRules, String[] partK, String train) throws DataReadException, RepairException, IOException, DataWriteException {

        SigmaRuleset eudractRulesp = eudractRules.project(partK);

        SufficientSigmaRuleset sufficientEudractRules = SufficientSigmaRuleset.create(eudractRulesp, new FCFGenerator());

        String eudractFile = path + "/trials_design.csv";

        Dataset data = Binding.readContractedData(eudractFile, dataName, ",", '"');

        Dataset cleanData = data.select(o -> eudractRules.isSatisfied(o))
                .inverseProject("eudract_number", "protocol_country_code", "source");

        System.out.print("dataset that satisfy the rules size " + cleanData.getSize() + "\n");

        ParkerModel costModel = new ParkerModel(getPartialKey(partK), sufficientEudractRules);

        Dataset parkerRepaired = Repairing.parkerRepair(data_to_repair, costModel, sufficientEudractRules, cleanData);

        if(done){
            System.out.println(path + "/trials_design_parker" + train + ".csv");
            Binding.writeDataCSV(path + "/trials_design_parker" + train + ".csv", parkerRepaired, ",", '"');
            done = false;
        }
        
        return parkerRepaired;
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
}
