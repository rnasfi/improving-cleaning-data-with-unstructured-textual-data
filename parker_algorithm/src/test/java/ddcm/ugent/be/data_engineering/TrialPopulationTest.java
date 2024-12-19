/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ddcm.ugent.be.data_engineering;

import be.ugent.ledc.core.binding.DataReadException;
import be.ugent.ledc.core.binding.DataWriteException;
import be.ugent.ledc.core.dataset.ContractedDataset;
import be.ugent.ledc.core.dataset.DataObject;
import be.ugent.ledc.core.dataset.Dataset;
import be.ugent.ledc.core.dataset.FixedTypeDataset;
import be.ugent.ledc.sigma.datastructures.rules.SigmaRuleset;
import be.ugent.ledc.sigma.io.ConstraintIo;
import be.ugent.ledc.sigma.repair.RepairException;
import ddcm.ugent.be.binding.Binding;
import ddcm.ugent.be.data_engineering.experiments.Eudract;
import ddcm.ugent.be.data_engineering.experiments.Population;
import static ddcm.ugent.be.data_engineering.experiments.Population.path;
import java.io.File;

import java.io.IOException;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.Test;

/**
 *
 * @author rnasfi
 */
public class TrialPopulationTest {
//    @Test

    public void testRule() throws DataReadException, IOException, DataWriteException {
        FixedTypeDataset<String> data = Binding.readFixedTypeData(path + "/csv/population_ex_in_clusion.csv", ",", '"');
        DataObject obj = data.getDataObjects().get(0);

        SigmaRuleset rules2 = Mining.liftBased(data, Population.healthAttribute);
        rules2.stream().filter(rule -> rule.getInvolvedAttributes().size() > 1).
                forEach(System.out::println);
        
        Population.validate_population_labeling();
        
//        Map<DataObject, Double> outliers =  Mining.isolationForest(data, Population.healthAttribute);
//        outliers.keySet().stream().filter(o -> outliers.get(o) < 0.7).forEach(System.out::println);

//        SigmaRuleset rules = ConstraintIo
//                .readSigmaRuleSet(new File("rules/trials_population.rules"));
//
//        System.out.println("Is the tuple ");
//        System.out.print(obj + " \n violating? ");
//        System.out.print(rules.isSatisfied(obj) + "\n");

    }

    public void validate() throws DataReadException, DataWriteException {

        Population.filling_corpus();

        Dataset data = Binding.readData(path + "/csv/population_full_data.csv", ",", '"');
        System.out.println("nb rows " + data.getSize());

        DataObject elt = data.getDataObjects().get(6);

        System.out.println("title:" + elt.getString("title"));

        String included = elt.getString("inclusion");
        System.out.println("included:" + included);

        String excluded = elt.getString("exclusion");
        System.out.println("excluded:" + excluded);

        if (included != null) {
            boolean valid = Population.isLabelTrue(included, "male");
            System.out.println("\noriginal male value: " + elt.getString("male") + " -- detected in the inclusion population: " + valid);
        }

    }
//@Test

    public void checkAge() {
        Population.filling_corpus();
        String text = "1. >/= 18 years of age and able to read and write. If over 40 years of age;"
                + "If over 40 years of age; must have FSH < 40 IU/mL "
                + "2. DUB defined as at least one of the following symptoms within the 90-day run-in phase: "
                + "(i) Prolonged bleeding: 2 or more bleeding episodes; each lasting 8 or more days "
                + "(ii) Frequent bleeding: greater than 5 bleeding episodes; "
                + "with a minimum of 20 bleeding days overall "
                + "(iii) Excessive bleeding: 2 or more bleeding episodes "
                + "each with blood loss volume of 80 mL or more; as assessed by the alkaline hematin method "
                + "3. Willingness to use barrier contraception (e.g.; condoms) from screening to study completion "
                + "4. Willingness to use and collect sanitary protection (pads and tampons) "
                + "provided by the sponsor and compatible with the alkaline hematin test throughout study completion"
                + " 5. Normal or clinically insignificant Pap smear results. A report within the last 6 months of visit 1 is acceptable. "
                + "6. Endometrial biopsy during the run-in phase "
                + "OR a valid endometrial biopsy performed within 6 months of visit "
                + "1; without evidence of malignancy or atypical hyperplasia; with an available report. "
                + "Women with simple hyperplasia can be included in the study; "
                + "but will undergo an endometrial biopsy at the end of treatment. "
                + "7. Signed the Informed consent form (ICF).";

        String w = "\\b(?:[^>])+([10|[0-9]])\\s(?:age|years|of years)+\\b";
        Pattern pattern = Pattern.compile(w, Pattern.CASE_INSENSITIVE);

        Matcher matcher = pattern.matcher(text.toLowerCase());

        while (matcher.find()) {
            System.out.println("Found at position " + matcher.start() + ": " + matcher.group());
            System.out.println("matcher: " + matcher);
            System.out.println("keyword: " + w);

        }
        System.out.print(Population.checkAge("adults", text));

    }
}
