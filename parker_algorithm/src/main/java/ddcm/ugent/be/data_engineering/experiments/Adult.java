/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ddcm.ugent.be.data_engineering.experiments;

import be.ugent.ledc.core.binding.DataReadException;
import be.ugent.ledc.core.dataset.ContractedDataset;
import be.ugent.ledc.sigma.datastructures.contracts.SigmaContractorFactory;
import be.ugent.ledc.sigma.datastructures.rules.SigmaRuleset;
import ddcm.ugent.be.binding.Binding;
import ddcm.ugent.be.data_engineering.Mining;

/**
 *
 * @author rnasfi
 */
public class Adult {

    public static void test(String file, String dataName) throws DataReadException {

        String[] adultsAttr = {"workclass", "education", "marital-status", "occupation", "relationship", "sex", "native-country", "age",
            "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"};
        String[] adultAttrOrd = {"age", "fnlwgt", "education-num",
            "capital-gain", "capital-loss", "hours-per-week"};

        ContractedDataset dataAdult = Binding.readContractedData(file, dataName, ",", null);
        SigmaRuleset rules2 = Mining.monotone(dataAdult, adultAttrOrd);
        rules2.stream().filter(rule -> rule.getInvolvedAttributes().size() > 1).
                forEach(System.out::println);
    }

    public static ContractedDataset fix_attributes(ContractedDataset data) {
        return data
                .asInteger("age", SigmaContractorFactory.INTEGER)
                .asInteger("fnlwgt", SigmaContractorFactory.INTEGER)
                .asInteger("education-num", SigmaContractorFactory.INTEGER)
                .asInteger("capital-gain", SigmaContractorFactory.INTEGER)
                .asInteger("capital-loss", SigmaContractorFactory.INTEGER)
                .asInteger("hours-per-week", SigmaContractorFactory.INTEGER);
    }
}
