/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ddcm.ugent.be.data_engineering;

import be.ugent.ledc.core.binding.DataReadException;
import be.ugent.ledc.core.dataset.ContractedDataset;
import be.ugent.ledc.core.dataset.DataObject;
import be.ugent.ledc.core.dataset.Dataset;
import be.ugent.ledc.core.dataset.FixedTypeDataset;
import be.ugent.ledc.core.operators.aggregation.tnorm.BasicTNorm;
import be.ugent.ledc.dino.outlierdetection.isolationforest.IsolationForestDetector;
import be.ugent.ledc.dino.outlierdetection.isolationforest.splitter.IntegerSplitter;
import be.ugent.ledc.dino.outlierdetection.zscore.ZScoreOutlierDetector;
import be.ugent.ledc.sigma.datastructures.rules.SigmaRuleset;
import be.ugent.ledc.dino.rulemining.association.ConfidenceMiner;
import be.ugent.ledc.dino.rulemining.ordinal.OrdinalBinaryMiner;
import be.ugent.ledc.dino.rulemining.tlift.TGenerator;
import java.util.Map;

/**
 *
 * @author rnasfi
 */
public class Mining {

    public static SigmaRuleset associationBased(FixedTypeDataset<String> dataset, String... attr) {
     
        ConfidenceMiner cm = new ConfidenceMiner.ConfidenceMinerBuilder(0.3, 0.98) //Pass alpha (support threshold) and beta (confidence threshold)
                .withEpsilon(0.9) //Optional epsilon parameter to avoid low-frequent values being "pushed out" by high frequent ones
                .withSigma(0.9) //Optional sigma parameters which is generalization of lift.
                .build();
     
        SigmaRuleset rules = cm.findRules(dataset, attr);

        return rules;
    }

    public static SigmaRuleset liftBased(FixedTypeDataset<String> dataset, String... attr) {
        SigmaRuleset rules = new TGenerator(
                0.01, //Threshold for the lift: patterns with lower lift are considered erroneous
                true, //Ignore null values in pattern computation
                BasicTNorm.MINIMUM //Triangular norm that must be used for the computation of the lift.
        ).findRules(dataset, attr);

        return rules;

    }

    public static SigmaRuleset monotone(ContractedDataset dataset, String... attr) throws DataReadException {
        SigmaRuleset rules = new OrdinalBinaryMiner(0.1)
                .findRules(
                        dataset, attr
                );

        return rules;
    }

    public static Map<DataObject, Double> zscore(Dataset dataset, String... attr) {
        Map<DataObject, Double> outliers = new ZScoreOutlierDetector()
                .findOutliers(dataset, attr
                );
        return outliers;

    }

    public static Map<DataObject, Double> isolationForest(Dataset dataset, String... attr) {
        Map<DataObject, Double> outliers = new IsolationForestDetector(
                new IntegerSplitter(),
                0.6, //Threshold for outlier score
                100, //Number of trees
                256 //Sample size
        ).findOutliers(dataset, attr );
        return outliers;

    }
}
