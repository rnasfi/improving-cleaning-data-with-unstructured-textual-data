/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ddcm.ugent.be.metrics;


import be.ugent.ledc.core.dataset.DataObject;
import be.ugent.ledc.core.dataset.Dataset;
import ddcm.ugent.be.exceptions.MultipleMatchException;
import ddcm.ugent.be.exceptions.NoMatchException;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

public class Statistics {

    /**
     * Calculate imputation statistics on cell-level between Dataset
     * originalDataset and Dataset repair based on given ground truth.
     *
     * error: cell in originalDataset differs in value with same cell in
     * groundTruth repair: cell in originalDataset differs in value with same
     * cell in repair correctRepair: cell in repair differs in value with same
     * cell in originalDataset but not with same cell in groundTruth
     *
     * @param originalDataset Original Dataset object that is repaired.
     * @param repair Repair of the original Dataset object.
     * @param groundTruth Dataset containing ground truth data.
     * @param repairKey Set of attributes that is used to match a data object
     * from originalDataset to a data object from repair. This set of attributes
     * should feature unique values (key) in repair, otherwise, multiple matches
     * can be found.
     * @param groundTruthKey Set of attributes that is used to match a data
     * object from originalDataset to a data object from the ground truth. *
     * This set of attributes should feature unique values (key) in the ground
     * truth, otherwise, multiple matches can be found.
     * @param diffAttributes Attributes that play a role in calculating the
     * statistics.
     * @return RepairMetrics object containing statistics on cell-level between
     * Dataset originalDataset and Dataset repair based on given ground truth.
     * @throws NoMatchException
     * @throws MultipleMatchException
     */
    public static RepairMetrics getCellImputationStatistics(Dataset originalDataset, Dataset repair, Dataset groundTruth, Set<String> repairKey, Set<String> groundTruthKey, Set<String> diffAttributes, boolean ignoreNoMatches) throws NoMatchException {

        long errors = 0;
        long repairs = 0;
        long correctRepairs = 0;
        // all attributes -> projected keys
        Map<DataObject, List<DataObject>> groundTruthMap = groundTruth
                .stream()
                .collect(Collectors.groupingBy(o -> o.project(groundTruthKey)));

        Map<DataObject, List<DataObject>> repairMap = repair
                .stream()
                .collect(Collectors.groupingBy(o -> o.project(repairKey)));

        int multipleMatches = 0;
        int err = 0;
        for (DataObject originalObject : originalDataset.getDataObjects()) {

            try {

                DataObject matchingGroundTruthObject = getFirstMatchingDataObject(originalObject, groundTruthMap, groundTruthKey);
                DataObject matchingRepairObject = getFirstMatchingDataObject(originalObject, repairMap, repairKey);

                Set<String> originalGTDiff = matchingGroundTruthObject.diff(originalObject, diffAttributes);
                Set<String> repairOriginalDiff = matchingRepairObject.diff(originalObject, diffAttributes);

                errors += originalGTDiff.size(); //how many cells are different per row
                
//                if(originalObject.getString("eudract_number").equals("2012-002633-11")){
//                    System.out.println("orig: \n \t" + originalObject);
//                    System.out.println("GroundTruthObject: \n \t" + matchingGroundTruthObject);
//                    System.out.println("RepairObject: \n \t" + matchingRepairObject);                    
//                }
                
//                if (!originalGTDiff.isEmpty()) {
//                    err += 1;
//                    System.out.println("orig: \n \t" + originalObject);
//                    System.out.println("GroundTruthObject: \n \t" + matchingGroundTruthObject);
//                    System.out.println("RepairObject: \n \t" + matchingRepairObject);
//                }
                repairs += repairOriginalDiff.size();
                correctRepairs += repairOriginalDiff
                        .stream()
                        .filter(a -> Objects.equals(
                        matchingRepairObject.get(a),
                        matchingGroundTruthObject.get(a)))
                        .count();
//                if(correctRepairs != repairs){
//                    System.out.println("\n" + originalObject.getString("new_lang")); 
//                    System.out.println("\t" + matchingGroundTruthObject.project(repairKey));
//                    System.out.println("\t" + originalObject.project(diffAttributes));
//                    System.out.println("--ground truth:");
//                    System.out.println("\t" + matchingGroundTruthObject.project(diffAttributes));
//                    System.out.println("--repaired:");
//                    System.out.println("\t" + matchingRepairObject.project(diffAttributes));
//                }
//                if(!repairOriginalDiff.isEmpty()){ 
//                    System.out.print("\n \n originalObject: \n \t"+ originalObject);
//                    System.out.print("\n original Ground Truth Diff: \t"+ originalGTDiff + "\n");
//                    System.out.println("RepairObject: \n \t"+ matchingRepairObject);
//                    System.out.println("GroundTruthObject: \n \t"+ matchingGroundTruthObject);
//                    System.out.println("diff btw original and repair: \t" + repairOriginalDiff + "\n");
//                }
            } catch (MultipleMatchException ex) {
                multipleMatches++;
                //Continue
            } catch (NoMatchException ex) {
                if (!ignoreNoMatches) {
                    throw new NoMatchException(ex);
                }
            }
        }

        if (multipleMatches > 0) {
            System.out.println("Warning: there where " + multipleMatches + " multiple matches for keys");
        }

        return new RepairMetrics(errors, repairs, correctRepairs);

    }


    /**
     * Find a match in given Dataset object to for DataObject from based on the
     * values of matchingAttributes
     *
     * @param from DataObject to search a match for in given dataset.
     * @param lookup Lookup transformed dataset in which a match is searched for
     * given DataObject from
     * @param matchingAttributes Find a match based on the values of
     * matchingAttributes. If the parameter ignoreMultipleMatches is set to
     * false, these attributes should feature unique values (key) for each
     * object in Dataset object to, otherwise, multiple matches can be found and
     * an exception will be thrown.
     * @return First match found in Dataset object to for given DataObject from
     * @throws NoMatchException
     * @throws MultipleMatchException
     *
     * TODO: adopt this method in class DataObject?
     *
     */
    public static DataObject getFirstMatchingDataObject(DataObject from, Map<DataObject, List<DataObject>> lookup, Set<String> matchingAttributes) throws NoMatchException, MultipleMatchException {

        List<DataObject> dataObjects = lookup.get(from.project(matchingAttributes));

        if (dataObjects == null || dataObjects.size() == 0) {
            throw new NoMatchException("No matching objects found for data object " + from + " based on attributes " + matchingAttributes);
        }

        if (dataObjects.size() > 1) {
            throw new MultipleMatchException("Multiple matching objects found for data object " + from + ". This is not permitted as set of matching attributes " + matchingAttributes + " should be key in dataset");
        }

        return dataObjects.get(0);

    }

}

