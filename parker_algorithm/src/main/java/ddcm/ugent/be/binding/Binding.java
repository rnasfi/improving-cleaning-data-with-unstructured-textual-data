/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ddcm.ugent.be.binding;

import be.ugent.ledc.core.binding.BindingException;
import be.ugent.ledc.core.binding.DataReadException;
import be.ugent.ledc.core.binding.DataWriteException;
import be.ugent.ledc.core.binding.csv.CSVBinder;
import be.ugent.ledc.core.binding.csv.CSVDataReader;
import be.ugent.ledc.core.binding.csv.CSVDataWriter;
import be.ugent.ledc.core.binding.csv.CSVProperties;
import be.ugent.ledc.core.binding.jdbc.DBMS;
import be.ugent.ledc.core.binding.jdbc.JDBCBinder;
import be.ugent.ledc.core.binding.jdbc.JDBCDataReader;
import be.ugent.ledc.core.binding.jdbc.JDBCDataWriter;
import be.ugent.ledc.core.binding.jdbc.RelationalDB;
import be.ugent.ledc.core.binding.jdbc.schema.TableSchema;
import be.ugent.ledc.core.dataset.ContractedDataset;
import be.ugent.ledc.core.dataset.DataObject;
import be.ugent.ledc.core.dataset.Dataset;
import be.ugent.ledc.core.dataset.FixedTypeDataset;
import be.ugent.ledc.sigma.datastructures.contracts.SigmaContractorFactory;
import be.ugent.ledc.sigma.datastructures.rules.SigmaRuleset;
import be.ugent.ledc.sigma.io.ConstraintIo;
import ddcm.ugent.be.data_engineering.experiments.Allergen;
import ddcm.ugent.be.data_engineering.experiments.Population;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 *
 * @author rnasfi
 */
public class Binding {

    static public JDBCBinder binderJDBC(String schema, String local) {
        switch (local) {
            case "localhost":
                return new JDBCBinder(new RelationalDB(
                        DBMS.POSTGRESQL,
                        "localhost",
                        "5432",
                        "postgres",
                        "admin",
                        "postgres",
                        schema)//euctr
                );
            case "remote":
                return new JDBCBinder(new RelationalDB(
                        DBMS.POSTGRESQL,
                        "ddcm-services",
                        "5432",
                        "postgres",
                        "+Fk&te7lYULRFkl",
                        "data_quality_experiments",
                        schema)//eudract
                );

        }
        return null;
    }

    static public Dataset readData(String query, String schema, String local) throws DataReadException {
        JDBCBinder jdbcBinder = binderJDBC(schema, local);
        return new JDBCDataReader(jdbcBinder).readData(query);
    }

    static private CSVBinder binderCSV(String file, String sep) {
        return binderCSV(file, sep, null);

    }

    static private CSVBinder binderCSV(String file, String sep, Character quote) {
        return new CSVBinder(
                new CSVProperties(true, sep, quote,
                        Stream.of("", "?", "[]", "-").collect(Collectors.toSet())// symbols treated as null
                ), //firstLineIsHeader, columnSeparator, quoteCharacter, Stream.of("").collect(Collectors.toSet())
                new File(file));
    }

    static public Dataset readData(String file, String sep, Character quote) throws DataReadException {
        CSVBinder csvBinder = binderCSV(file, sep, quote);
        return new CSVDataReader(csvBinder)//.readData();
                .readDataWithTypeInference(100);
    }

    static public Dataset readData(String file, String sep) throws DataReadException {
        CSVBinder csvBinder = binderCSV(file, sep);
        return new CSVDataReader(csvBinder).readData();
        //  .readDataWithTypeInference(100);
    }

    static public FixedTypeDataset<String> readFixedTypeData(String file, String sep, Character quote) throws DataReadException {//String path, 
//Prepare a binding
        CSVBinder csvBinder = binderCSV(file, sep, quote);
//Read the data
        FixedTypeDataset<String> dataset = new CSVDataReader(csvBinder).readData();
        return dataset;
    }

    static public ContractedDataset readContractedData(String file, String dataname, String sep, Character quote) throws DataReadException {//String path, 
        CSVBinder csvBinder = binderCSV(file, sep, quote);
        //Read the raw data and convert datatypes 
        // readData();//. works with clinical trials
        // read with inference works
        ContractedDataset data = null;

        switch (dataname) {
            case "trials_population":
                data = new CSVDataReader(csvBinder).readDataWithTypeInference(1000);
                ContractedDataset trialsPopulationTyped = Population.fix_attributes(data);

                return trialsPopulationTyped;
                
            case "trials_design":
                data = new CSVDataReader(csvBinder).readData();
                ContractedDataset eudractTyped = data.asString("arms", SigmaContractorFactory.STRING);
                return eudractTyped;

            case "flight":
                data = new CSVDataReader(csvBinder).readDataWithTypeInference(1000);
                data
                        .asDouble("actual_departure_num") //                .asDateTime("actual_departure", "yyyy-MM-dd'T'HH:mm")
                        //                        .asDateTime("actual_arrival", "yyyy-MM-dd'T'HH:mm")
                        //                        .asDateTime("scheduled_departure", "yyyy-MM-dd'T'HH:mm")
                        //                        .asDateTime("scheduled_arrival", "yyyy-MM-dd'T'HH:mm")
                        //                        .asDate("date_collected", "yyyy-MM-dd")
                        ;
                ContractedDataset flightTyped = data;
                return flightTyped;

            case "allergens":
                data = new CSVDataReader(csvBinder).readDataWithTypeInference(100); // allergens better with reading with type inference
                ContractedDataset allergenTyped = data.asLong("code", SigmaContractorFactory.LONG);
                allergenTyped = Allergen.fix_attributes(allergenTyped);
//                if (allergenTyped.getDataObjects().get(0).getAttributes().contains("ingredients")) {
//                    allergenTyped = allergenTyped
//                            .asString("ingredients", SigmaContractorFactory.STRING)
////                            .asString("allergens", SigmaContractorFactory.STRING)
//                            .asString("traces", SigmaContractorFactory.STRING)
////                            .asString("source", SigmaContractorFactory.STRING)
//                            ;
//                }
                return allergenTyped;
            default:
                return data;
        }

        //return data;
    }

    static public ContractedDataset readContractedData(String file, String dataname, String sep) throws DataReadException {
        return readContractedData(file, dataname, sep, null);
    }
    
        static public void writeJDBC(String schema, Dataset dataset, String local, TableSchema table) throws DataWriteException, BindingException {
        JDBCBinder jdbcBinder = binderJDBC(schema, local);       
        new JDBCDataWriter(jdbcBinder, table, false).writeData(dataset);
    }

    static public void writeDataCSV(String file, Dataset dataset, String sep, Character quote) throws DataWriteException {
        CSVBinder csvBinder = binderCSV(file, sep, quote);
        new CSVDataWriter(csvBinder).writeData(dataset);
    }

    static public void writeDataCSV(String file, Dataset dataset, String sep) throws DataWriteException {
        writeDataCSV(file, dataset, sep, null);
    }

    static public void writeData(String file, String suff, Dataset dataset, String sep, Character quote) throws DataWriteException {
        String newFile = file.replace(".csv", "") + suff + ".csv";
        writeDataCSV(newFile, dataset, sep, quote);
    }

    static public void writeData(String file, String suff, Dataset dataset, String sep) throws DataWriteException {
        String newFile = file.replace(".csv", "") + suff + ".csv";
        writeDataCSV(newFile, dataset, sep, null);
    }

    // get the rows that does not violates the constraints and functional dependancies
    public static Dataset getCleanData(String file, String dataname, String rulesPath, String sep, Character quote) throws DataReadException, IOException {
        Dataset data;
        if (quote == null) {
            data = readContractedData(file, dataname, sep);
        } else {
            data = readContractedData(file, dataname, sep, quote);
        }

        //Read the rules
        SigmaRuleset eudractRules = ConstraintIo.readSigmaRuleSet(new File(rulesPath));

        Dataset cleanData = data.select(o -> eudractRules.isSatisfied(o));

        return cleanData;

    }

    public static Dataset getCleanData(Dataset data, String rulesPath) throws DataReadException, IOException {
        //Read the rules
        SigmaRuleset rules = ConstraintIo.readSigmaRuleSet(new File(rulesPath));

        Dataset cleanData = data.select(o -> rules.isSatisfied(o));

        return cleanData;

    }

    public static Dataset imputeMissingValues(String file, String dataname, String rulesPath, String sep, Character quote, String[] attributes) throws DataReadException, IOException {
        Dataset data;
        if (quote == null) {
            data = readContractedData(file, dataname, sep);
        } else {
            data = readContractedData(file, dataname, sep, quote);
        }

        for (DataObject o : data) {
            for (String a : attributes) {
                if (o.getString(a) == null) {
                    int v = Integer.parseInt(o.getString(a));
                    v = 0;
                }
            }
        }

        return data;

    }

    public static Dataset imputeMissingValues(Dataset uncleanData, String[] attributes) {
        uncleanData.stream().forEach(o -> {
        });
        for (DataObject o : uncleanData) {
            for (String a : attributes) {
                if (o.getString(a) == null) {
                    int v = Integer.parseInt(o.getString(a));
                    v = 0;
                }
            }
        }
        return uncleanData;
    }

    public static Dataset getRawGoldStd(String dataname, String sep) throws DataReadException {
        // define the file directory of the gold standard dataset
        String file = "data/golden_standards/" + dataname + "_golden_standard.csv";        
        return Binding.readData(file, sep, '"');//.getAsSimpleDataset();
    }

    // Testing writing csv file
    public static void writeCSV(String file, String sep, Character quote, Dataset data) throws FileNotFoundException, DataWriteException {
        PrintWriter writer = null;
        CSVBinder binder = binderCSV(file, sep, quote);
        boolean mustWriteHeader = false;
        List<String> attributes = data
                .stream()
                .flatMap(o -> o.getAttributes().stream())
                .distinct()
                .collect(Collectors.toList());

        for (DataObject o : data.getDataObjects()) {
            if (writer == null) {
                try {
                    //open the writer, with the correct encoding and auto-flush

                    writer = new PrintWriter(
                            new BufferedWriter(
                                    new OutputStreamWriter(
                                            new FileOutputStream(
                                                    binder.getCsvFile()),
                                            binder.getCsvProperties().getEncoding()
                                    )
                            ),
                            true);

                    if (binder.getCsvProperties() != null && binder.getCsvProperties().isFirstLineIsHeader()) {
                        mustWriteHeader = true;
                    }
                } catch (FileNotFoundException ex) {
                    throw new DataWriteException(ex);
                }
            }

            String q = "" + (binder.getCsvProperties().getQuoteCharacter() == null
                    ? ""
                    : binder.getCsvProperties().getQuoteCharacter());
            String c = binder.getCsvProperties().getColumnSeparator();
            Set<String> nullSymbols = binder.getCsvProperties().getNullSymbols();

            //Choose null symbol
            final String nil = nullSymbols
                    .stream()
                    .findFirst()
                    .orElse("")
                    .trim();
            if (mustWriteHeader) {
                String line = attributes
                        .stream()
                        .map(a -> q.concat(a).concat(q)) //Quote symbols
                        .collect(Collectors.joining(c)); //Join with separator c

                //Mark header as written
                mustWriteHeader = false;

                //Write the header
                writer.println(line);
            }

            //Write the values
            String line = "";

            line = attributes.stream().map(a -> o.get(a) == null
                    ? nil
                    : (q.concat("" + o.get(a)).concat(q))).collect(Collectors.joining(c));
            //Write the header
            writer.println(line);

        }

    }

}
