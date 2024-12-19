package ddcm.ugent.be.data_engineering;

import be.ugent.ledc.core.binding.DataReadException;
import be.ugent.ledc.core.binding.DataWriteException;
import be.ugent.ledc.core.dataset.DataObject;
import be.ugent.ledc.core.dataset.Dataset;
import be.ugent.ledc.core.dataset.SimpleDataset;
import be.ugent.ledc.sigma.repair.RepairException;
import ddcm.ugent.be.binding.Binding;
import ddcm.ugent.be.data_engineering.experiments.Allergen;
import ddcm.ugent.be.metrics.WebCrawling;

import java.io.IOException;
import org.junit.Test;

/**
 *
 * @author rnasfi
 */
public class AllergenTest {

    public void AllergenTest() throws DataReadException, IOException, RepairException, DataWriteException {

        System.out.print("skipping Eudract experiment -->\n");

        //Data_engineering.execute(1);
    }

    //@Test
    public void getDescriptions() throws DataReadException, DataWriteException {
        String dataName = Data_engineering.datasets[1];

        String url = "https://be.openfoodfacts.org/product/"; // URL of the website to crawl
        String targetId = "panel_ingredients_content";// openFoodFacts
        //alnatura categorie        
        //<a class="product-teaser" href="/de-de/produkte/alle-produkte/suesses-salziges/suessigkeiten/suesse-../">
        //product
        //<div id="tab-ingredients" class="tab-box__pane tab-box__pane--features js-tabs__content" style="">

        System.out.println("Fetch ingredients");

        String allergenFile = Allergen.path + "csv/allergen_train.csv";
        System.out.println(allergenFile);

        Dataset dataset = Binding.readContractedData(allergenFile, dataName, ",", '"');// to convert code into Long type

        Dataset descriptionDataset = new SimpleDataset();

        //2. Loop over each urls
        int i = 0;
        for (DataObject d : dataset) {
            String code = String.valueOf(d.getLong("code"));
            
            String ingredients = WebCrawling.crawl(url + code, targetId).replace(";", "");

            DataObject newRow = new DataObject().set("code", d.getLong("code"));
            newRow.set("ingredients", ingredients);

            descriptionDataset.addDataObject(newRow);
            
            if(i % 1000 == 0){
               System.out.println("URL: " + url + code);
               Binding.writeDataCSV(Allergen.path + "csv/ingredients_description" + i + ".csv", descriptionDataset, ",", '"'); 
            }
            i += 1;
        }
        System.out.println("done");
        Binding.writeDataCSV(Allergen.path + "csv/ingredients_description" + i + ".csv", descriptionDataset, ",", '"');
        
    }

    public void mergeAllergensIngredients() throws DataReadException {
        String dataName = Data_engineering.datasets[1];
        String ingredientsFile = Allergen.path + "csv/allergen_updated.csv";
        String allergenFile = Allergen.path + "csv/allergen_updated.csv";

        Dataset dataAllergens = Binding.readContractedData(allergenFile, dataName, ",", '"');
        Dataset dataIngredients = Binding.readData(ingredientsFile, ",", '"');// to convert code into Long type
        
        for(DataObject d1 : dataAllergens.getDataObjects()){
            for(DataObject d2 : dataIngredients.getDataObjects()){
            String ingredients1 = d1.getString("ingredients");
            String ingredients2 = d2.getString("ingredients");
            }
            
        }
    }

}
