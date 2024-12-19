The following dataset are composed of two parts :
* structured tabular data
** unstructured textual data

### Eudract Trial design
This dataset was obtained from the European Union Drug Regulating Authorities Clinical Trials Database ([EudraCT](https://eudract.ema.europa.eu/)) register 
and the ground truth was created by Antoon Bronselaer and Maribel Acosta, [Parker: Data fusion through consistent repairs using edit rules under partial keys](https://www.sciencedirect.com/science/article/abs/pii/S1566253523002580), Information Fusion, (**2023**) from external registries. 
In the dataset, multiple countries, identified by the attribute country_protocol_code, conduct the same clinical trials which is identified by eudract_number. 
Each clinical trial has a title that can help  find informative details about the design of the trial.

### Eudract Trials population
This dataset delineates the demographic origins of participants in clinical trials primarily conducted across European countries. 
This dataset include structured attributes indicating whether the trial pertains to a specific gender, age group or healthy volunteers. 
Each of these categories is labeled as (`1') or (`0') respectively denoting whether it is included in the trials or not. 
It is important to note that the population category should remain consistent across all countries conducting the same clinical trial identified by an eudract_number. 
The ground truth samples in the dataset were established by aligning information about the trial populations provided by external registries, specifically the [CT.gov](https://clinicaltrials.gov/) database and the [German Trials database](https://drks.de/search/en).  
Additionally, the dataset comprises other unstructured attributes that categorize the inclusion criteria for trial participants such as inclusion.

### allergens
The data was retrieved from different websites, which are: 
- [Alnatura](https://www.alnatura.de/)
- [Open Food facts](https://world.openfoodfacts.org/)
- [Migipedia](https://migipedia.migros.ch/)
- [Piccantino ](https://www.piccantino.com)
- [Das ist Drin](http://das-ist-drin.de/) 

There exist two csv files that include samples of the products from these websites:
- allergen_eng.csv : 1636 rows giving description of food products (source, bar code, ingredients, set of allergens that it contains).
- allergens_golden_standard.csv : 103 row of food products with the ground truth of the allergens.
