class Repairer:
    def __init__(self, dataset, parker=False):
        """
        dataset (dict): a set of metadata about the repaired dataset        
        """
        self.dataset = dt.datasets[dataset_index]
        self.data_index = dataset_index
        self.keys = self.dataset['keys']
        self.partial_key = self.dataset['keys'][0]
        self.labels = self.dataset['labels']
        self.features = self.dataset['features'][0]
        self.dataName = self.dataset['data_dir']

    def test_model_robustness(self):
        """
        Returns: 
            statistics (dict): a set of metrics
        """
        statistics = {}

        #read test data
        dtest = dt.read_test_csv(self.dataName, False)

        source = self.keys[1]      
        cols = [self.partial_key, self.features, source]
        
        sources = self.get_vars(dtest,  source)
        print('Sources', sources)

        logging.info(self.dataset)
        print('+++++++++++++++++++++Start+++++++++++++++++++++++++++++')

        # Suppress all warnings 
        warnings.filterwarnings("ignore")
        
        # each time the nb of samples is increased 
        # for s in range(len(sources)):
        #     self.data = dtest[dtest[source].isin(sources[:s+1])]
        #     print(self.data.shape, sources[:s+1])
        #     statistics[s] = {}
        #     for ind, a in enumerate(self.labels):            
        #         print('label',a)
        #         # save appart the original values
        #         self.data[a + '_orig'] = self.data[a].values

        #         ## load saved model
        #         file_model_name = os.path.join('.', 'models', \
        #             f"_{a}_classifier_{self.model_name}_{self.strategy}.pth")
        #         with open(file_model_name, 'rb') as f: model = pickle.load(f)   


        #         # get the encoder if exists
        #         enc = {}
        #         y_orig = self.data[a + '_orig']
        #         y_gs = self.data[a + '_gs']

        #         enc, y_orig, y_gs = tp.encode(self.get_data_encoders(), a, self.data)
        #         print("------ done encoding ----------")

        #         y_pred, outputs, dtest, accuracy = tr.clf_test(model, self.data, a, self.dataset, enc)
        #         print("------ done predicting ----------")

        #         # replace with the predicted values 
        #         if len(enc) > 0:
        #             self.data[a] = tp.decode(enc, a, y_pred)
        #         else: self.data[a] = y_pred

        #         time.sleep(1)

        #         attrs = self.labels[:ind+1]

        #         # compute the repair metrics for the 
                
        #         statistics[s][ind + 1] = {}
        #         statistics[s][ind + 1]['correct_repairs'] = eva.get_all_stats(self.data, attrs)[0]
        #         statistics[s][ind + 1]['repairs'] = eva.get_all_stats(self.data, attrs)[1]
        #         statistics[s][ind + 1]['errors'] = eva.get_all_stats(self.data, attrs)[2]

        #         print(len(sources[:s+1]), eva.get_all_stats(self.data, attrs))
        #         print(attrs, 'data size', self.data[attrs].shape)
        #         logging.info("")
        #         logging.info("repair metrics for %s (data size: %s)", attrs, self.data[attrs].shape)
        #         logging.info(statistics)
        #         print(f"+++++++++++++++++++++done with {a} ({ind + 1})+++++++++++++++++++++++++++++")
        #         print()

        #         # if ind > 1 : break             
        #     print('+++++++++++++++++++++more sources+++++++++++++++++++++++++++++')
        #     print()

        self.statistics = statistics
        return statistics            