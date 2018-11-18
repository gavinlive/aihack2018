import os,sys,csv, collections
import numpy as np

class DataManager(object):
    def __init__(self):
        self.path="data/california/california/train/"
        self.meta_path=self.path+"BG_METADATA_2016.csv"

        self.data_filenames = []
        for root, dirs, filenames in os.walk(self.path):
            self.data_filenames = self.data_filenames + [x for x in filenames if x[0]=='X']
        print(self.data_filenames)

        labels = {}
        with open(self.meta_path) as f:
            reader = csv.reader(f)
            next(reader)
            count = 0
            for row in reader:
                labels[row[1]] = row[2]
        self.labels = labels
    def import_data(self, filename):
        data = {}
        data_path=self.path+filename
        objectid_idx = None
        geoid_idx = None
        objectids = []
        header = None
        with open(data_path) as f:
            reader = csv.reader(f)
            count = 0
            for row in reader:
                if(count==0):
                    objectid_idx = row.index("OBJECTID")  # check for the object ID
                    geoid_idx = row.index("GEOID")
                    header = row[:]
                    del(header[objectid_idx])
                    del(header[geoid_idx])
                    del(header[0])
                else:
                    this_row = row[:]
                    del(this_row[objectid_idx])
                    del(this_row[geoid_idx])
                    del(this_row[0])
                    for i, x in enumerate(this_row):
                        if(x==""):
                            this_row[i] = 0
                        else:
                            this_row[i] = np.float32(x)
                    data[str(row[objectid_idx])] = this_row
                    objectids.append(row[objectid_idx])
                count+=1
        #print("header")
        #print(header)
        return objectids, data, header
    @staticmethod
    def find_missing_data(lol):
        missing = []
        main_list = lol[0]
        del(lol[0])
        for i in main_list:
            for other_list in lol:
                present = (i in other_list)
                if(present==False):
                    missing.append(i)
                    break
        return missing
    def import_all(self):
        list_of_objectidlists = []
        list_of_data = []
        list_of_headers = []
        headers = []
        for filename in self.data_filenames:
            print(filename)
            objectids, data, header = self.import_data(filename)
            list_of_data.append(data)
            list_of_objectidlists.append(objectids)
            list_of_headers.append(header)
            headers = headers + header
        [print(len(x)) for x in list_of_objectidlists]
        #missing = self.find_missing_data(list_of_objectidlists)
        #print(missing)
        #print(len(missing))
        main_list = list_of_objectidlists[0]
        aggregated = []
        for this_id in main_list:
            this_agg = []
            for this_data in list_of_data:
                #print(this_data)
                #print(this_id)
                this_agg = this_agg +this_data[str(this_id)]
            aggregated.append(this_agg)
        self.data = aggregated
        self.headers = headers
        print('all headers')
        #print(headers)
        return headers, aggregated
