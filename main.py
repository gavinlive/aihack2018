import data_manager
import gc

DM = data_manager.DataManager()
_,_ = DM.import_all()
'''total_data = [headers] + data
import csv
print(len(headers[0]))
with open(path+'total_agg_2.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i in total_data:
        wr.writerow(i)'''


print(len(DM.data[0]))
print(len(DM.headers))

print(DM.data[0])

del(DM.data[0])



gc.collect()
