from lib.multi_gpu_frame import multi_gpu_model as resources_model
from lib.data_frame import Data_V2 as Data
import os, sys
import tensorflow as tf
import collections
from pprint import pprint
#from data.load_mnist import load_data_light as load_data

from lib.execution import execution
DataConfiguration = collections.namedtuple("DataConfiguration", ['project_path', 'execution_type', 'model_load_dir'])
SystemConfiguration = collections.namedtuple("SystemConfiguration", ["cpu_only", "num_gpus", "eager", "mini_batch_size", "validation_size"])

''' Config here '''
from load_data import load_data as Data_Loader
#from capsules.architectures.v1_rev4_1 import architecture as Architecture_old
#from models.dc_cnn.architectures.main import architecture as Architecture
from models.dc_cnn.architectures.main import architecture as Architecture
experiment_name = 'dc_cnn_main_aliased_RAND0.29_1_DATASET'
data_config = DataConfiguration(project_path='/vol/biomedic/users/kgs13/projects/aihack/nn/',execution_type='evaluate',
    model_load_dir='dc_cnn_main_aliased_RAND0.29_1_DATASET_2018-11-18-04:00:52.645154')
system_config = SystemConfiguration(cpu_only=False, num_gpus=1, eager=False, mini_batch_size=64, validation_size=64)
load_data = Data_Loader() # acc=0.29
''' End Config '''
try:
  if system_config.eager==True:
    tf.enable_eager_execution()
  DataModel = Data(load_data, num_gpus=system_config.num_gpus, validation_size=system_config.validation_size)
  print("Start resource manager...")
  System = resources_model(cpu_only=system_config.cpu_only,eager=system_config.eager)
  print("Create Network Architecture...")
  CapsuleNetwork = Architecture()
  print("Strap Architecture to Resource Manager")
  System.strap_architecture(CapsuleNetwork)

  print("Strap Managed Architecture to a training scheme `Executer`")
  with execution(data_config.project_path, System, DataModel, experiment_name=experiment_name, max_steps_to_save=5, mini_batch_size=system_config.mini_batch_size, type=data_config.execution_type, load=data_config.model_load_dir) as Executer:
        Executer.run_task(max_epochs=1000000, save_step=1)
except Exception as e:
  err_message = e.args
  print("Exception thrown, see below:")
  print(err_message)
  pprint(e)
