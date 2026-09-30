import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/media/galdrux/galdrux_storage/sem_5/ROS/exp_5_ws/src/install/py_pubsub'
