import lcm
import threading
import time
# from lcm_types.franka.franka_states_t import franka_states_t
"""
    这个类比较简单，也没做太多处理，其实应该按需要把他做的更好一些的
"""

class LCM:
    def __init__(self, channel):
        """ Step1: LCM通信实例化 """
        self.lcm = lcm.LCM()
        self.channel = channel
        """ Step2: LCM发布的处理 """
        # 线程锁，防止消息处理过程被修改
        # self.pub_mutex = threading.Lock()

    def pub_msg(self, msg):
        # with self.pub_mutex:
            # self.lcm.publish(self.channel, msg.encode())
        self.lcm.publish(self.channel, msg.encode())


