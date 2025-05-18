import lcm
from lcm_msg.exlcm import example_t
import threading
import time


class LCMPub:
    def __init__(self):
        """ Step1: LCM通信实例化 """
        self.lcm = lcm.LCM()
        """ Step2: LCM发布的处理 """
        # 线程锁，防止消息处理过程被修改
        self.pub_mutex = threading.Lock()
        self.pub_msg = None

    def _handle_pub(self, data):
        current_time = int(time.time() * 1e9)  # 纳秒时间戳
        """
            int64_t  timestamp;
            double   position[3];
            double   orientation[4]; 
            int32_t  num_ranges;
            int16_t  ranges[num_ranges];
            string   name;
            boolean  enabled;
        """
        with self.pub_mutex:
            self.pub_msg = {
                'timestamp' : current_time,
                'position' : [data.position[0], data.position[1], data.position[2]],
                'orientation' : [data.orientation[0], data.orientation[1], data.orientation[2], data.orientation[3]],
                'num_ranges' : range(15),
                'num_ranges' : len(range(15)),
                'name' : "example string", 
                'enabled' : True
            }

    def pub_msg(self):
        """ 将多个话题同时发布 """
        with self.pub_mutex:
            self.lcm.publish("EXAMPLE", self.pub_msg.encode())


    def main(self):
        data = []
        count = 0
        while True:
            count += 1
            data.position = [count, 0, 0]
            self._handle_pub(data)
            self.pub_msg()
            time.sleep(1)


if __name__ == "__main__":
    lcmPub = LCMPub()
    lcmPub.main()