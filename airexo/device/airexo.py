from omegaconf import OmegaConf

from airexo.device.encoder import AngleEncoder


class AirExo(AngleEncoder):
    """
    AirExo Encoder.
    """
    def __init__(
        self,
        #port: 串口设备路径（如 /dev/ttyUSB0），用于与编码器通信。
        port,
        joint_cfgs,
        #baudrate: 串口通信的比特率，默认为 115200。
        baudrate = 115200,
        sleep_gap = 0.02, 
        logger_name: str = "AirExo",
        **kwargs
    ):
        #将joint_cfgs转换为OmegaConf格式，以便于配置管理。
        self.joint_cfgs = OmegaConf.create(joint_cfgs)

        self.ids = []
        self.countings = []
        self.counter = []

        for joint_id in range(1, self.joint_cfgs.num_joints + 1):
            self.ids.append(self.joint_cfgs["joint{}".format(joint_id)].id)
        
        super(AirExo, self).__init__(
            ids = self.ids,
            port = port,
            baudrate = baudrate,
            sleep_gap = sleep_gap,
            logger_name = logger_name,
            **kwargs
        )

    #获取所有关节的角度读数，然后返回一个字典
    def get_states(self):
        return {"encoder": self.get_angle()}
    