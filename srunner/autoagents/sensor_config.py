class SensorConfiguration:
    def __init__(self, sensor_setup, img_w, img_h):
        self.sensor_setup = sensor_setup
        self.img_w = img_w 
        self.img_h = img_h 

    def get_sensor_configuration(self):
        """
        Define the sensor suite required by the agent
        """
        img_w = self.img_w
        img_h = self.img_h 
        base_sensors = [
                            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
                            {'type': 'sensor.other.imu', 'id': 'IMU'},
                            {'id': 'front_rgb','type': 'sensor.camera.rgb', 'x':0, 'y': 0, 'z':2.2, 'pitch':0, 
                                     'yaw': 0, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                       ]
        sensors_hd_map = base_sensors + [
                            {'id': 'bev_sem','type': 'sensor.camera.semantic_segmentation', 'x':0, 'y': 0, 'z':40, 'pitch':-90, 
                             'yaw': 0, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                         ]

        sensors_hd_map_frontal = sensors_hd_map + [
                                    {'id': 'front_sem', 'type': 'sensor.camera.semantic_segmentation', 'x':.4, 'y': 0, 'z':1.5, 'pitch':0, 
                                     'yaw': 0, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                                    {'id': 'front_right_sem', 'type': 'sensor.camera.semantic_segmentation', 'x':.4, 'y': 0, 'z':1.5, 'pitch':0, 
                                    'yaw': 60, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                                    {'id': 'front_left_sem', 'type': 'sensor.camera.semantic_segmentation', 'x':.4, 'y': 0, 'z':1.5, 'pitch':0, 
                                    'yaw': -60, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                                ]
        full_view = base_sensors + [   
                        {'id': '360_0_sem','type': 'sensor.camera.semantic_segmentation', 'x':0, 'y': 0, 'z':2.2, 'pitch':0, 
                                     'yaw': 0, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                        {'id': '360_1_sem','type': 'sensor.camera.semantic_segmentation', 'x':0, 'y': 0, 'z':2.2, 'pitch':0, 
                            'yaw': 60, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                        {'id': '360_2_sem','type': 'sensor.camera.semantic_segmentation', 'x':0, 'y': 0, 'z':2.2, 'pitch':0, 
                            'yaw': 120, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100}, 
                        {'id': '360_3_sem','type': 'sensor.camera.semantic_segmentation', 'x':0, 'y': 0, 'z':2.2, 'pitch':0, 
                            'yaw': 180, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                        {'id': '360_4_sem','type': 'sensor.camera.semantic_segmentation', 'x':0, 'y': 0, 'z':2.2, 'pitch':0, 
                            'yaw': 240, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                        {'id': '360_5_sem','type': 'sensor.camera.semantic_segmentation', 'x':0, 'y': 0, 'z':2.2, 'pitch':0, 
                            'yaw': 300, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                    ]
        
        if self.sensor_setup == 'hd_map':
            sensors = sensors_hd_map
        elif self.sensor_setup == 'hd_map_frontal':
            sensors = sensors_hd_map_frontal
        elif self.sensor_setup == '360_map':
            sensors = full_view
        return sensors