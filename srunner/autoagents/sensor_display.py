import pygame 
import numpy as np

class Display(object):

    """
    Class to control a vehicle manually for debugging purposes
    """

    def __init__(self, width, height, sensor_setup):
        self.use_pygame = False
        self._width = width
        self._height = height
        self._surface = None
        self.setup_sensor_specifics(sensor_setup)

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display_name = f"Displaying sensors: {sensor_setup}"
        self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption(self._display_name)

    def setup_sensor_specifics(self, sensor_setup):
        if sensor_setup == 'hd_map':
            self.layout_func = self.hd_map_layout
            self._height = self._height * 2
        elif sensor_setup == 'hd_map_frontal':
            self.layout_func = self.hd_map_frontal_layout
            self._width = self._width * 3 
            self._height = self._height * 2 
        elif sensor_setup == '360_map':
            self.layout_func = self.full_view_layout
            self._width = self._width * 3 
            self._height = self._height * 2 

    def run_interface(self, input_data):
        """
        Run the GUI
        """
        # process sensor data
        # image_center = input_data['bev_sem'][1][:, :, -2::-1]
        layout = self.layout_func(input_data)
        
        # display image
        self._surface = pygame.surfarray.make_surface(layout.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()
        # to avoid freezes
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return


    def quit_interface(self):
        """
        Stops the pygame window
        """
        pygame.quit()
    
    def hd_map_layout(self, input_data):
        return np.concatenate([input_data['bev_sem'][1][:, :, -2::-1],
                               input_data['front_rgb'][1][:,:, -2::-1]], axis=0)
        # return input_data['bev_sem'][1][:, :, -2::-1]
    
    def hd_map_frontal_layout(self, input_data):
        hd_img = input_data['bev_sem'][1][:, :, -2::-1]
        row1 = np.concatenate([input_data['front_left_sem'][1][:, :, -2::-1], input_data['front_sem'][1][:, :, -2::-1], 
                               input_data['front_right_sem'][1][:, :, 2::-1]], axis=1)
        row2 = np.concatenate([np.zeros_like(hd_img), hd_img, np.zeros_like(hd_img)], axis=1)
        tiled = np.concatenate([row1,row2], axis=0)
        return tiled 
    
    def full_view_layout(self, input_data):
        row1 = np.concatenate([input_data['360_5_sem'][1][:, :, -2::-1], input_data['360_0_sem'][1][:, :, -2::-1], 
                               input_data['360_1_sem'][1][:, :, -2::-1]], axis=1)
        row2 = np.concatenate([input_data['360_4_sem'][1][:, :, -2::-1], input_data['360_3_sem'][1][:, :, -2::-1], 
                               input_data['360_2_sem'][1][:, :, -2::-1]], axis=1)
        tiled = np.concatenate([row1,row2], axis=0) 
        return tiled