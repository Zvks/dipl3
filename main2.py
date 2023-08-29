# Uncomment these lines to see all the messages
# from kivy.logger import Logger
# import logging
# Logger.setLevel(logging.TRACE)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
import time
import numpy
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image


Builder.load_string('''
<CameraClick>:
    BoxLayout:
        orientation: 'vertical'
        Camera:
            id: camera
            resolution: (640, 480)
            play: True
        ToggleButton:
            text: 'Play'
            on_press: camera.play = not camera.play
            size_hint_y: None
            height: '48dp'
        Button:
            text: 'Capture'
            size_hint_y: None
            height: '48dp'
            on_press: root.capture()
        Button:
            text: 'Detection'
            size_hint_y: None
            height: '48dp'
            on_press: root.manager.current = 'output'
<OutputScreen>:
    BoxLayout:
        Label:
            text: 'Дефекты металлической поверхности'
        Button:
            text: 'Back to menu'
            size_hint_y: None
            height: '48dp'
            on_press: root.manager.current = 'camera'
''')


class CameraClick(Screen):
    def capture(self):
        col_image = []
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        texture = camera.texture
        size=texture.size
        pixels = texture.pixels
        pil_image=Image.frombytes(mode='RGBA', size=size,data=pixels)
        col_image = numpy.array(pil_image.convert('L'))
        print("Captured")
        plt.imshow(col_image)
        plt.show()
        col_image = img_to_array(col_image)
        col_image = preprocess_input(col_image)
        return col_image
    collection = []
    col_image = capture
    collection.append(col_image)
    print(collection, 'вася')
class OutputScreen(Screen):
    pass

class TestCamera(App):

    def build(self):
        sm = ScreenManager()
        sm.add_widget(CameraClick(name='camera'))
        sm.add_widget(OutputScreen(name='output'))
        return CameraClick()
    


TestCamera().run()




