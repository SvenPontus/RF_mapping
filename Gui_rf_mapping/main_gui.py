from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from kivy.lang import Builder

from screens.mainpage import MainPage
from screens.info import Info
from screens.predict_video import PredictVideo

class GuiMain(App):

    def build(self):
        self.title = "RF Mapping"
        Builder.load_file('frontend.kv')
        sm = ScreenManager()
        sm.add_widget(MainPage(name='main_page'))
        sm.add_widget(Info(name='info'))
        sm.add_widget(PredictVideo(name='predict_video'))

        return sm
    def exit_app(self):
        self.stop() 

if __name__ == '__main__':
    GuiMain().run()