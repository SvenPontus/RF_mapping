from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from kivy.lang import Builder

from screens.mainpage import MainPage

class GuiMain(App):

    def build(self):
        self.title = "Gui for Erik and Pontus Project"
        Builder.load_file('frontend.kv')
        sm = ScreenManager()
        sm.add_widget(MainPage(name='main_page'))

        return sm
    def exit_app(self):
        self.stop() 

if __name__ == '__main__':
    GuiMain().run()