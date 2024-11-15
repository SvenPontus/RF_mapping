from kivy.uix.screenmanager import Screen

class PredictVideo(Screen):
    
    def back_to_main_menu(self):
        self.manager.current = 'main_page'


        