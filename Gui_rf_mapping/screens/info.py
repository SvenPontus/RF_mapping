from kivy.uix.screenmanager import Screen

class Info(Screen):

    def on_enter(self):
        # Ställer in texten för etiketten med id "info_label"
        self.ids.info_label.text = """Application Information

This application has been developed as part of a research project and is not intended for commercial use. The model was trained using DeepLabCut and the application utilizes Miniconda3 and Kivy frameworks to facilitate the work of researchers and their colleagues.

Licensing and Usage Rights

This software is provided for research purposes only. Any modification, redistribution, or commercial use of this application, in part or in whole, is prohibited without explicit permission from the original developers.

Attributions

Model Training: DeepLabCut
Environment Management: Miniconda3
User Interface: Kivy

How to use

---"""
    
    def back_to_main_menu(self):
        self.manager.current = 'main_page'