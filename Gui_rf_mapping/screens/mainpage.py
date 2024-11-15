from kivy.uix.screenmanager import Screen
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

class MainPage(Screen):
    
    def open_filechooser(self):
        # Create a layout for the popup content
        content = BoxLayout(orientation='vertical')
        filechooser = FileChooserListView(filters=['*.mp4', '*.avi', '*.mov'])  # Filter for video files
        content.add_widget(filechooser)

        # Add a button to select the file
        select_button = Button(text="Select", size_hint_y=None, height=40)
        select_button.bind(on_release=lambda x: self.select_file(filechooser.selection))
        content.add_widget(select_button)

        # Create the popup window
        self.popup = Popup(title="Select Video File", content=content, size_hint=(0.9, 0.9))
        self.popup.open()

    def select_file(self, selection):
        if selection:
            # Set the file path to the TextInput
            self.ids.user_video_path.text = selection[0]
        self.popup.dismiss()  # Close the popup window
