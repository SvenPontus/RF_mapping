from kivy.uix.screenmanager import Screen
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
import os

class MainPage(Screen):
    
    def open_filechooser(self):
        content = BoxLayout(orientation='vertical')
        filechooser = FileChooserListView(filters=['*.mp4', '*.avi', '*.mov'])  # Filter for video files
        content.add_widget(filechooser)

        # "Select" button to choose the file and validate the path
        select_button = Button(text="Select", size_hint_y=None, height=40)
        select_button.bind(on_release=lambda x: self.select_file(filechooser.selection))
        content.add_widget(select_button)

        # "Quit" button to close the popup without selection
        quit_button = Button(text="Quit", size_hint_y=None, height=40)
        quit_button.bind(on_release=lambda x: self.popup.dismiss())
        content.add_widget(quit_button)

        # Open the popup
        self.popup = Popup(title="Select Video File", content=content, size_hint=(0.9, 0.9))
        self.popup.open()

    def select_file(self, selection):
        if selection:
            # Update TextInput with the selected file path
            self.ids.user_video_path.text = selection[0]
            
            # Verify the video path
            video_path = selection[0]
            if os.path.isfile(video_path) and video_path.lower().endswith(('.mp4', '.avi', '.mov')):
                # Update label to show success in green
                self.ids.path_status_label.text = "The video path is valid."
                self.ids.path_status_label.color = (0, 1, 0, 1)  # Green color
            else:
                # Update label to show failure in red
                self.ids.path_status_label.text = "Invalid video path. Please select a valid video file."
                self.ids.path_status_label.color = (1, 0, 0, 1)  # Red color

        # Close the popup after selection
        self.popup.dismiss()

    def info_button(self):
        self.manager.current = 'info'

    def predict_video_button(self):
        self.manager.current = 'predict_video'
