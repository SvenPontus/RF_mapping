from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

class MyWidget(BoxLayout):
    pass

class MyApp(App):
    def build(self):
        return MyWidget()

    def on_button_click(self):
        print("Knappen är klickad!")

# Startar appen
if __name__ == "__main__":
    MyApp().run()
