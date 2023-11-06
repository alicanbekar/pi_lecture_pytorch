import ipywidgets as widgets
from IPython.display import display, Javascript, clear_output
from ipywidgets import Layout


class LoadingInterface:
      def __init__(self):
        self.dropdown_name_nn = widgets.Dropdown(
            options=['Plain-UNet', 'UNet-BCs', 'Flux-UNet', 'Hybrid-UNet'],
            value='Plain-UNet',
        )

        # Create a button to print the values
        self.select_button = widgets.Button(
            description='Select Network',
        )
        self.select_button.on_click(self.select_values)

      def display(self):
        # Organize the widgets using HBox and VBox with HTML descriptions
        display(widgets.VBox([
            widgets.HBox([
                widgets.HTML('<p style="width:250px">NN Constraints:</p>'),
                self.dropdown_name_nn
            ]),
            self.select_button
        ]))
      
      def select_values(self, button):
          print(f'NN Constraints: {self.dropdown_name_nn.value}')

class TrainingInterface:
    def __init__(self):
        # Create widgets
        self.batch_size_slider = widgets.SelectionSlider(
            options=[8, 16, 32, 64],
            value=32,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True
        )

        self.num_epochs_slider = widgets.IntSlider(
            value=10,
            min=2,
            max=50,
            step=1,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True
        )

        self.learning_rate_slider = widgets.FloatLogSlider(
            value=0.001,
            base=10,
            min=-4,  # max exponent of base
            max=-1,  # min exponent of base
            step=1,  # exponent step
            readout=True,
            readout_format='.4f'
        )

        self.num_timesteps_slider = widgets.IntSlider(
            value=50,
            min=10,
            max=100,
            step=10,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True
        )

        self.dropdown_name_nn = widgets.Dropdown(
            options=['Plain-UNet', 'UNet-BCs', 'Flux-UNet', 'Hybrid-UNet'],
            value='Plain-UNet',
        )

        # Create a button to print the values
        self.print_button = widgets.Button(
            description='Print Values',
        )
        self.print_button.on_click(self.print_values)

    def display(self):
        # Organize the widgets using HBox and VBox with HTML descriptions
        display(widgets.VBox([
            widgets.HBox([
                widgets.HTML('<p style="width:250px">Batch Size:</p>'),
                self.batch_size_slider
            ]),
            widgets.HBox([
                widgets.HTML('<p style="width:250px">Number of Epochs:</p>'),
                self.num_epochs_slider
            ]),
            widgets.HBox([
                widgets.HTML('<p style="width:250px">Learning Rate:</p>'),
                self.learning_rate_slider
            ]),
            widgets.HBox([
                widgets.HTML('<p style="width:250px">Number of ICs:</p>'),
                self.num_timesteps_slider
            ]),
            widgets.HBox([
                widgets.HTML('<p style="width:250px">NN Constraints:</p>'),
                self.dropdown_name_nn
            ]),
            self.print_button
        ]))

    def print_values(self, button):
        # Accessing widget values and printing them
        print(f'Batch Size: {self.batch_size_slider.value}')
        print(f'Number of Epochs: {self.num_epochs_slider.value}')
        print(f'Learning Rate: {self.learning_rate_slider.value}')
        print(f'Number of ICs: {self.num_timesteps_slider.value}')
        print(f'NN Constraints: {self.dropdown_name_nn.value}')

class NeuralNetworkInterface:
    def __init__(self):
        self.load_button = widgets.Button(description="Load Results")
        self.train_button = widgets.Button(description="Train")

        # Set up button event handlers
        self.load_button.on_click(self.load_results)
        self.train_button.on_click(self.train)

    def display(self):
        # Display the buttons
        display(widgets.HBox([self.load_button, self.train_button]))

    def load_results(self, button):
        # Example action for loading weights
        self.loading_interface = LoadingInterface()
        self.loading_interface.display()
        self.train = False
        self.dropdown_name_nn = self.loading_interface.dropdown_name_nn
        clear_output(wait=True)
        

    def train(self, button):
        # Display the TrainingInterface for parameter selection
        self.training_interface = TrainingInterface()
        self.training_interface.display()
        self.train = True
        self.dropdown_name_nn = self.training_interface.dropdown_name_nn
        clear_output(wait=True)
