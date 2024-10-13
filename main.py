import sys
from PyQt5 import QtWidgets
from controllers.main_controller import MainController
from utils.config_manager import ConfigManager
from utils.constants import PADDING_SIZE  # pixels

def main():
    """Main function to run the application."""
    app = QtWidgets.QApplication(sys.argv)

    # Initialize ConfigManager
    config_manager = ConfigManager()

    # Initialize MainController with ConfigManager
    controller = MainController(config_manager=config_manager)
    controller.run()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()