import sys
from PyQt5 import QtWidgets
from controllers.main_controller import MainController

def main():
    """Main function to run the application."""
    app = QtWidgets.QApplication(sys.argv)
    controller = MainController()
    controller.run()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()