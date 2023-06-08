import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QFile
from ui import Ui_ND
from core import main


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_ND()
        self.ui.setupUi(self)
        self.setWindowTitle("Оптимизация начальных параметров нейтрализационного диализа")

        # обработчик событий
        self.ui.pushButton.clicked.connect(self.button_listener)

    def button_listener(self):
        float_input = []
        bool_input = []

        float_input.append(float(self.ui.NaCl.value()))
        float_input.append(float(self.ui.HClstart.value()))
        float_input.append(float(self.ui.HClend.value()))
        float_input.append(float(self.ui.NaOHstart.value()))
        float_input.append(float(self.ui.NaOHend.value()))
        float_input.append(float(self.ui.DBLstart.value()))
        float_input.append(float(self.ui.DBLend.value()))

        if self.ui.model2.isChecked():
            bool_input.append(True)
        else:
            bool_input.append(False)

        if self.ui.opt2.isChecked():
            bool_input.append(True)
        else:
            bool_input.append(False)
        self.ui.output1.clear()
        self.ui.output2.clear()
        self.ui.output1.append("Выбраны следующие начальные параметры:")
        self.ui.output1.append("\nКонцентрация NaCl: " + str(float_input[0]))
        self.ui.output1.append("Концентрация HCl: от " + str(float_input[1]) + " до " + str(float_input[2]))
        self.ui.output1.append("Концентрация NaOH: от " + str(float_input[3]) + " до " + str(float_input[4]))
        self.ui.output1.append("Толщина DBL: от " + str(float_input[5]) + " до " + str(float_input[6]))
        if bool_input[0]:
            self.ui.output1.append("\nМетод моделирования: нейросетевой")
        else:
            self.ui.output1.append("\nМетод моделирования: численный")

        if (bool_input[1]):
            self.ui.output1.append("\nМетод оптимизации: отжиг Больцмана")
        else:
            self.ui.output1.append("\nМетод оптимизации: быстрый спуск")

        self.ui.pushButton.setText("Обработка")
        self.ui.pushButton.setEnabled(False)
        QApplication.processEvents()

        output1, output2 = main(float_input, bool_input)

        self.ui.output1.append("\n")
        self.ui.output1.append(output1)
        self.ui.output2.append(output2)

        self.ui.pushButton.setText("Начать")
        self.ui.pushButton.setEnabled(True)
        QApplication.processEvents()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())