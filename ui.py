# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui.ui'
##
## Created by: Qt User Interface Compiler version 6.5.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QGroupBox, QLabel,
    QMainWindow, QMenuBar, QPushButton, QRadioButton,
    QSizePolicy, QStatusBar, QTextBrowser, QWidget)

class Ui_ND(object):
    def setupUi(self, ND):
        if not ND.objectName():
            ND.setObjectName(u"ND")
        ND.resize(999, 732)
        self.centralwidget = QWidget(ND)
        self.centralwidget.setObjectName(u"centralwidget")
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(10, 10, 981, 671))
        self.pushButton = QPushButton(self.widget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(430, 620, 121, 41))
        font = QFont()
        font.setPointSize(10)
        self.pushButton.setFont(font)
        self.output1 = QTextBrowser(self.widget)
        self.output1.setObjectName(u"output1")
        self.output1.setGeometry(QRect(0, 40, 481, 381))
        self.groupBox = QGroupBox(self.widget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(0, 440, 981, 161))
        self.groupBox.setFont(font)
        self.label_3 = QLabel(self.groupBox)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(10, 30, 51, 21))
        self.NaCl = QDoubleSpinBox(self.groupBox)
        self.NaCl.setObjectName(u"NaCl")
        self.NaCl.setGeometry(QRect(90, 30, 81, 22))
        self.NaCl.setDecimals(3)
        self.NaCl.setMinimum(0.001000000000000)
        self.NaCl.setMaximum(1.000000000000000)
        self.NaCl.setSingleStep(0.001000000000000)
        self.label_4 = QLabel(self.groupBox)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(10, 60, 51, 21))
        self.HClstart = QDoubleSpinBox(self.groupBox)
        self.HClstart.setObjectName(u"HClstart")
        self.HClstart.setGeometry(QRect(90, 60, 81, 22))
        self.HClstart.setDecimals(3)
        self.HClstart.setMinimum(0.001000000000000)
        self.HClstart.setMaximum(1.000000000000000)
        self.HClstart.setSingleStep(0.001000000000000)
        self.HClend = QDoubleSpinBox(self.groupBox)
        self.HClend.setObjectName(u"HClend")
        self.HClend.setGeometry(QRect(190, 60, 81, 22))
        self.HClend.setDecimals(3)
        self.HClend.setMaximum(1.000000000000000)
        self.HClend.setSingleStep(0.001000000000000)
        self.HClend.setValue(1.000000000000000)
        self.label_5 = QLabel(self.groupBox)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(10, 90, 51, 21))
        self.NaOHstart = QDoubleSpinBox(self.groupBox)
        self.NaOHstart.setObjectName(u"NaOHstart")
        self.NaOHstart.setGeometry(QRect(90, 90, 81, 22))
        self.NaOHstart.setDecimals(3)
        self.NaOHstart.setMinimum(0.001000000000000)
        self.NaOHstart.setMaximum(1.000000000000000)
        self.NaOHstart.setSingleStep(0.001000000000000)
        self.NaOHend = QDoubleSpinBox(self.groupBox)
        self.NaOHend.setObjectName(u"NaOHend")
        self.NaOHend.setGeometry(QRect(190, 90, 81, 22))
        self.NaOHend.setDecimals(3)
        self.NaOHend.setMaximum(1.000000000000000)
        self.NaOHend.setSingleStep(0.001000000000000)
        self.NaOHend.setValue(1.000000000000000)
        self.label_6 = QLabel(self.groupBox)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(80, 60, 16, 21))
        self.label_7 = QLabel(self.groupBox)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(80, 90, 16, 21))
        self.label_8 = QLabel(self.groupBox)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(180, 60, 16, 21))
        self.label_9 = QLabel(self.groupBox)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(180, 90, 16, 21))
        self.label_10 = QLabel(self.groupBox)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(280, 60, 16, 21))
        self.label_11 = QLabel(self.groupBox)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setGeometry(QRect(280, 90, 16, 21))
        self.label_12 = QLabel(self.groupBox)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(280, 120, 16, 21))
        self.label_13 = QLabel(self.groupBox)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setGeometry(QRect(180, 120, 16, 21))
        self.DBLend = QDoubleSpinBox(self.groupBox)
        self.DBLend.setObjectName(u"DBLend")
        self.DBLend.setGeometry(QRect(190, 120, 81, 22))
        self.DBLend.setDecimals(3)
        self.DBLend.setMinimum(50.000000000000000)
        self.DBLend.setMaximum(150.000000000000000)
        self.DBLend.setSingleStep(0.001000000000000)
        self.DBLend.setValue(150.000000000000000)
        self.label_14 = QLabel(self.groupBox)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(80, 120, 16, 21))
        self.DBLstart = QDoubleSpinBox(self.groupBox)
        self.DBLstart.setObjectName(u"DBLstart")
        self.DBLstart.setGeometry(QRect(90, 120, 81, 22))
        self.DBLstart.setDecimals(3)
        self.DBLstart.setMinimum(50.000000000000000)
        self.DBLstart.setMaximum(150.000000000000000)
        self.DBLstart.setSingleStep(0.001000000000000)
        self.label_15 = QLabel(self.groupBox)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(10, 120, 51, 21))
        self.groupBox_2 = QGroupBox(self.groupBox)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(500, 20, 201, 121))
        self.model1 = QRadioButton(self.groupBox_2)
        self.model1.setObjectName(u"model1")
        self.model1.setGeometry(QRect(10, 40, 181, 20))
        self.model1.setChecked(True)
        self.model2 = QRadioButton(self.groupBox_2)
        self.model2.setObjectName(u"model2")
        self.model2.setGeometry(QRect(10, 70, 181, 20))
        self.groupBox_3 = QGroupBox(self.groupBox)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(740, 20, 201, 121))
        self.opt1 = QRadioButton(self.groupBox_3)
        self.opt1.setObjectName(u"opt1")
        self.opt1.setGeometry(QRect(10, 40, 181, 20))
        self.opt1.setChecked(True)
        self.opt2 = QRadioButton(self.groupBox_3)
        self.opt2.setObjectName(u"opt2")
        self.opt2.setGeometry(QRect(10, 60, 181, 51))
        self.output2 = QTextBrowser(self.widget)
        self.output2.setObjectName(u"output2")
        self.output2.setGeometry(QRect(500, 40, 481, 381))
        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(0, 10, 481, 16))
        self.label.setFont(font)
        self.label_2 = QLabel(self.widget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(500, 10, 481, 16))
        self.label_2.setFont(font)
        ND.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(ND)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 999, 26))
        ND.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(ND)
        self.statusbar.setObjectName(u"statusbar")
        ND.setStatusBar(self.statusbar)

        self.retranslateUi(ND)

        QMetaObject.connectSlotsByName(ND)
    # setupUi

    def retranslateUi(self, ND):
        ND.setWindowTitle(QCoreApplication.translate("ND", u"MainWindow", None))
        self.pushButton.setText(QCoreApplication.translate("ND", u"\u041d\u0430\u0447\u0430\u0442\u044c", None))
        self.groupBox.setTitle(QCoreApplication.translate("ND", u"\u041d\u0430\u0447\u0430\u043b\u044c\u043d\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u043d\u0435\u0439\u0442\u0440\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u043e\u043d\u043d\u043e\u0433\u043e \u0434\u0438\u0430\u043b\u0438\u0437\u0430", None))
        self.label_3.setText(QCoreApplication.translate("ND", u"NaCl", None))
        self.label_4.setText(QCoreApplication.translate("ND", u"HCl", None))
        self.label_5.setText(QCoreApplication.translate("ND", u"NaOH", None))
        self.label_6.setText(QCoreApplication.translate("ND", u"[", None))
        self.label_7.setText(QCoreApplication.translate("ND", u"[", None))
        self.label_8.setText(QCoreApplication.translate("ND", u";", None))
        self.label_9.setText(QCoreApplication.translate("ND", u";", None))
        self.label_10.setText(QCoreApplication.translate("ND", u"]", None))
        self.label_11.setText(QCoreApplication.translate("ND", u"]", None))
        self.label_12.setText(QCoreApplication.translate("ND", u"]", None))
        self.label_13.setText(QCoreApplication.translate("ND", u";", None))
        self.label_14.setText(QCoreApplication.translate("ND", u"[", None))
        self.label_15.setText(QCoreApplication.translate("ND", u"DBL", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("ND", u"\u041c\u0435\u0442\u043e\u0434 \u043c\u043e\u0434\u0435\u043b\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f", None))
        self.model1.setText(QCoreApplication.translate("ND", u"\u0427\u0438\u0441\u043b\u0435\u043d\u043d\u044b\u0435 \u043c\u0435\u0442\u043e\u0434\u044b", None))
        self.model2.setText(QCoreApplication.translate("ND", u"\u041d\u0435\u0439\u0440\u043e\u0441\u0435\u0442\u044c", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("ND", u"\u041c\u0435\u0442\u043e\u0434 \u043e\u043f\u0442\u0438\u043c\u0438\u0437\u0430\u0446\u0438\u0438", None))
        self.opt1.setText(QCoreApplication.translate("ND", u"\u0411\u044b\u0441\u0442\u0440\u044b\u0439 \u0441\u043f\u0443\u0441\u043a", None))
        self.opt2.setText(QCoreApplication.translate("ND", u"\u041e\u0442\u0436\u0438\u0433 \u0411\u043e\u043b\u044c\u0446\u043c\u0430\u043d\u0430", None))
        self.label.setText(QCoreApplication.translate("ND", u"\u0421\u0442\u0430\u0442\u0443\u0441 \u0432\u044b\u043f\u043e\u043b\u043d\u0435\u043d\u0438\u044f", None))
        self.label_2.setText(QCoreApplication.translate("ND", u"\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u0438\u0442\u043e\u0433\u043e\u0432\u043e\u0433\u043e \u043c\u043e\u0434\u0435\u043b\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f", None))
    # retranslateUi
