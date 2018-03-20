import sys
from PyQt4 import QtGui

app = QtGui.QApplication([])

window = QtGui.QWidget()
window.setGeometry(50,50,300,300)
window.setWindowTitle('Astronomy')


window.show()
sys.exit(app.exec_())
