#PyQt5 Drawing resources 
from PyQt5.QtWidgets import  QGraphicsPathItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QPainterPath

class StrokeItem(QGraphicsPathItem):
    def __init__(self, stroke_points, pen_color=Qt.black, pen_width=3, parent=None):
        super().__init__(parent)
        
        # Create the stroke path
        path = QPainterPath()
        if stroke_points:
            first_point = stroke_points[0]
            path.moveTo(first_point[0], first_point[1])
            for x, y in stroke_points[1:]:
                path.lineTo(x, y)

        self.setPath(path)

        # Save the pens
        self.default_pen = QPen(pen_color, pen_width)             # Normal color
        self.highlight_pen = QPen(Qt.blue, pen_width + 1)          # Highlight color (blue and slightly thicker)
        self.setPen(self.default_pen)  # Set to default at start

        # Make item movable and selectable
        self.setFlags(
            self.ItemIsMovable |
            self.ItemIsSelectable
        )
    
    def highlight(self):
        self.setPen(self.highlight_pen)

    def unhighlight(self):
        self.setPen(self.default_pen)
