#File for the editor window controls what happens with mouse movements and the mode the editor is in
#Widgets like the buttons intialized in the Main script are placed here
#Drawing objects are also spawned on this canvas

#PyQt5 Drawing resources 
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsLineItem
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPen, QPainter

from stroke_item import StrokeItem
from drawing import Drawing  # required to build a new Drawing from strokes

class SketchEditor(QGraphicsView):
    
    #class constructor
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)

        self.mode = "select"  # or "draw"
        self.drawing = False      # Flag to track if mouse is drawing
        self.last_point = None    # Store the last point for continuous lines
        self.pen = QPen(Qt.black, 3)

        # 🔥 Allow multi-selection
        self.setDragMode(QGraphicsView.RubberBandDrag)

        self.drawings = []
        self.selected_drawing = None

        # 🖊️ Stroke capture during freehand drawing
        self.current_stroke_points = []  # one stroke's raw points
        self.freehand_strokes = []       # list of strokes (for multi-stroke drawings)
        self.temp_lines = []             # temporary pen marks to be cleared

    def toggle_mode(self):
        if self.mode == "draw":
            self.mode = "select"
            print("Switched to SELECT mode")
        else:
            self.mode = "draw"
            print("Switched to DRAW mode")

    def select_drawing_by_name(self, name):
        for drawing in self.drawings:
            if drawing.label.lower() == name.lower():
                if self.selected_drawing and self.selected_drawing != drawing:
                    self.selected_drawing.unhighlight()
                self.selected_drawing = drawing
                self.selected_drawing.highlight()
                print(f"[INFO] Drawing '{name}' selected.")
                return
        print(f"[WARN] No drawing found with label: {name}")

    # Mouse events
    def mousePressEvent(self, event):
        if self.mode == "draw":
            if event.button() == Qt.LeftButton:
                self.drawing = True
                self.last_point = self.mapToScene(event.pos())
                self.current_stroke_points = [self.last_point]
        else:
            # In SELECT mode:
            item = self.itemAt(event.pos())
            if item and isinstance(item, StrokeItem):
                # Find which Drawing this StrokeItem belongs to
                for drawing in self.drawings:
                    if item in drawing.strokes:
                        
                        if self.selected_drawing and self.selected_drawing != drawing:
                            self.selected_drawing.unhighlight()

                        self.selected_drawing = drawing
                        self.selected_drawing.highlight()
                        print(f"Selected drawing: {drawing.label}")
                        break
                else:
                    if self.selected_drawing:
                        self.selected_drawing.unhighlight()
                    self.selected_drawing = None
                    print("No drawing selected.")
            else:
                if self.selected_drawing:
                    self.selected_drawing.unhighlight()
                self.selected_drawing = None
                print("No drawing selected.")

            # Always pass event up to allow item dragging/moving
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.mode == "draw" and self.drawing:
            current_point = self.mapToScene(event.pos())
            line = QGraphicsLineItem(self.last_point.x(), self.last_point.y(), current_point.x(), current_point.y())
            line.setPen(self.pen)
            self.scene.addItem(line)
            self.temp_lines.append(line)  # track temporary pen mark
            self.last_point = current_point
            self.current_stroke_points.append(current_point)
        else:
            # Allow item dragging/selection
            super().mouseMoveEvent(event)
            #move label
            if self.selected_drawing:
                self.selected_drawing.update_label_position()

    def mouseReleaseEvent(self, event):
        if self.mode == "draw":
            if event.button() == Qt.LeftButton:
                self.drawing = False

                # Convert raw QPoints to (x, y)
                stroke_xy = [(p.x(), p.y()) for p in self.current_stroke_points]
                x_list = [x for x, y in stroke_xy]
                y_list = [y for x, y in stroke_xy]

                if x_list and y_list:
                    self.freehand_strokes.append([x_list, y_list])

                self.current_stroke_points.clear()

        else:
            super().mouseReleaseEvent(event)
            if self.selected_drawing:
                self.selected_drawing.update_label_position()

    def keyPressEvent(self, event):
        if self.mode == "draw" and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.freehand_strokes:
                label = f"user_drawing_{len(self.drawings)+1}"
                new_drawing = Drawing(label=label)
                new_drawing.build_from_quickdraw(self.freehand_strokes)
                new_drawing.add_to_scene(self.scene)
                self.drawings.append(new_drawing)
                print(f"[INFO] Finalized drawing '{label}' with {len(self.freehand_strokes)} stroke(s).")
                self.freehand_strokes.clear()

                # 🧹 Clean up temporary pen marks after drawing is finalized
                for line in self.temp_lines:
                    self.scene.removeItem(line)
                self.temp_lines.clear()
            else:
                print("[INFO] No strokes to finalize.")
        else:
            super().keyPressEvent(event)

    def select_drawing_by_item(self, clicked_item):
        for drawing in self.drawings:
            if clicked_item in drawing.strokes:
                if self.selected_drawing and self.selected_drawing != drawing:
                    self.selected_drawing.unhighlight()
                self.selected_drawing = drawing
                self.selected_drawing.highlight()
                print(f"Selected drawing: {drawing.label}")
                return
        if self.selected_drawing:
            self.selected_drawing.unhighlight()
        self.selected_drawing = None
        print("No drawing selected.")
