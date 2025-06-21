from PyQt5.QtGui import QImage, QPainter, QColor
from PyQt5.QtCore import QRectF 
import numpy as np

from stroke_item import StrokeItem
#from PyQt5.QtWidgets import QGraphicsItemGroup
from PyQt5.QtWidgets import QGraphicsSimpleTextItem

class Drawing:
    def __init__(self, label="Unknown"):
        self.label = label
        self.strokes = []  # List of StrokeItem objects
        self.label_item = None  # Placeholder for the floating label

    def build_from_quickdraw(self, strokes, offset_x=0, offset_y=0, scale=1):
        for stroke in strokes:
            x_points, y_points = stroke
            points = []
            for i in range(len(x_points)):
                x = offset_x + x_points[i] * scale
                y = offset_y + y_points[i] * scale
                points.append((x, y))

            stroke_item = StrokeItem(points)
            self.strokes.append(stroke_item)

    def add_to_scene(self, scene):
        for stroke in self.strokes:
            scene.addItem(stroke)

        # Create and add label when first added to scene
        self.label_item = QGraphicsSimpleTextItem(self.label)
        self.label_item.setVisible(False)  # Start invisible
        scene.addItem(self.label_item)
        self.update_label_position()

    def delete_from_scene(self, scene):
        # Remove strokes from the scene
        for stroke in self.strokes:
            scene.removeItem(stroke)
        self.strokes.clear()

        # Remove label from the scene
        if self.label_item:
            scene.removeItem(self.label_item)
            self.label_item = None

    def move_by(self, dx, dy):
        
        for stroke in self.strokes:
            stroke.moveBy(dx, dy)
        #if self.label_item:
        #    self.label_item.moveBy(dx, dy)
        
        print(f"[DEBUG] Stroke 0 new pos: {self.strokes[0].scenePos()}")

        #After moving, reposition label
        self.update_label_position()

    def update_label_position(self):
        """Position the label above the first stroke."""
        if self.strokes and self.label_item:
            # Get position of the first stroke
            bbox = self.strokes[0].boundingRect()
            scene_pos = self.strokes[0].scenePos()

            x = scene_pos.x() + bbox.x()
            y = scene_pos.y() + bbox.y() - 30  # Slightly above
            self.label_item.setPos(x, y)

    def highlight(self):
        for stroke in self.strokes:
            stroke.highlight()
        if self.label_item:
            self.label_item.setVisible(True)  # Show label when highlighted
            self.update_label_position()

    def unhighlight(self):
        for stroke in self.strokes:
            stroke.unhighlight()
        if self.label_item:
            self.label_item.setVisible(False)  # Hide label when unselected


    #Export for use in a model
    def export_strokes(self):
        """Export strokes in [ [x points], [y points] ] format for model input."""
        all_strokes = []
        for stroke in self.strokes:
            path = stroke.path()
            points = []
            for i in range(int(path.elementCount())):
                elem = path.elementAt(i)
                points.append((elem.x, elem.y))
            
            if points:
                x_points = [p[0] for p in points]
                y_points = [p[1] for p in points]
                all_strokes.append([x_points, y_points])

        return all_strokes
    
    def normalize_strokes(strokes, size=256):
        all_x = []
        all_y = []
        for stroke in strokes:
            all_x.extend(stroke[0])
            all_y.extend(stroke[1])

        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)

        scale = size / max(max_x - min_x, max_y - min_y)

        normalized = []
        for stroke in strokes:
            x_points = [(x - min_x) * scale for x in stroke[0]]
            y_points = [(y - min_y) * scale for y in stroke[1]]
            normalized.append([x_points, y_points])

        return normalized

    def prepare_for_model(self):
        strokes = self.export_strokes()

        # Normalize strokes to a fixed scale (e.g., 256x256 canvas)
        normalized_strokes = Drawing.normalize_strokes(strokes, size=256)

        flat_points = []
        for stroke in normalized_strokes:
            x_points, y_points = stroke
            for x, y in zip(x_points, y_points):
                # Clamp values to [0, 255] for safety
                x = max(0.0, min(255.0, x))
                y = max(0.0, min(255.0, y))
                flat_points.append(x)
                flat_points.append(y)

        # Optionally pad if model expects fixed length input
        desired_length = 512

        # Normalize to [0, 1] to match transformer input expectations
        flat_points = [p / 255.0 for p in flat_points]

        if len(flat_points) < desired_length:
            flat_points.extend([0.0] * (desired_length - len(flat_points)))  # Padding
        else:
            flat_points = flat_points[:desired_length]  # Truncate if too long

        return flat_points

    
from PyQt5.QtGui import QImage, QPainter, QColor
import numpy as np

def strokes_to_image(strokes, canvas_size=28, normalize=True):
    """Convert strokes [[x], [y]] into a 28x28 grayscale numpy array with scaling and centering."""

    # Flatten all points to compute bounding box
    all_x, all_y = [], []
    for stroke in strokes:
        all_x.extend(stroke[0])
        all_y.extend(stroke[1])

    if not all_x or not all_y:
        return np.ones((canvas_size, canvas_size), dtype=np.float32)  # blank white

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    width = max_x - min_x
    height = max_y - min_y
    scale = (canvas_size - 4) / max(width, height)  # Add margin

    # Compute offsets to center the drawing
    offset_x = (canvas_size - scale * (min_x + max_x)) / 2
    offset_y = (canvas_size - scale * (min_y + max_y)) / 2

    # Create blank image
    image = QImage(canvas_size, canvas_size, QImage.Format_Grayscale8)
    image.fill(QColor(255, 255, 255))  # white

    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QColor(0, 0, 0))  # black pen

    # Draw all strokes with new scaling
    for stroke in strokes:
        x_points, y_points = stroke
        for i in range(1, len(x_points)):
            x1 = int(x_points[i - 1] * scale + offset_x)
            y1 = int(y_points[i - 1] * scale + offset_y)
            x2 = int(x_points[i] * scale + offset_x)
            y2 = int(y_points[i] * scale + offset_y)
            painter.drawLine(x1, y1, x2, y2)

    painter.end()

    # Convert QImage to numpy array
    ptr = image.bits()
    ptr.setsize(image.byteCount())
    arr = np.array(ptr).reshape((canvas_size, canvas_size)).astype(np.float32)

    arr /= 255.0  # normalize to [0, 1]

    if normalize:
        arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]

    return arr

def strokes_to_continuous_format(strokes):
    points = []
    for stroke in strokes:
        x_list, y_list = stroke
        for i in range(len(x_list)):
            x = x_list[i]
            y = y_list[i]
            # p1 = pen_down, p2 = pen_up, p3 = end_of_stroke
            p1 = 1 if i == 0 else 0
            p2 = 0
            p3 = 1 if i == len(x_list) - 1 else 0
            points.append([x, y, p1, p2, p3])
    return points


