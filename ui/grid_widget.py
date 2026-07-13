from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from core.action import Action
from core.world_object import Candy, Empty, Goal, Slippery, Trap, Wall


class GridWidget(QWidget):
    map_changed = Signal()

    def __init__(self, world, agent, cell_size: int = 40):
        super().__init__()

        self.world = world
        self.agent = agent
        self.cell_size = cell_size

        self.edit_mode = False
        self.editor_margin = 36
        self.edit_button_size = 34

        self.update_widget_size()

    def update_widget_size(self):
        margin = self.editor_margin if self.edit_mode else 0

        width = margin + self.world.width * self.cell_size + 50
        height = margin + self.world.height * self.cell_size + 50

        self.setMinimumSize(width, height)
        self.resize(width, height)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.edit_mode:
            self.draw_editor_controls(painter)

        for row_index, row in enumerate(self.world.map):
            for col_index, cell in enumerate(row):
                self.draw_cell(painter, row_index, col_index, cell)

        self.draw_agent(painter)
        self.draw_edit_button(painter)

    def draw_cell(self, painter: QPainter, row: int, col: int, cell):
        painter.save()

        x, y = self.cell_to_pixel(row, col)
        size = self.cell_size

        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.NoBrush)
        painter.fillRect(x, y, size, size, self.get_cell_color(cell))

        pen = QPen(QColor(70, 70, 70))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        painter.drawRect(x, y, size, size)
        painter.drawLine(x, y, x + size, y + size)
        painter.drawLine(x + size, y, x, y + size)

        state = (row, col)
        self.agent.init_state(state)
        q_values = self.agent.q_table[state]

        painter.setPen(QColor(30, 30, 30))
        painter.setBrush(Qt.NoBrush)

        self.draw_centered_text(
            painter,
            f"{q_values[Action.UP]:.1f}",
            x + size / 2,
            y + size / 6,
        )
        self.draw_centered_text(
            painter,
            f"{q_values[Action.DOWN]:.1f}",
            x + size / 2,
            y + size * 5 / 6,
        )
        self.draw_centered_text(
            painter,
            f"{q_values[Action.LEFT]:.1f}",
            x + size / 6,
            y + size / 2,
        )
        self.draw_centered_text(
            painter,
            f"{q_values[Action.RIGHT]:.1f}",
            x + size * 5 / 6,
            y + size / 2,
        )

        painter.restore()

    def draw_editor_controls(self, painter: QPainter):
        painter.save()

        margin = self.editor_margin
        size = self.cell_size

        for col in range(self.world.width):
            x = margin + col * size + size // 2
            y = 14
            self.draw_round_button(painter, x, y, QColor(220, 80, 80), "-")

        for col_index in range(self.world.width + 1):
            x = margin + col_index * size
            y = margin - 8
            self.draw_round_button(painter, x, y, QColor(80, 170, 100), "+")

        for row in range(self.world.height):
            x = 14
            y = margin + row * size + size // 2
            self.draw_round_button(painter, x, y, QColor(220, 80, 80), "-")

        for row_index in range(self.world.height + 1):
            x = margin - 8
            y = margin + row_index * size
            self.draw_round_button(painter, x, y, QColor(80, 170, 100), "+")

        painter.restore()

    def draw_round_button(
        self,
        painter: QPainter,
        center_x: int,
        center_y: int,
        color: QColor,
        text: str,
    ):
        painter.save()

        radius = 12

        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(
            center_x - radius,
            center_y - radius,
            radius * 2,
            radius * 2,
        )

        painter.setPen(QColor(255, 255, 255))
        painter.setBrush(Qt.NoBrush)

        font = painter.font()
        font.setBold(True)
        painter.setFont(font)

        rect = QRect(
            center_x - radius,
            center_y - radius,
            radius * 2,
            radius * 2,
        )

        painter.drawText(rect, Qt.AlignCenter, text)

        painter.restore()

    def draw_edit_button(self, painter: QPainter):
        painter.save()

        rect = self.get_edit_button_rect()
        color = QColor(55, 100, 160) if self.edit_mode else QColor(100, 100, 100)

        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(rect)

        painter.setPen(QColor(255, 255, 255))
        painter.setBrush(Qt.NoBrush)
        self.draw_centered_text(painter, "✎", rect.center().x(), rect.center().y())

        painter.restore()

    def get_edit_button_rect(self) -> QRect:
        x = self.width() - self.edit_button_size - 12
        y = self.height() - self.edit_button_size - 12

        return QRect(x, y, self.edit_button_size, self.edit_button_size)

    def draw_agent(self, painter: QPainter):
        painter.save()

        row, col = self.world.current_agent_position
        x, y = self.cell_to_pixel(row, col)

        center_x = x + self.cell_size // 2
        center_y = y + self.cell_size // 2

        painter.setBrush(QColor(45, 90, 150))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center_x - 8, center_y - 8, 16, 16)

        painter.restore()

    def draw_centered_text(
        self,
        painter: QPainter,
        text: str,
        center_x: float,
        center_y: float,
    ):
        font_metrics = painter.fontMetrics()
        text_rect = font_metrics.boundingRect(text)

        text_x = int(center_x - text_rect.width() / 2)
        text_y = int(center_y + text_rect.height() / 2)

        painter.drawText(text_x, text_y, text)

    def mousePressEvent(self, event):
        pos = event.position().toPoint()

        if event.button() == Qt.LeftButton:
            if self.get_edit_button_rect().contains(pos):
                self.edit_mode = not self.edit_mode
                self.update_widget_size()
                self.update()
                return

            if not self.edit_mode:
                return

            clicked = self.handle_editor_click(pos.x(), pos.y())

            if clicked:
                self.agent.q_table.clear()
                self.map_changed.emit()
                self.update_widget_size()
                self.update()
                return

            cell_position = self.pixel_to_cell(pos.x(), pos.y())

            if cell_position is not None:
                self.cycle_cell_type(cell_position)
                self.agent.q_table.clear()
                self.map_changed.emit()
                self.update()
                return

        if event.button() == Qt.RightButton:
            if not self.edit_mode:
                return

            cell_position = self.pixel_to_cell(pos.x(), pos.y())

            if cell_position is not None:
                self.world.set_start_position(cell_position)
                self.agent.q_table.clear()
                self.map_changed.emit()
                self.update()

    def handle_editor_click(self, mouse_x: int, mouse_y: int) -> bool:
        margin = self.editor_margin
        size = self.cell_size

        for col in range(self.world.width):
            center_x = margin + col * size + size // 2
            center_y = 14

            if self.is_inside_circle(mouse_x, mouse_y, center_x, center_y, 12):
                self.world.remove_column(col)
                return True

        for col_index in range(self.world.width + 1):
            center_x = margin + col_index * size
            center_y = margin - 8

            if self.is_inside_circle(mouse_x, mouse_y, center_x, center_y, 12):
                self.world.insert_column(col_index)
                return True

        for row in range(self.world.height):
            center_x = 14
            center_y = margin + row * size + size // 2

            if self.is_inside_circle(mouse_x, mouse_y, center_x, center_y, 12):
                self.world.remove_row(row)
                return True

        for row_index in range(self.world.height + 1):
            center_x = margin - 8
            center_y = margin + row_index * size

            if self.is_inside_circle(mouse_x, mouse_y, center_x, center_y, 12):
                self.world.insert_row(row_index)
                return True

        return False

    def is_inside_circle(
        self,
        mouse_x: int,
        mouse_y: int,
        center_x: int,
        center_y: int,
        radius: int,
    ) -> bool:
        return (mouse_x - center_x) ** 2 + (mouse_y - center_y) ** 2 <= radius ** 2

    def cell_to_pixel(self, row: int, col: int) -> tuple[int, int]:
        margin = self.editor_margin if self.edit_mode else 0

        x = margin + col * self.cell_size
        y = margin + row * self.cell_size

        return x, y

    def get_cell_color(self, cell):
        if isinstance(cell, Wall):
            return QColor(150, 150, 150)

        if isinstance(cell, Goal):
            return QColor(200, 230, 200)
        
        if isinstance(cell, Candy):
            return QColor(245, 225, 170)

        if isinstance(cell, Trap):
            return QColor(235, 200, 200)
        
        if isinstance(cell, Slippery):
            return QColor(180, 200, 255)

        if isinstance(cell, Empty):
            return QColor(250, 250, 250)

        return QColor(250, 250, 250)
    
    def pixel_to_cell(self, mouse_x: int, mouse_y: int):
        margin = self.editor_margin if self.edit_mode else 0

        grid_x = mouse_x - margin
        grid_y = mouse_y - margin

        if grid_x < 0 or grid_y < 0:
            return None

        col = grid_x // self.cell_size
        row = grid_y // self.cell_size

        if row < 0 or row >= self.world.height:
            return None

        if col < 0 or col >= self.world.width:
            return None

        return int(row), int(col)


    def cycle_cell_type(self, position: tuple[int, int]):
        row, col = position
        cell = self.world.map[row][col]

        if isinstance(cell, Empty):
            self.world.set_cell(position, Wall())
            return

        if isinstance(cell, Wall):
            self.world.set_cell(position, Trap())
            return

        if isinstance(cell, Trap):
            self.world.set_cell(position, Candy())
            return
        
        if isinstance(cell, Candy):
            self.world.set_cell(position, Slippery())
            return

        if isinstance(cell, Slippery):
            self.world.set_cell(position, Goal())
            return

        if isinstance(cell, Goal):
            self.world.set_cell(position, Empty())
            return

        self.world.set_cell(position, Empty())