"""Project explorer and guided actions for KuffGravMag Studio."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class StudioWorkspace(QWidget):
    """A small project tree with actions wired to the established workflows."""

    action_requested = pyqtSignal(str)
    item_activated = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("studioProjectExplorer")
        self.setMinimumWidth(230)
        root = QVBoxLayout(self)
        self.project_label = QLabel("Untitled project")
        self.project_label.setObjectName("projectNameLabel")
        self.project_label.setWordWrap(True)
        root.addWidget(self.project_label)

        self.state_label = QLabel("Saved")
        self.state_label.setObjectName("projectSaveStateLabel")
        root.addWidget(self.state_label)

        self.tree = QTreeWidget()
        self.tree.setObjectName("projectExplorerTree")
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(16)
        self.tree.itemDoubleClicked.connect(self._activate_item)
        root.addWidget(self.tree, stretch=1)

        actions = QGroupBox("Workflow")
        action_layout = QVBoxLayout(actions)
        self.buttons = {}
        for label, action in (
            ("Import observed profile…", "import_observations"),
            ("Process survey profile…", "process_survey"),
            ("Interpret regular grid…", "interpret_grid"),
            ("Academic & Research Laboratory…", "academic_lab"),
            ("Load local XYZ spatial layer…", "load_spatial_layer"),
            ("Add model body", "add_body"),
            ("Run gravity inversion", "invert_gravity"),
            ("Run magnetic inversion", "invert_magnetic"),
            ("Export calculated CSV…", "export_results"),
            ("Generate reproducible report…", "generate_report"),
        ):
            button = QPushButton(label)
            button.setObjectName(f"workflow_{action}")
            button.clicked.connect(lambda checked=False, key=action: self.action_requested.emit(key))
            action_layout.addWidget(button)
            self.buttons[action] = button
        root.addWidget(actions)
        self.update_project()

    def update_project(
        self,
        *,
        project_name="Untitled project",
        project_path=None,
        dirty=False,
        bodies=(),
        active_body_index=0,
        survey_profile=None,
        observation_count=0,
        inversion_ready=False,
        processing_steps=(),
        grid_name=None,
        spatial_layer_name=None,
    ):
        self.project_label.setText(project_name)
        self.state_label.setText("Unsaved changes" if dirty else "All changes saved")
        self.state_label.setProperty("dirty", bool(dirty))
        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)
        self.tree.clear()

        root = QTreeWidgetItem([project_name])
        root.setData(0, Qt.ItemDataRole.UserRole, "project")
        self.tree.addTopLevelItem(root)
        survey_count = 0 if survey_profile is None else int(survey_profile.distance_m.size)
        observations = int(observation_count)
        if survey_count:
            survey_text = f"Survey profile · {survey_count} stations"
        elif observations:
            survey_text = f"Observed profile · {observations} stations"
        else:
            survey_text = "Survey & observations · none loaded"
        survey = QTreeWidgetItem([survey_text])
        survey.setData(0, Qt.ItemDataRole.UserRole, "survey")
        root.addChild(survey)

        processing = QTreeWidgetItem([f"Processing history · {len(processing_steps)} steps"])
        processing.setData(0, Qt.ItemDataRole.UserRole, "processing")
        root.addChild(processing)
        for step in processing_steps[-8:]:
            if isinstance(step, dict):
                processing.addChild(QTreeWidgetItem([str(step.get("name", "Processing step"))]))

        model = QTreeWidgetItem([f"Subsurface model · {len(bodies)} bodies"])
        model.setData(0, Qt.ItemDataRole.UserRole, "model")
        root.addChild(model)
        for index, body in enumerate(bodies):
            active = "  • active" if index == active_body_index else ""
            body_item = QTreeWidgetItem([
                f"{body.name} · {body.strike_mode} strike{active}"
            ])
            body_item.setData(0, Qt.ItemDataRole.UserRole, ("body", index))
            model.addChild(body_item)

        grid_label = f"Grid interpretation · {grid_name}" if grid_name else "Grid interpretation · none loaded"
        grid_item = QTreeWidgetItem([grid_label])
        grid_item.setData(0, Qt.ItemDataRole.UserRole, "grid")
        root.addChild(grid_item)
        spatial_label = (
            f"Local spatial layer · {spatial_layer_name}"
            if spatial_layer_name else "Local spatial layer · none loaded"
        )
        spatial_item = QTreeWidgetItem([spatial_label])
        spatial_item.setData(0, Qt.ItemDataRole.UserRole, "spatial_layer")
        root.addChild(spatial_item)
        reports_item = QTreeWidgetItem(["Reports & exports"])
        reports_item.setData(0, Qt.ItemDataRole.UserRole, "reports")
        root.addChild(reports_item)
        root.setExpanded(True)
        model.setExpanded(True)
        processing.setExpanded(bool(processing_steps))
        self.tree.resizeColumnToContents(0)

        self.buttons["invert_gravity"].setEnabled(bool(bodies) and bool(inversion_ready))
        self.buttons["invert_magnetic"].setEnabled(bool(bodies) and bool(inversion_ready))

    def _activate_item(self, item, _column):
        self.item_activated.emit(item.data(0, Qt.ItemDataRole.UserRole))
