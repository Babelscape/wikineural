from src.pl_modules.model import MyModel
from src.ui.ui_utils import select_checkpoint, get_model

checkpoint_path = select_checkpoint()
model: MyModel = get_model(checkpoint_path=checkpoint_path)
