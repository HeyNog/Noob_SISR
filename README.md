# Noob_SISR
Noob_SISR, Pytorch project

1, Design your SISR method:

	/Noob_SISR/models/SISR_MODELS/your_method.py  =>  Class Your_model(nn.Module): ...
	
2, Import your design:

	/Noob_SISR/models/SISR_MODELS/__init__.py  =>  from .your_method import *
	
3, Application of your design:

	import models.build_model as M
	
	model = M.build_model("Your_model_1")
	
	# State dict saved in /Noob_SISR/models/torch_models_save/Your_model_1/
