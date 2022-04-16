import numpy as np

from flax_models.vgg import build_vgg19


model, parameters = build_vgg19(show_parameter_overview=True, include_top=False)
x = np.random.normal(size=(1, 224, 224, 3))
out = model.apply(parameters, x)
print(out.shape)
